"""Bounded synthetic orthographic patterns and local Kraken fine-tuning.

Synthetic labels are not linguistic ground truth. They contain independent
vowels, consonants with at most one vowel sign, optional anusvara, and boundary
punctuation. Conjuncts, nukta, shadda, sukun, and abbreviation usage require
reviewed examples and are deliberately excluded. No real text is used as a
synthetic prompt; gold/test data must never be supplied as training input.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from khojki_ocr.core import (
    KHOJKI_CODEPOINTS, POLICY_VERSION, normalize_text, raster_digest, validate_text,
)


INDEPENDENT_VOWELS = (frozenset(range(0x11200, 0x11208)) | {0x11240}) & KHOJKI_CODEPOINTS
CONSONANTS = (frozenset(range(0x11208, 0x1122C)) | {0x1123F}) & KHOJKI_CODEPOINTS
VOWEL_SIGNS = (frozenset(range(0x1122C, 0x11234)) | {0x11241}) & KHOJKI_CODEPOINTS
ANUSVARA = 0x11234
PUNCTUATION = frozenset(range(0x11238, 0x1123D)) & KHOJKI_CODEPOINTS
BASIC_ALPHABET = INDEPENDENT_VOWELS | CONSONANTS | VOWEL_SIGNS | PUNCTUATION | {ANUSVARA}
PATTERN_EXCLUSIONS = KHOJKI_CODEPOINTS - BASIC_ALPHABET
MAX_SYNTHETIC_LINES = 10000
MAX_SYNTHETIC_BYTES = 512 * 1024**2
MIN_FREE_BYTES = 2 * 1024**3


def line_text(text: str) -> str:
    """Apply the shared NFC policy without changing spaces or punctuation."""
    normalized = validate_text(text)
    if "\n" in normalized or not normalized.strip():
        raise ValueError("Expected one nonempty Khojki line, without line breaks")
    if not any(ord(character) in CONSONANTS | INDEPENDENT_VOWELS for character in normalized):
        raise ValueError("A training line must contain a Khojki base letter")
    return normalized


def generate_splits(
    count: int = 1000,
    seed: int = 42,
    validation_fraction: float = 0.1,
    alphabet: frozenset[int] = BASIC_ALPHABET,
) -> tuple[list[str], list[str]]:
    """Generate distinct basic patterns, with supported alphabet coverage in train."""
    if not 20 <= count <= MAX_SYNTHETIC_LINES:
        raise ValueError(f"count must be between 20 and {MAX_SYNTHETIC_LINES}")
    if not 0.05 <= validation_fraction <= 0.25:
        raise ValueError("validation_fraction must be between 0.05 and 0.25")
    if not alphabet <= BASIC_ALPHABET:
        raise ValueError("Synthetic alphabet contains excluded or unassigned code points")
    consonants = sorted(alphabet & CONSONANTS)
    independent = sorted(alphabet & INDEPENDENT_VOWELS)
    vowels = sorted(alphabet & VOWEL_SIGNS)
    punctuation = sorted(alphabet & PUNCTUATION)
    if not consonants or not independent:
        raise ValueError("Font must support consonants and independent vowels")
    randomizer = random.Random(seed)
    coverage = []
    for codepoint in sorted(alphabet):
        if codepoint in VOWEL_SIGNS or codepoint == ANUSVARA:
            coverage.append(chr(consonants[0]) + chr(codepoint))
        elif codepoint in PUNCTUATION:
            coverage.append(chr(independent[0]) + chr(codepoint))
        else:
            coverage.append(chr(codepoint))
    randomizer.shuffle(coverage)
    labels = []
    seen = set()
    for index in range(count):
        prefix = coverage[index * 6:(index + 1) * 6]
        for _ in range(1000):
            tokens = list(prefix)
            for _ in range(randomizer.randint(6, 12)):
                if randomizer.random() < 0.2:
                    token = chr(randomizer.choice(independent))
                else:
                    token = chr(randomizer.choice(consonants))
                    if vowels and randomizer.random() < 0.6:
                        token += chr(randomizer.choice(vowels))
                if ANUSVARA in alphabet and randomizer.random() < 0.08:
                    token += chr(ANUSVARA)
                tokens.append(token)
            text = " ".join(tokens)
            if punctuation and randomizer.random() < 0.3:
                text += chr(randomizer.choice(punctuation))
            text = line_text(text)
            if text not in seen:
                seen.add(text)
                labels.append(text)
                break
        else:
            raise ValueError("Unable to generate enough unique synthetic content")
    validation_count = max(1, round(count * validation_fraction))
    training = labels[:-validation_count]
    validation = labels[-validation_count:]
    if not alphabet <= {ord(character) for text in training for character in text}:
        raise ValueError("Not enough training lines for complete supported alphabet coverage")
    return training, validation


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")


def write_input_contract(weights: Path, max_width: int, padding: int = 16) -> Path:
    if type(max_width) is not int or not 64 <= max_width <= 4096:
        raise ValueError("Invalid trained tensor width")
    if type(padding) is not int or padding != 16:
        raise ValueError("This inference adapter requires training padding of 16 pixels")
    path = weights.with_suffix(".input.json")
    write_json(path, {"schema": "khojki-ppocrv6-input-v1", "model_sha256": file_sha256(weights),
                      "max_width": max_width, "padding": padding,
                      "operation": "right-pad normalized tensors with zeros; preserve sequence lengths"})
    return path


def free_bytes(path: Path) -> int:
    existing = path.resolve()
    while not existing.exists():
        existing = existing.parent
    return shutil.disk_usage(existing).free


def load_font(path: Path, size: int = 48):
    """Require explicit shaping support and inspect the actual supplied font."""
    try:
        from fontTools.ttLib import TTFont
        from PIL import ImageFont, features
    except ImportError as error:
        raise RuntimeError("Synthesis requires Pillow with RAQM/HarfBuzz and fonttools") from error
    if not path.is_file():
        raise FileNotFoundError(f"Missing local font: {path}")
    if not 24 <= size <= 96:
        raise ValueError("font_size must be between 24 and 96 pixels")
    if not features.check_feature("raqm"):
        raise RuntimeError(
            "Pillow lacks RAQM/HarfBuzz shaping. Ask the environment worker for a "
            "RAQM-enabled Pillow build or a verified Pango renderer; BASIC is unsafe for Khojki."
        )
    with TTFont(str(path), lazy=True) as inspected:
        cmap = inspected.getBestCmap() or {}
        supported = frozenset(codepoint for codepoint, glyph in cmap.items()
                              if glyph != ".notdef" and inspected.getGlyphID(glyph) != 0)
        if "GSUB" not in inspected:
            raise ValueError("Font has no GSUB shaping table; refusing unshaped Khojki")
        scripts = inspected["GSUB"].table.ScriptList
        if scripts is None or not {record.ScriptTag for record in scripts.ScriptRecord} & {"khoj", "DFLT"}:
            raise ValueError("Font has no Khojki/default OpenType shaping script")
    if 32 not in supported:
        raise ValueError("Font lacks the space glyph")
    font = ImageFont.truetype(str(path), size=size, layout_engine=ImageFont.Layout.RAQM)
    if font.layout_engine != ImageFont.Layout.RAQM:
        raise RuntimeError("Pillow did not activate the requested RAQM layout engine")
    renderer = {
        "name": "Pillow-RAQM", "direction": "ltr", "font_size": size,
        "raqm_version": features.version_feature("raqm"),
        "harfbuzz_version": features.version_feature("harfbuzz"),
        "freetype_version": features.version_module("freetype2"),
        "visual_review": "required; cmap and RAQM checks do not certify linguistic or conjunct correctness",
    }
    return font, supported, renderer


def render_line(text: str, font, randomizer: random.Random | None = None):
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    text = line_text(text)
    margin = 16
    left, top, right, bottom = font.getbbox(text, direction="ltr")
    width, height = right - left + 2 * margin, bottom - top + 2 * margin
    if not 1 <= width <= 2560 or not 1 <= height <= 256 or width * 96 / height > 2560:
        raise ValueError("Rendered line exceeds the bounded PP-OCRv6 line dimensions")
    background = randomizer.randint(239, 255) if randomizer else 255
    foreground = randomizer.randint(0, 32) if randomizer else 0
    image = Image.new("L", (width, height), background)
    ImageDraw.Draw(image).text((margin - left, margin - top), text,
                              font=font, fill=foreground, direction="ltr")
    if randomizer:
        image = ImageEnhance.Contrast(image).enhance(randomizer.uniform(0.85, 1.05))
        image = image.filter(ImageFilter.GaussianBlur(randomizer.uniform(0, 0.45)))
    minimum, maximum = image.getextrema()
    if maximum - minimum < 32:
        raise ValueError("Rendered line is blank or has insufficient contrast")
    return image


def synthesize(
    output: Path,
    font_path: Path,
    count: int = 1000,
    seed: int = 42,
    font_size: int = 48,
    require_full_coverage: bool = False,
) -> dict:
    """Write new PNG/GT pairs and manifests; never overwrite an existing output."""
    output, font_path = output.resolve(), font_path.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new synthetic output directory: {output}")
    font, supported, renderer = load_font(font_path, font_size)
    missing = BASIC_ALPHABET - supported
    if require_full_coverage and missing:
        raise ValueError("Font lacks requested glyphs: " + ", ".join(f"U+{value:04X}" for value in sorted(missing)))
    alphabet = BASIC_ALPHABET & supported
    training, validation = generate_splits(count, seed, alphabet=alphabet)
    render_line(training[0], font)
    if free_bytes(output) < MIN_FREE_BYTES + MAX_SYNTHETIC_BYTES:
        raise OSError("Insufficient free disk space for bounded synthesis and a 2 GiB reserve")
    output.mkdir(parents=True, exist_ok=False)
    randomizer = random.Random(seed ^ 0x534E5448)
    records = []
    total_bytes = 0
    for split, labels in (("train", training), ("validation", validation)):
        split_directory = output / split
        split_directory.mkdir()
        image_paths = []
        for index, text in enumerate(labels):
            image_path = split_directory / f"line_{index:05d}.png"
            label_path = image_path.with_suffix(".gt.txt")
            image = render_line(text, font, randomizer if split == "train" else None)
            with image_path.open("xb") as stream:
                image.save(stream, format="PNG")
            with label_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            total_bytes += image_path.stat().st_size + label_path.stat().st_size
            if total_bytes > MAX_SYNTHETIC_BYTES or free_bytes(output) < MIN_FREE_BYTES:
                raise OSError("Synthetic disk budget reached; partial files retained, no complete manifest published")
            image_paths.append(str(image_path))
            records.append({
                "image": image_path.relative_to(output).as_posix(), "split": split,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "image_sha256": file_sha256(image_path),
                "size": list(image.size),
            })
        with (output / f"{split}.txt").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(image_paths) + "\n")
    metadata = {
        "schema": "khojki-synthetic-v1", "policy_version": POLICY_VERSION,
        "seed": seed, "count": count, "train_count": len(training),
        "validation_count": len(validation), "normalization": "NFC",
        "font_sha256": file_sha256(font_path), "font_path": str(font_path),
        "renderer": renderer, "used_alphabet": [f"U+{value:04X}" for value in sorted(alphabet)],
        "font_missing": [f"U+{value:04X}" for value in sorted(missing)],
        "pattern_exclusions": [f"U+{value:04X}" for value in sorted(PATTERN_EXCLUSIONS)],
        "text_source": "seeded basic orthographic patterns; no real/gold/test text input",
        "limitations": "Not linguistic ground truth or manuscript simulation. Conjuncts, nukta, shadda, sukun and abbreviations excluded.",
        "augmentation": {"train_only": True, "background": [239, 255],
                         "foreground": [0, 32], "contrast": [0.85, 1.05], "blur_radius": [0, 0.45]},
        "split_policy": "unique NFC content; supported alphabet covered in training; no corpus prompts",
        "bytes": total_bytes, "records": records,
    }
    write_json(output / "metadata.json", metadata)
    return metadata


def reject_held_out_path(path: Path) -> None:
    for part in path.resolve().parts:
        name = Path(part).stem.lower().replace("-", "_")
        if name in {"gold", "test", "heldout", "held_out"} or name.startswith(("gold_", "test_", "heldout_")):
            raise ValueError(f"Gold/test paths are forbidden as training or validation input: {path}")


def read_manifest(path: Path, limit: int = MAX_SYNTHETIC_LINES) -> list[dict]:
    """Read line-image manifests relative to their own directory, without edits."""
    from PIL import Image

    path = path.resolve()
    reject_held_out_path(path)
    records = []
    seen_paths = set()
    with path.open(encoding="utf-8") as stream:
        for entry in stream:
            entry = entry.rstrip("\r\n")
            if not entry:
                continue
            if len(records) >= limit:
                raise ValueError(f"Manifest exceeds the {limit}-line bound: {path}")
            image_path = Path(entry)
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            image_path = image_path.resolve()
            reject_held_out_path(image_path)
            if image_path in seen_paths:
                raise ValueError(f"Duplicate image path in manifest: {image_path}")
            seen_paths.add(image_path)
            label_path = image_path.with_suffix(".gt.txt")
            reject_held_out_path(label_path)
            text = line_text(normalize_text(label_path.read_text(encoding="utf-8")).removesuffix("\n"))
            with Image.open(image_path) as opened:
                width, height = opened.size
                if width * height > 4_000_000 or width < 1 or height < 1 or width * 96 / height > 2560:
                    raise ValueError(f"Input is not a bounded line image: {image_path}")
                image = opened.convert("RGB")
                minimum, maximum = image.convert("L").getextrema()
                if maximum - minimum < 8:
                    raise ValueError(f"Blank/near-blank image with a nonempty CTC label: {image_path}")
                digest = raster_digest(image.tobytes(), width, height)
            records.append({"image": str(image_path), "text": text, "raster_sha256": digest})
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def validate_manifests(
    train_manifest: Path,
    validation_manifest: Path,
    pilot: bool = False,
    page_groups: Path | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Reject cross-split content/raster leakage; page IDs must come from the parent."""
    training = read_manifest(train_manifest, 1000 if pilot else MAX_SYNTHETIC_LINES)
    validation = read_manifest(validation_manifest, 250 if pilot else MAX_SYNTHETIC_LINES)
    for field in ("image", "text", "raster_sha256"):
        if {record[field] for record in training} & {record[field] for record in validation}:
            raise ValueError(f"Training/validation overlap in {field}")
    train_alphabet = {ord(character) for record in training for character in record["text"]}
    validation_alphabet = {ord(character) for record in validation for character in record["text"]}
    if unseen := validation_alphabet - train_alphabet:
        raise ValueError("Validation glyphs absent from training codec: " + ", ".join(f"U+{value:04X}" for value in sorted(unseen)))
    page_separation = "not verified: parent must split real lines by source page and exclude gold/test pages"
    if page_groups is not None:
        with page_groups.open(encoding="utf-8") as stream:
            raw_groups = json.load(stream)
        if not isinstance(raw_groups, dict):
            raise ValueError("page_groups must map image paths to nonempty source-page IDs")
        groups = {}
        for name, group in raw_groups.items():
            if not isinstance(group, str) or not group.strip():
                raise ValueError("Every source-page ID must be a nonempty string")
            image_path = Path(name)
            if not image_path.is_absolute():
                image_path = page_groups.parent / image_path
            key = str(image_path.resolve())
            if key in groups and groups[key] != group:
                raise ValueError("Conflicting source-page IDs for the same image")
            groups[key] = group
        if any(record["image"] not in groups for record in training + validation):
            raise ValueError("page_groups is missing an input image")
        if {groups[record["image"]] for record in training} & {groups[record["image"]] for record in validation}:
            raise ValueError("Training/validation source-page overlap")
        page_separation = "verified disjoint supplied source-page IDs; gold exclusion remains caller responsibility"
    audit = {
        "train_count": len(training), "validation_count": len(validation),
        "train_manifest_sha256": file_sha256(train_manifest),
        "validation_manifest_sha256": file_sha256(validation_manifest),
        "page_separation": page_separation,
        "training_alphabet": [f"U+{value:04X}" for value in sorted(train_alphabet)],
    }
    return training, validation, audit


def build_ketos_command(
    train_manifest: Path,
    validation_manifest: Path,
    base_model: Path,
    output: Path,
    epochs: int = 1,
    threads: int = 4,
    batch_size: int = 4,
    seed: int = 2,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    """Kraken 7.1.1 ketos/recognition.py and ketos/__init__.py CLI contract."""
    if not 1 <= epochs <= 10:
        raise ValueError("epochs must be between 1 and 10 (Kraken retains ten checkpoints)")
    if not 1 <= threads <= 8 or not 1 <= batch_size <= 8:
        raise ValueError("threads and batch_size must be between 1 and 8 on this CPU budget")
    if not 0 <= seed < 2**32:
        raise ValueError("Training seed must be an unsigned 32-bit integer")
    weights = (["--resume", str(resume_checkpoint.resolve())] if resume_checkpoint is not None
               else ["--load", str(base_model), "--resize", "new"])
    return [
        sys.executable, str(Path(sys.executable).with_name("ketos")),
        "--device", "cpu", "--precision", "32-true", "--workers", "0",
        "--threads", str(threads), "--seed", str(seed), "--deterministic",
        "train", "--arch", "ppocrv6", "--variant", "small",
        *weights,
        "--training-data", str(train_manifest), "--evaluation-data", str(validation_manifest),
        "--format-type", "path", "--normalization", "NFC",
        "--no-normalize-whitespace", "--no-reorder", "--base-dir", "L",
        "--quit", "fixed", "--epochs", str(epochs), "--min-epochs", "1",
        "--batch-size", str(batch_size), "--freq", "1", "--height", "96", "--max-width", "2560",
        "--optimizer", "AdamW", "--lrate", "0.0001", "--schedule", "constant", "--warmup", "0", "--no-augment",
        "--output", str(output / "checkpoints"), "--weights-format", "safetensors",
        "--logger", "tensorboard", "--log-dir", str(output / "logs"),
    ]


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_training(
    train_manifest: Path,
    validation_manifest: Path,
    base_model: Path,
    output: Path,
    epochs: int = 1,
    threads: int = 4,
    batch_size: int = 4,
    seed: int = 2,
    pilot: bool = True,
    timeout_seconds: int = 1800,
    max_output_mib: int = 4096,
    page_groups: Path | None = None,
    resume_checkpoint: Path | None = None,
) -> dict:
    """Launch local ketos and retain measured diagnostics, including failed runs.

    This is deliberately fixed-epoch training, not gold-driven early stopping.
    Disk/time limits are monitored every two seconds and may overshoot by one
    write. Partial outputs are retained, never cleaned up automatically.
    """
    output, base_model = output.resolve(), base_model.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new training output directory: {output}")
    if pilot and epochs != 1:
        raise ValueError("A pilot is exactly one epoch; use train for longer runs")
    if not 1 <= timeout_seconds <= 7200 or not 64 <= max_output_mib <= 4096:
        raise ValueError("timeout_seconds must be 1..7200 and max_output_mib must be 64..4096")
    command = build_ketos_command(output / "train.txt", output / "validation.txt", base_model,
                                  output, epochs, threads, batch_size, seed, resume_checkpoint)
    if resume_checkpoint is not None and (not resume_checkpoint.is_file() or resume_checkpoint.suffix != ".ckpt"):
        raise ValueError("Resume requires a trusted local Kraken .ckpt checkpoint")
    if not base_model.is_file() or base_model.suffix != ".safetensors":
        raise ValueError("A local PP-OCRv6 small .safetensors base model is required")
    if not Path(command[1]).is_file():
        raise RuntimeError(f"ketos is missing beside this interpreter: {command[1]}")
    try:
        kraken_version = importlib.metadata.version("kraken")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Kraken 7.1.1 is required in the active environment") from error
    if kraken_version != "7.1.1":
        raise RuntimeError(f"This CLI contract is pinned to Kraken 7.1.1, found {kraken_version}")
    training, validation, audit = validate_manifests(train_manifest, validation_manifest, pilot, page_groups)
    budget = max_output_mib * 1024**2
    if free_bytes(output) < budget + MIN_FREE_BYTES:
        raise OSError("Insufficient disk space for the output budget plus a 2 GiB reserve")
    base_sha256 = file_sha256(base_model)
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    (output / "checkpoints").mkdir()
    for name, records in (("train", training), ("validation", validation)):
        with (output / f"{name}.txt").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(record["image"] for record in records) + "\n")
    environment = os.environ.copy()
    environment.update({name: str(threads) for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    )})
    environment.update({
        "PYTHONHASHSEED": str(seed), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "WANDB_DISABLED": "true", "TORCH_HOME": str(output / "cache"),
        "HF_HOME": str(output / "cache/huggingface"), "PYTHONUNBUFFERED": "1",
        "TORCH_COMPILE_DISABLE": "1",
    })
    configuration = {
        "command": command, "kraken_version": kraken_version, "policy_version": POLICY_VERSION,
        "base_model_sha256": base_sha256, "seed": seed, "threads": threads, "epochs": epochs,
        "batch_size": batch_size, "pilot": pilot, "timeout_seconds": timeout_seconds,
        "max_output_mib": max_output_mib, "audit": audit,
        "torch_compile": False, "optimizer": "AdamW",
        "resume_checkpoint": str(resume_checkpoint.resolve()) if resume_checkpoint is not None else None,
        "determinism": "seeded; Kraken deterministic mode warns on nondeterministic operations",
        "checkpoint_policy": "at most ten fixed epochs; no automatic deletion by this wrapper",
    }
    write_json(output / "run.json", configuration)
    print("Command: " + shlex.join(command), flush=True)
    print("Page separation: " + audit["page_separation"], flush=True)
    print(f"Log: {output / 'ketos.log'}", flush=True)
    started = time.monotonic()
    process = None
    stop_reason = None
    launch_error = None
    received_signal = None
    previous_handlers = {}

    def interrupt_training(signum, frame):
        nonlocal received_signal
        received_signal = signum
        raise KeyboardInterrupt

    try:
        for signum in (signal.SIGHUP, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, interrupt_training)
        with (output / "ketos.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, cwd=output, start_new_session=True)
            while process.poll() is None:
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    stop_reason = "time_limit"
                    break
                output_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
                if output_bytes > budget or free_bytes(output) < MIN_FREE_BYTES:
                    stop_reason = "disk_limit"
                    break
                try:
                    process.wait(timeout=min(2, remaining))
                except subprocess.TimeoutExpired:
                    pass
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except (OSError, ValueError) as error:
        stop_reason = "execution_error"
        launch_error = str(error)
    finally:
        if process is not None:
            stop_process(process)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    elapsed = time.monotonic() - started
    returncode = process.returncode if process is not None else None
    artifacts = [str(path.relative_to(output)) for path in sorted((output / "checkpoints").glob("*.safetensors"))
                 if path.stat().st_size > 0]
    if returncode == 0 and stop_reason is None:
        for artifact in artifacts:
            write_input_contract(output / artifact, 2560)
    checkpoints = [str(path.relative_to(output)) for path in sorted((output / "checkpoints").glob("*.ckpt"))]
    summary = {
        "status": "completed" if returncode == 0 and artifacts and stop_reason is None else "failed",
        "returncode": returncode, "elapsed_seconds": round(elapsed, 3),
        "stop_reason": stop_reason, "error": launch_error, "weights": artifacts,
        "termination_signal": received_signal,
        "checkpoints": checkpoints, "train_count": len(training), "validation_count": len(validation),
        "metrics": "Read actual Kraken metrics in ketos.log and local TensorBoard logs; no gold/test evaluation performed",
        "output_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("synth", help="Generate basic orthographic patterns without corpus/gold text")
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--font", type=Path, required=True)
    synthetic.add_argument("--count", type=int, default=1000)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--font-size", type=int, default=48)
    synthetic.add_argument("--require-full-coverage", action="store_true",
                           help="Fail instead of reporting and excluding glyphs absent from this font")
    for name, epochs, timeout in (("pilot", 1, 1800), ("train", 3, 3600)):
        command = commands.add_parser(name, help="Bounded fixed-epoch CPU fine-tuning on explicit non-gold splits")
        command.add_argument("--train-manifest", type=Path, required=True)
        command.add_argument("--validation-manifest", type=Path, required=True)
        command.add_argument("--base-model", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--epochs", type=int, default=epochs)
        command.add_argument("--threads", type=int, default=4)
        command.add_argument("--batch-size", type=int, default=4)
        command.add_argument("--seed", type=int, default=2)
        command.add_argument("--timeout-seconds", type=int, default=timeout)
        command.add_argument("--max-output-mib", type=int, default=4096)
        command.add_argument("--resume-checkpoint", type=Path,
                     help="Resume optimizer and model state from a trusted local checkpoint")
        command.add_argument("--page-groups", type=Path,
                             help="Optional JSON image-path to source-page-ID map; relative paths resolve beside the JSON")
    arguments = vars(parser.parse_args(argv))
    selected = arguments.pop("command")
    try:
        if selected == "synth":
            arguments["font_path"] = arguments.pop("font")
            metadata = synthesize(**arguments)
            print(json.dumps({key: value for key, value in metadata.items() if key != "records"}, sort_keys=True))
            return 0
        summary = run_training(**arguments, pilot=selected == "pilot")
        if summary["stop_reason"] == "time_limit":
            return 124
        if summary["stop_reason"] == "interrupted":
            return 130
        return 0 if summary["status"] == "completed" else 1
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"training: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())