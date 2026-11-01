"""CPU fine-tuning on reviewed train/development crops, never test data."""

import argparse
import json
import math
import os
import signal
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from khojki_ocr.core import POLICY_VERSION, raster_digest
from khojki_ocr.evaluation import score_line
from khojki_ocr.training import file_sha256, line_text, write_input_contract, write_json


HELD_OUT_PAGES = frozenset({"1", "18", "19", "23", "24", "34", "44"})


def _new_path(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {path}")


def _read_references(path: Path) -> dict:
    if path.name == "references.json" or any(
        token in path.stem.lower().replace("_", "-").split("-") for token in ("test", "gold")
    ):
        raise ValueError("Test/gold references must not be opened by adaptation")
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict) or manifest.get("policy_version") != POLICY_VERSION:
        raise ValueError(f"References require policy_version {POLICY_VERSION!r}")
    if manifest.get("training_allowed") is False:
        raise ValueError("Evaluation-only references must not enter adaptation")
    if manifest.get("dataset_kind") not in ("real", "synthetic", "mixed"):
        raise ValueError("Adaptation requires dataset_kind real, synthetic or mixed")
    records = manifest.get("lines")
    if not isinstance(records, list) or not records:
        raise ValueError("Each split requires a nonempty lines list")
    identifiers = set()
    texts = set()
    lines = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each reference line must be an object")
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError("Each reference requires a unique nonempty string id")
        if record.get("status") != "reviewed":
            raise ValueError(f"Reference {identifier!r} must be reviewed")
        if record.get("uncertain"):
            raise ValueError(f"Reference {identifier!r} has unresolved characters")
        if not isinstance(record.get("text"), str):
            raise ValueError(f"Reference {identifier!r} requires string text")
        text = line_text(record["text"])
        if text in texts:
            raise ValueError("Duplicate NFC full text within a split")
        page = record.get("page")
        if type(page) is int and page > 0:
            page = str(page)
        elif isinstance(page, str) and page.strip():
            page = page.strip()
            if page.isdecimal():
                page = str(int(page))
        else:
            raise ValueError(f"Reference {identifier!r} requires a page ID")
        if page in HELD_OUT_PAGES:
            raise ValueError(f"Held-out page {page} must not enter adaptation")
        source = record.get("source_preview")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Reference {identifier!r} requires source_preview")
        box = record.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box):
            raise ValueError(f"Invalid bbox for {identifier!r}")
        source_path = Path(source).expanduser()
        if not source_path.is_absolute() and not source_path.is_file():
            source_path = path.parent / source_path
        lines.append({**record, "text": text, "page": page,
                      "source_preview": str(source_path.resolve())})
        identifiers.add(identifier)
        texts.add(text)
    import hashlib

    return {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
            "policy_version": POLICY_VERSION, "dataset_kind": manifest["dataset_kind"],
            "metadata": {key: value for key, value in manifest.items() if key != "lines"},
            "lines": lines}


def prepare_data(training: Path, development: Path, output: Path, *,
                 height: int = 96, max_width: int = 1280, padding: int = 16) -> dict:
    """Validate both splits before creating stable, unclipped RGB/GT pairs."""
    from PIL import Image

    _new_path(output)
    if type(height) is not int or height <= 0:
        raise ValueError("Expected a positive model input height")
    if type(max_width) is not int or not 64 <= max_width <= 4096 or padding != 16:
        raise ValueError("Invalid max_width or padding for the inference contract")
    splits = {"training": _read_references(training), "development": _read_references(development)}
    for key in ("page", "id", "text"):
        overlap = ({line[key] for line in splits["training"]["lines"]}
                   & {line[key] for line in splits["development"]["lines"]})
        if overlap:
            raise ValueError(f"Train/development {key} overlap (NFC full text for text)")
    sources = {}
    for split in splits.values():
        seen_rasters = set()
        for record in split["lines"]:
            source = Path(record["source_preview"])
            with Image.open(source) as original, original.convert("RGB") as rgb:
                left, top, right, bottom = record["bbox"]
                if not 0 <= left < right <= rgb.width or not 0 <= top < bottom <= rgb.height:
                    raise ValueError(f"Out-of-bounds bbox for {record['id']!r}")
                with rgb.crop((left, top, right, bottom)) as crop:
                    width = math.ceil(crop.width * height / crop.height) + 2 * padding
                    if width > max_width:
                        raise ValueError(f"Crop {record['id']!r} normalized width {width} exceeds max_width {max_width}; no clipping")
                    digest = raster_digest(crop.tobytes(), crop.width, crop.height)
                    if digest in seen_rasters:
                        raise ValueError("Duplicate image raster within a split")
                    seen_rasters.add(digest)
                    record.update(raster_sha256=digest, crop_size=list(crop.size),
                                  normalized_width=width)
            if str(source) not in sources:
                sources[str(source)] = file_sha256(source)
            record["source_sha256"] = sources[str(source)]
    for key in ("raster_sha256", "source_sha256"):
        if ({line[key] for line in splits["training"]["lines"]}
                & {line[key] for line in splits["development"]["lines"]}):
            raise ValueError(f"Train/development image hash overlap: {key}")
    output.mkdir(parents=True)
    for name, split in splits.items():
        directory = output / name
        directory.mkdir()
        paths = []
        for index, record in enumerate(split["lines"]):
            source = Path(record["source_preview"])
            if file_sha256(source) != record["source_sha256"]:
                raise ValueError("Source preview changed during preparation")
            with Image.open(source) as original, original.convert("RGB") as rgb:
                with rgb.crop(tuple(record["bbox"])) as crop:
                    if raster_digest(crop.tobytes(), crop.width, crop.height) != record["raster_sha256"]:
                        raise ValueError("Crop changed during preparation")
                    path = directory / f"line-{index + 1:04d}.png"
                    with path.open("xb") as stream:
                        crop.save(stream, format="PNG")
            with path.with_suffix(".gt.txt").open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(record["text"])
            record.update(image=str(path.resolve()), image_sha256=file_sha256(path))
            paths.append(record["image"])
        with (output / f"{name}.txt").open("x", encoding="utf-8") as stream:
            stream.write("\n".join(paths) + "\n")
    kinds = {split["dataset_kind"] for split in splits.values()}
    result = {"policy_version": POLICY_VERSION,
              "dataset_kind": next(iter(kinds)) if len(kinds) == 1 else "mixed", "splits": splits,
              "input_height": height, "max_width": max_width, "padding": padding,
              "crop_policy": "exact reviewed bbox, RGB, no blank clipping or resizing",
              "held_out_pages_excluded": sorted(HELD_OUT_PAGES)}
    write_json(output / "prepared.json", result)
    return result


def character_coverage(prepared: dict, mapping: dict) -> dict:
    training = set("".join(line["text"] for line in prepared["splits"]["training"]["lines"]))
    development = set("".join(line["text"] for line in prepared["splits"]["development"]["lines"]))
    existing = set("".join(mapping))

    def codepoints(characters):
        return [f"U+{ord(character):04X}" for character in sorted(characters)]

    return {"checkpoint_codec": {text: list(labels) for text, labels in mapping.items()},
            "training": codepoints(training), "development": codepoints(development),
            "development_not_in_training": codepoints(development - training),
            "training_missing_from_checkpoint": codepoints(training - existing),
            "development_missing_from_checkpoint": codepoints(development - existing),
            "resize": "union" if training - existing else "fail"}


def _measure_loader(module, loader, target_codec, expected: list[dict], *,
                    height: int, max_width: int) -> dict:
    import torch
    from kraken.configs import RecognitionInferenceConfig

    decoder = RecognitionInferenceConfig().decoder
    records = []
    shapes = []
    was_training = module.net.training
    module.net.eval()
    try:
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            for batch in loader:
                image = batch["image"]
                if tuple(image.shape[1:]) != (3, height, max_width) or not torch.isfinite(image).all():
                    raise ValueError("Actual loader input violates RGB/height/static-width contract")
                if (batch["seq_lens"] > max_width).any() or (batch["seq_lens"] <= 0).any():
                    raise ValueError("Invalid loader sequence lengths")
                shapes.append(list(image.shape))
                logits, lengths = module.net(image, batch["seq_lens"])
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite recognition logits")
                decoded = list(decoder(logits.softmax(1).squeeze(2), lengths))
                target_lengths = batch["target_lens"].tolist()
                if len(decoded) != len(target_lengths) or sum(target_lengths) != batch["target"].numel():
                    raise ValueError("Packed labels and decoded predictions do not align")
                offset = 0
                for positions, target_length in zip(decoded, target_lengths):
                    encoded = [(int(label), 0, 0, 0.0)
                               for label in batch["target"][offset:offset + target_length]]
                    reference = "".join(item[0] for item in target_codec.decode(encoded))
                    prediction = "".join(item[0] for item in module.codec.decode(positions))
                    records.append(score_line(reference, prediction))
                    offset += target_length
    finally:
        module.net.train(was_training)
    if Counter(record["reference_text"] for record in records) != Counter(line["text"] for line in expected):
        raise ValueError("Labels failed the actual loader/codec multiset round trip; samples were lost or changed")
    by_text = {record["reference_text"]: record for record in records}
    predictions = [{"id": line["id"], "page": line["page"], "provenance": "model",
                    **by_text[line["text"]]} for line in expected]
    characters = sum(record["reference_characters"] for record in records)
    errors = sum(record["errors"] for record in records)
    return {"cer": errors / characters, "errors": errors, "reference_characters": characters,
            "input_shapes": shapes, "predictions": predictions}


class _StopRun(Exception):
    pass


def run_adaptation(training: Path, development: Path, checkpoint: Path, output: Path, *,
                   seconds: int = 1200, max_steps: int = 100, lrate: float = 0.0005,
                   max_width: int = 1280, batch_size: int = 4, measure_every: int = 5,
                   augment: bool = False) -> dict:
    """Start fresh optimization from a trusted checkpoint; select by dev CER only."""
    _new_path(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    summary = {"scope": "training_and_development_only_no_test_evaluation",
               "selection_metric": "development_codepoint_cer", "error": None,
               "stop_reason": None, "steps": 0, "best_step": None, "best_dev_cer": None,
               "seconds": seconds, "max_steps": max_steps, "export_complete": False,
               "initial_train_cer": None, "initial_dev_cer": None,
               "selected_train_cer": None, "selected_dev_cer": None}
    previous_handlers = {}
    previous_timer = None
    trainer = model = observer = None
    best = None
    history = []
    best_path = output / "best.ckpt"
    temporary_checkpoint = output / "best.next.ckpt"
    weights = output / "best.safetensors"

    def interrupt(signum, frame):
        raise _StopRun("time_limit" if signum == signal.SIGALRM else "interrupted")

    def failure():
        summary["error"] = traceback.format_exc()
        summary["stop_reason"] = "error"
        with (output / "error.log").open("a", encoding="utf-8") as stream:
            stream.write(summary["error"] + "\n")
        print(summary["error"], file=sys.stderr, flush=True)

    try:
        if type(seconds) is not int or not 1 <= seconds <= 1200:
            raise ValueError("seconds must be an integer between 1 and 1200")
        if type(augment) is not bool:
            raise ValueError("augment must be a boolean")
        for name, value in (("max_steps", max_steps), ("batch_size", batch_size), ("measure_every", measure_every)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if measure_every not in (5, 10):
            raise ValueError("measure_every must be 5 or 10 steps")
        if isinstance(lrate, bool) or not math.isfinite(lrate) or lrate <= 0:
            raise ValueError("lrate must be finite and positive")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Trusted initialization checkpoint not found: {checkpoint}")
        deadline = started + seconds
        fit_deadline = deadline - min(90.0, seconds / 5)
        summary["finalization_reserve_seconds"] = min(90.0, seconds / 5)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        for signum in (signal.SIGALRM, signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, interrupt)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, fit_deadline - time.monotonic()))
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "2"
        import torch
        from lightning.pytorch import Callback, seed_everything
        from kraken.models.convert import convert_models
        from kraken.train import KrakenTrainer
        from kraken.train.ppocr import PPOCRv6RecognitionModel

        seed_everything(42, workers=True)
        torch.set_num_threads(2)
        options = dict(partition=1, format_type="path", linetype="bbox", normalization="NFC",
                       normalize_whitespace=False, bidi_reordering=False, augment=augment,
                       num_workers=0, max_width=max_width, padding=16, variant="small", resize="fail",
                       batch_size=batch_size, optimizer="AdamW", lrate=lrate, schedule="constant",
                       warmup=0, quit="fixed", epochs=max_steps, min_epochs=0,
                       checkpoint_path=str(output.resolve()))
        config = PPOCRv6RecognitionModel._config_class(**options)
        summary["initial_checkpoint"] = str(checkpoint.resolve())
        summary["initial_checkpoint_sha256"] = file_sha256(checkpoint)
        model = PPOCRv6RecognitionModel.load_from_checkpoint(str(checkpoint), config=config, weights_only=False)
        _, channels, height, width = model.net.input
        if channels != 3 or type(height) is not int or height <= 0 or width != 0:
            raise ValueError(f"Unsupported checkpoint input geometry: {model.net.input!r}")
        prepared = prepare_data(training, development, output / "data", height=height, max_width=max_width)
        coverage = character_coverage(prepared, model.net.codec.c2l)
        model.hparams.config.resize = coverage["resize"]
        options.update(resize=coverage["resize"], height=height, variant=model.net.variant,
                       training_data=[line["image"] for line in prepared["splits"]["training"]["lines"]],
                       evaluation_data=[line["image"] for line in prepared["splits"]["development"]["lines"]])
        data_config = PPOCRv6RecognitionModel._data_config_class(**options)
        data = PPOCRv6RecognitionModel._data_module_class(data_config)
        measurement_train_loader = data.train_dataloader
        real_texts = {line["text"] for line in prepared["splits"]["training"]["lines"]
                      if line.get("example_kind") != "synthetic"}
        if prepared["splits"]["training"]["dataset_kind"] == "mixed":
            from torch.utils.data import DataLoader, WeightedRandomSampler

            labels = [data.train_set.dataset._gt[index] for index in data.train_set.indices]
            real_count = sum(label in real_texts for label in labels)
            synthetic_count = len(labels) - real_count
            if not real_count or not synthetic_count:
                raise ValueError("Mixed training requires both real and synthetic examples")
            sampling_weights = [0.75 / real_count if label in real_texts else 0.25 / synthetic_count
                                for label in labels]

            def mixed_loader():
                return DataLoader(data.train_set, batch_size=batch_size, num_workers=0,
                                  sampler=WeightedRandomSampler(sampling_weights, len(labels), replacement=True),
                                  collate_fn=data._collate_fn)

            data.train_dataloader = mixed_loader
            summary["sampling"] = {"real_probability": 0.75, "synthetic_probability": 0.25}
        summary.update(character_coverage=coverage, input_height=height, max_width=max_width, padding=data_config.padding,
                       references={name: {key: split[key] for key in ("path", "sha256", "metadata")}
                                   for name, split in prepared["splits"].items()})
        write_json(output / "config.json", {"options": options, "references": summary["references"],
                   "character_coverage": coverage, "initial_checkpoint_sha256": summary["initial_checkpoint_sha256"],
                   "input_geometry": list(model.net.input), "seconds": seconds, "max_steps": max_steps,
                   "measure_every": measure_every, "train_measurements": "initial and selected best at finalization",
                   "auxiliary_loss": "unchanged Kraken CTC + NRTR", "resume": False,
                   "checkpoint_loading": "trusted pickle; load_from_checkpoint(config=new_config, weights_only=False)",
                   "selection": "strictly lowest development CER, including step zero; ties keep earlier checkpoint"})

        class FitObserver(Callback):
            pending_best = None
            last_loss = None

            def measure(self, trainer, module, phase, include_train=False, select=True, step=None):
                nonlocal best
                dev = _measure_loader(module, trainer.datamodule.val_dataloader(), module._val_codec,
                                      prepared["splits"]["development"]["lines"], height=height, max_width=max_width)
                train = (_measure_loader(module, measurement_train_loader(), module.codec,
                                         prepared["splits"]["training"]["lines"], height=height, max_width=max_width)
                         if include_train else None)
                item = {"timestamp": datetime.now(timezone.utc).isoformat(), "phase": phase,
                        "step": trainer.global_step if step is None else step, "epoch": trainer.current_epoch,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "loss": self.last_loss if select else best["loss"],
                        "dev_cer": dev["cer"], "train_cer": train["cer"] if train else None,
                        "development": dev, "training": train}
                improved = select and (best is None or item["dev_cer"] < best["dev_cer"])
                item["new_best"] = improved
                if improved:
                    self.pending_best = item
                    try:
                        trainer.save_checkpoint(temporary_checkpoint, weights_only=False)
                        os.replace(temporary_checkpoint, best_path)
                    finally:
                        self.pending_best = None
                    best = item
                    summary.update(best_step=item["step"], best_dev_cer=item["dev_cer"])
                history.append(item)
                with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(item, ensure_ascii=True, allow_nan=False) + "\n")
                print(json.dumps({key: item[key] for key in ("timestamp", "step", "phase", "loss", "dev_cer", "train_cer", "new_best")}), flush=True)
                if select and dev["cer"] <= 0.01:
                    summary["stop_reason"] = "target_development_cer"
                    trainer.should_stop = True
                return item

            def on_train_start(self, trainer, module):
                current = {text: list(labels) for text, labels in module.codec.c2l.items()}
                if any(current.get(text) != labels for text, labels in coverage["checkpoint_codec"].items()):
                    raise ValueError("Existing checkpoint codec labels were remapped")
                if module.nrtr_head is None:
                    raise ValueError("The auxiliary NRTR head must be retained")
                summary["effective_codec"] = current
                initial = self.measure(trainer, module, "initial", include_train=True)
                summary.update(initial_train_cer=initial["train_cer"], initial_dev_cer=initial["dev_cer"])

            def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
                loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
                self.last_loss = float(loss.detach()) if loss is not None else None
                if self.last_loss is not None and not math.isfinite(self.last_loss):
                    raise ValueError("Non-finite training loss")
                summary["steps"] = trainer.global_step
                if trainer.global_step % measure_every == 0:
                    self.measure(trainer, module, "development")
                if time.monotonic() >= fit_deadline:
                    summary["stop_reason"] = "time_limit"
                    trainer.should_stop = True

            def on_save_checkpoint(self, trainer, module, saved):
                if self.pending_best is not None:
                    saved["adapt_selection"] = {
                        "step": self.pending_best["step"], "development_cer": self.pending_best["dev_cer"],
                        "metric": "development_codepoint_cer", "optimizer_matches_weights": True,
                        "initial_checkpoint_sha256": summary["initial_checkpoint_sha256"],
                        "references": summary["references"], "input_height": height,
                        "max_width": max_width, "padding": data_config.padding,
                    }

        observer = FitObserver()
        trainer = KrakenTrainer(accelerator="cpu", devices=1, precision="32-true", min_epochs=0,
                                max_epochs=max_steps, max_steps=max_steps, enable_progress_bar=False,
                                enable_summary=False, enable_checkpointing=False, enable_model_summary=False,
                                num_sanity_val_steps=0, limit_val_batches=0, callbacks=[observer],
                                gradient_clip_val=1.0, deterministic="warn", default_root_dir=str(output.resolve()))
        try:
            trainer.fit(model, data)
        except _StopRun as stopped:
            summary["stop_reason"] = str(stopped)
        summary["steps"] = trainer.global_step
        summary["stop_reason"] = summary["stop_reason"] or ("max_steps" if trainer.global_step >= max_steps else "epochs_complete")
        signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
        if best is None:
            raise RuntimeError("No complete development measurement/checkpoint; nothing can be exported")
        if summary["stop_reason"] not in ("time_limit", "interrupted") and history[-1]["step"] != trainer.global_step:
            observer.measure(trainer, model, "final_development")
        convert_models([best_path], weights, weights_format="safetensors")
        sidecar = write_input_contract(weights, data_config.max_width, data_config.padding)
        summary.update(export_complete=True, weights=str(weights.resolve()),
                       weights_sha256=file_sha256(weights), input_contract=str(sidecar.resolve()),
                       input_contract_sha256=file_sha256(sidecar), checkpoint=str(best_path.resolve()),
                       checkpoint_sha256=file_sha256(best_path), checkpoint_optimizer_matches_best_step=True)
        selected = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(selected["state_dict"])
        del selected
        final = observer.measure(trainer, model, "selected_best", include_train=True, select=False, step=best["step"])
        summary.update(selected_train_cer=final["train_cer"], selected_dev_cer=final["dev_cer"],
                       selected_metrics_complete=True,
                       final_model_handling="restored best weights for measurement only; no further optimizer steps")
    except _StopRun as stopped:
        summary["stop_reason"] = str(stopped)
        summary["finalization_incomplete"] = True
    except Exception:
        failure()
    finally:
        if previous_timer is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if trainer is not None:
            summary["steps"] = trainer.global_step
        if temporary_checkpoint.exists():
            temporary_checkpoint.unlink()
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        summary["checkpoint_available"] = best_path.is_file()
        summary["selected_metrics_complete"] = summary.get("selected_metrics_complete", False)
        write_json(output / "summary.json", summary)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if previous_timer is not None and previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(0.001, previous_timer[0] - (time.monotonic() - started)), previous_timer[1])
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted local Kraken checkpoint (pickle)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lrate", type=float, default=0.0005)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--measure-every", type=int, choices=(5, 10), default=5)
    parser.add_argument("--augment", action="store_true", help="Use Kraken's training image augmentation")
    args = parser.parse_args()
    result = run_adaptation(**vars(args))
    if result["error"] or not result["export_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()