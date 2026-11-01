import contextlib
import hashlib
import io
import json
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from khojki_ocr.core import KHOJKI_CODEPOINTS, normalize_text
from khojki_ocr.training import (
    ANUSVARA, BASIC_ALPHABET, CONSONANTS, INDEPENDENT_VOWELS,
    PATTERN_EXCLUSIONS, PUNCTUATION, VOWEL_SIGNS, generate_splits, line_text,
    build_ketos_command, load_font, read_manifest, reject_held_out_path,
    run_training, stop_process, synthesize, validate_manifests, write_input_contract,
)


class InputContractTests(unittest.TestCase):
    def test_sidecar_records_schema_model_hash_and_width_boundaries(self):
        payload = b"mocked recognition weights\x00\xff"
        with tempfile.TemporaryDirectory() as directory:
            for width in (64, 512, 4096):
                with self.subTest(width=width):
                    weights = Path(directory) / f"model-{width}.safetensors"
                    weights.write_bytes(payload)
                    sidecar = write_input_contract(weights, width)
                    self.assertEqual(sidecar, weights.with_suffix(".input.json"))
                    contract = json.loads(sidecar.read_text(encoding="utf-8"))
                    self.assertEqual(contract["schema"], "khojki-ppocrv6-input-v1")
                    self.assertEqual(contract["model_sha256"], hashlib.sha256(payload).hexdigest())
                    self.assertEqual(contract["max_width"], width)
                    self.assertEqual(contract["padding"], 16)
                    self.assertEqual(contract["operation"],
                                     "right-pad normalized tensors with zeros; preserve sequence lengths")
                    self.assertEqual(weights.read_bytes(), payload)

    def test_invalid_width_including_bool_creates_no_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.safetensors"
            weights.write_bytes(b"mocked weights")
            for width in (-1, 0, 63, 4097, True, False, 64.0, "64", None):
                with self.subTest(width=width), self.assertRaisesRegex(ValueError, "tensor width"):
                    write_input_contract(weights, width)
                self.assertFalse(weights.with_suffix(".input.json").exists())
            self.assertEqual(weights.read_bytes(), b"mocked weights")

    def test_unsupported_padding_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.safetensors"
            weights.write_bytes(b"mocked weights")
            for padding in (0, 8, 32, True, 16.0, "16", None):
                with self.subTest(padding=padding), self.assertRaisesRegex(ValueError, "padding"):
                    write_input_contract(weights, 512, padding)
            self.assertFalse(weights.with_suffix(".input.json").exists())

    def test_existing_contract_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.safetensors"
            weights.write_bytes(b"mocked weights")
            sidecar = write_input_contract(weights, 512)
            original = sidecar.read_bytes()
            with self.assertRaises(FileExistsError):
                write_input_contract(weights, 256)
            self.assertEqual(sidecar.read_bytes(), original)


class SyntheticTextTests(unittest.TestCase):
    def test_seed_reproducibility_and_distinct_splits(self):
        first = generate_splits(100, seed=42)
        self.assertEqual(first, generate_splits(100, seed=42))
        self.assertNotEqual(first, generate_splits(100, seed=43))
        training, validation = first
        self.assertEqual((len(training), len(validation)), (90, 10))
        self.assertFalse(set(training) & set(validation))
        self.assertEqual(len(set(training + validation)), 100)

    def test_coverage_includes_rare_letters_and_independent_vowels(self):
        training, validation = generate_splits(20)
        covered = {ord(character) for text in training for character in text}
        self.assertTrue(BASIC_ALPHABET <= covered)
        self.assertTrue({0x1123F, 0x11240, 0x11241} <= covered)
        self.assertTrue(INDEPENDENT_VOWELS <= covered)
        for text in training + validation:
            self.assertEqual(text, normalize_text(text))
            self.assertTrue({ord(character) for character in text} <= KHOJKI_CODEPOINTS | {32})
            self.assertNotIn("\U00011212", text)

    def test_dependent_marks_have_bases_and_one_vowel_per_syllable(self):
        training, validation = generate_splits(100)
        for text in training + validation:
            for token in text.split(" "):
                self.assertIn(ord(token[0]), CONSONANTS | INDEPENDENT_VOWELS)
                signs = [ord(character) for character in token if ord(character) in VOWEL_SIGNS]
                self.assertLessEqual(len(signs), 1)
                if signs:
                    self.assertIn(ord(token[0]), CONSONANTS)
                if chr(ANUSVARA) in token:
                    syllable = token[:-1] if ord(token[-1]) in PUNCTUATION else token
                    self.assertEqual(syllable[-1], chr(ANUSVARA))

    def test_supported_font_subset_is_explicit(self):
        supported = BASIC_ALPHABET - {0x11240, 0x11241}
        training, validation = generate_splits(20, alphabet=supported)
        covered = {ord(character) for text in training + validation for character in text}
        self.assertTrue(supported <= covered)
        self.assertFalse({0x11240, 0x11241} & covered)
        self.assertTrue(PATTERN_EXCLUSIONS)
        self.assertFalse(PATTERN_EXCLUSIONS & BASIC_ALPHABET)

    def test_invalid_alphabet_and_unbounded_counts_are_rejected(self):
        for alphabet in (BASIC_ALPHABET | {0x11212}, BASIC_ALPHABET | {65}, frozenset()):
            with self.assertRaises(ValueError):
                generate_splits(20, alphabet=alphabet)
        for count in (0, 19, 10001):
            with self.assertRaises(ValueError):
                generate_splits(count)

    def test_line_policy_rejects_empty_reserved_non_target_and_multiline(self):
        for text in ("", " ", "\U00011208\n", "\U00011208\t", "A", "1", "\U00011212", "\U0001122d"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                line_text(text)
        self.assertEqual(line_text("\U00011208  \U00011238"), "\U00011208  \U00011238")


class SyntheticFileTests(unittest.TestCase):
    def test_coverage_curriculum_uses_disjoint_synthetic_labels_and_pages(self):
        from PIL import Image
        from khojki_ocr.coverage import build_coverage

        supported = {0x11200, 0x11208, 0x1121F, 0x11229, 0x1122C, 0x11230}
        with tempfile.TemporaryDirectory() as temporary, patch(
                "khojki_ocr.coverage.load_font", return_value=(None, supported, {"fixture": True})), patch(
                "khojki_ocr.coverage.render_line", side_effect=lambda text, font: Image.new("RGB", (40, 20), "white")):
            root = Path(temporary)
            (root / "font.ttf").write_bytes(b"mocked renderer font fixture")
            training = build_coverage(root / "font.ttf", root / "training")
            development = build_coverage(root / "font.ttf", root / "development", development=True)
            self.assertEqual(training["dataset_kind"], "synthetic")
            self.assertEqual(development["dataset_kind"], "synthetic")
            self.assertFalse({line["text"] for line in training["lines"]}
                             & {line["text"] for line in development["lines"]})
            self.assertFalse({line["page"] for line in training["lines"]}
                             & {line["page"] for line in development["lines"]})

    def test_raqm_absence_fails_before_opening_an_unshaped_font(self):
        try:
            import fontTools.ttLib
            import PIL.features
        except ImportError:
            self.skipTest("Pillow/fonttools are not yet available")
        with tempfile.TemporaryDirectory() as directory:
            font_path = Path(directory) / "font.ttf"
            font_path.write_bytes(b"not a font; RAQM must be checked first")
            with patch("PIL.features.check_feature", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "BASIC is unsafe"):
                    load_font(font_path)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinel = output / "keep.txt"
            sentinel.write_text("user data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                synthesize(output, output / "absent.ttf", count=20)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "user data")

    def test_full_coverage_can_be_required_before_any_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "synthetic"
            with patch("khojki_ocr.training.load_font", return_value=(None, BASIC_ALPHABET - {0x11241}, {})):
                with self.assertRaisesRegex(ValueError, "U\\+11241"):
                    synthesize(output, Path(directory) / "font.ttf", count=20, require_full_coverage=True)
            self.assertFalse(output.exists())

    def test_actual_font_shaped_generation_when_environment_is_available(self):
        font_path = Path(__file__).resolve().parents[1] / "web/assets/NotoSansKhojki-Regular.ttf"
        if not font_path.is_file():
            self.skipTest("Bundled Khojki font is not available")
        try:
            load_font(font_path)
        except RuntimeError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            with patch("khojki_ocr.training.free_bytes", return_value=10 * 1024**3):
                metadata = synthesize(first, font_path, count=20)
                repeated = synthesize(second, font_path, count=20)
            self.assertEqual(metadata, repeated)
            self.assertEqual(len(list(first.rglob("*.png"))), 20)
            self.assertEqual(len(list(first.rglob("*.gt.txt"))), 20)
            for record in metadata["records"]:
                self.assertEqual((first / record["image"]).read_bytes(), (second / record["image"]).read_bytes())
            self.assertEqual(len((first / "train.txt").read_text().splitlines()), 18)
            self.assertEqual(len((first / "validation.txt").read_text().splitlines()), 2)


class TrainingCommandTests(unittest.TestCase):
    def test_stop_escalates_only_the_owned_process_group(self):
        process = MagicMock()
        process.pid = 12345
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("ketos", 10), 0]
        with patch("khojki_ocr.training.os.killpg") as kill_group:
            stop_process(process)
        self.assertEqual([call.args for call in kill_group.call_args_list],
                         [(12345, signal.SIGTERM), (12345, signal.SIGKILL)])

    def test_711_command_uses_correct_scope_explicit_splits_and_finite_cpu_training(self):
        command = build_ketos_command(Path("train.txt"), Path("validation.txt"),
                                      Path("base-small.safetensors"), Path("run"))
        self.assertEqual(command[:2], [sys.executable, str(Path(sys.executable).with_name("ketos"))])
        boundary = command.index("train")
        self.assertLess(command.index("--threads"), boundary)
        self.assertLess(command.index("--precision"), boundary)
        for flag, value in (("--device", "cpu"), ("--precision", "32-true"), ("--seed", "2"),
                            ("--workers", "0"), ("--arch", "ppocrv6"), ("--variant", "small"),
                            ("--optimizer", "AdamW"),
                            ("--resize", "new"), ("--quit", "fixed"), ("--epochs", "1"),
                            ("--min-epochs", "1"), ("--normalization", "NFC"),
                            ("--logger", "tensorboard"), ("--training-data", "train.txt"),
                            ("--evaluation-data", "validation.txt")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--no-normalize-whitespace", command)
        self.assertIn("--no-reorder", command)
        self.assertNotIn("--freeze-backbone", command)
        self.assertNotIn("--partition", command)

    def test_resource_bounds_and_pilot_epoch_limit(self):
        for options in ({"epochs": 0}, {"epochs": 11}, {"threads": 0}, {"batch_size": 9}, {"seed": -1}):
            with self.assertRaises(ValueError):
                build_ketos_command(Path("train"), Path("validation"), Path("base"), Path("output"), **options)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "exactly one epoch"):
                run_training(Path("train"), Path("validation"), Path("base"), Path(directory) / "run", epochs=2)

    def test_resume_preserves_checkpoint_head_instead_of_resizing(self):
        checkpoint = Path("run/checkpoint_abort.ckpt")
        command = build_ketos_command(Path("train"), Path("validation"), Path("base"), Path("output"),
                                      resume_checkpoint=checkpoint)
        self.assertEqual(command[command.index("--resume") + 1], str(checkpoint.resolve()))
        self.assertNotIn("--load", command)
        self.assertNotIn("--resize", command)

    def test_obvious_gold_and_test_paths_are_not_consumed(self):
        for path in ("data/gold/line.png", "data/test/line.png", "data/gold_manifest.txt", "data/held-out/line.png"):
            with self.assertRaises(ValueError):
                reject_held_out_path(Path(path))


class ManifestTests(unittest.TestCase):
    def setUp(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not yet available")
        self.image_module = Image
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.training = self.root / "train.txt"
        self.validation = self.root / "validation.txt"
        self.make_pair("first", "\U00011208 \U00011209", 0)
        self.make_pair("second", "\U00011209 \U00011208", 10)
        self.training.write_text("first.png\n", encoding="utf-8")
        self.validation.write_text("second.png\n", encoding="utf-8")

    def make_pair(self, name, label, shade):
        image = self.image_module.new("L", (200, 48), 255)
        image.paste(shade, (10, 10, 50, 30))
        image.save(self.root / f"{name}.png")
        (self.root / f"{name}.gt.txt").write_text(label, encoding="utf-8")

    def test_relative_paths_and_single_terminal_newline(self):
        (self.root / "first.gt.txt").write_text("\U00011208 \U00011209\n", encoding="utf-8")
        records = read_manifest(self.training)
        self.assertEqual(records[0]["image"], str((self.root / "first.png").resolve()))
        self.assertEqual(records[0]["text"], "\U00011208 \U00011209")
        training, validation, audit = validate_manifests(self.training, self.validation)
        self.assertEqual((len(training), len(validation)), (1, 1))
        self.assertIn("not verified", audit["page_separation"])

    def test_duplicate_text_or_raster_cannot_cross_splits(self):
        (self.root / "second.gt.txt").write_text("\U00011208 \U00011209", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "overlap in text"):
            validate_manifests(self.training, self.validation)
        self.make_pair("second", "\U00011209 \U00011208", 0)
        with self.assertRaisesRegex(ValueError, "overlap in raster"):
            validate_manifests(self.training, self.validation)

    def test_page_separation_requires_complete_disjoint_ids(self):
        groups = self.root / "groups.json"
        groups.write_text(json.dumps({"first.png": "page-1", "second.png": "page-1"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source-page overlap"):
            validate_manifests(self.training, self.validation, page_groups=groups)
        groups.write_text(json.dumps({"first.png": "page-1", "second.png": "page-2"}), encoding="utf-8")
        _, _, audit = validate_manifests(self.training, self.validation, page_groups=groups)
        self.assertTrue(audit["page_separation"].startswith("verified disjoint"))

    def test_empty_ctc_labels_missing_glyphs_and_blank_images_are_rejected(self):
        for label in ("", " ", "\U00011208\n\U00011209", "\U00011212"):
            (self.root / "first.gt.txt").write_text(label, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_manifest(self.training)
        (self.root / "first.gt.txt").write_text("\U00011208", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "absent from training codec"):
            validate_manifests(self.training, self.validation)
        self.image_module.new("L", (200, 48), 255).save(self.root / "first.png")
        with self.assertRaisesRegex(ValueError, "Blank/near-blank"):
            read_manifest(self.training)

    def test_mocked_launch_records_elapsed_return_code_and_artifacts(self):
        executable = self.root / "python"
        (self.root / "ketos").write_text("mocked entry point", encoding="utf-8")
        base = self.root / "base-small.safetensors"
        base.write_bytes(b"mocked base model")
        for returncode in (0, 7):
            output = self.root / f"run-{returncode}"
            process = MagicMock()
            process.returncode = returncode
            process.poll.return_value = returncode

            def fake_launch(command, **options):
                if returncode == 0:
                    (output / "checkpoints/best_0.1234.safetensors").write_bytes(b"mocked weights")
                return process

            with patch("khojki_ocr.training.sys.executable", str(executable)), \
                    patch("khojki_ocr.training.importlib.metadata.version", return_value="7.1.1"), \
                    patch("khojki_ocr.training.free_bytes", return_value=10 * 1024**3), \
                    patch("khojki_ocr.training.subprocess.Popen", side_effect=fake_launch) as launch, \
                    patch("khojki_ocr.training.time.monotonic", side_effect=[100.0, 103.5]), \
                    contextlib.redirect_stdout(io.StringIO()):
                summary = run_training(self.training, self.validation, base, output)
            self.assertEqual(summary["elapsed_seconds"], 3.5)
            self.assertEqual(summary["returncode"], returncode)
            self.assertEqual(summary["status"], "completed" if returncode == 0 else "failed")
            self.assertNotIn("accuracy", summary)
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
            self.assertEqual(launch.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.assertEqual(launch.call_args.kwargs["env"]["OMP_NUM_THREADS"], "4")
            self.assertEqual(launch.call_args.kwargs["env"]["TORCH_COMPILE_DISABLE"], "1")
            self.assertFalse(json.loads((output / "run.json").read_text())["torch_compile"])
            self.assertTrue(launch.call_args.kwargs["start_new_session"])
            self.assertFalse(launch.call_args.kwargs.get("shell", False))
            self.assertEqual((output / "train.txt").read_text().strip(), str((self.root / "first.png").resolve()))


if __name__ == "__main__":
    unittest.main()