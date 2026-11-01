import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from khojki_ocr.adapt import character_coverage, prepare_data, run_adaptation
from khojki_ocr.core import POLICY_VERSION, raster_digest


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "prepared"
        self.training = self.root / "real-train.json"
        self.development = self.root / "real-dev.json"
        self.train = self.reference("train", 20, "\U00011208\U0001122c", 4)
        self.dev = self.reference("dev", 22, "\U00011219\U0001122d", 12)
        self.save()

    def reference(self, identifier, page, text, mark):
        source = self.root / f"{identifier}.png"
        with Image.new("RGB", (200, 50), "white") as image:
            ImageDraw.Draw(image).rectangle((mark, 2, mark + 3, 15), fill="black")
            image.save(source)
        return {"id": identifier, "page": page, "status": "reviewed", "text": text,
                "source_preview": str(source), "bbox": [0, 0, 100, 25],
                "label_provenance": "unit-test fixture, not real reference text"}

    def save(self):
        for path, line in ((self.training, self.train), (self.development, self.dev)):
            path.write_text(json.dumps({"policy_version": POLICY_VERSION, "dataset_kind": "real",
                                        "label_provenance": "fixture", "lines": [line]}), encoding="utf-8")

    def prepare(self, **options):
        return prepare_data(self.training, self.development, self.output, **options)

    def test_exact_rgb_crop_gt_paths_and_provenance(self):
        prepared = self.prepare()
        for name, original in (("training", self.train), ("development", self.dev)):
            record = prepared["splits"][name]["lines"][0]
            path = Path(record["image"])
            self.assertTrue(path.is_absolute())
            self.assertEqual(path.with_suffix(".gt.txt").read_text(), original["text"])
            self.assertEqual(record["normalized_width"], 416)
            self.assertEqual(record["label_provenance"], original["label_provenance"])
            with Image.open(path) as crop:
                self.assertEqual(crop.mode, "RGB")
                self.assertEqual(crop.size, (100, 25))
                self.assertEqual(raster_digest(crop.tobytes(), *crop.size), record["raster_sha256"])
            self.assertEqual((self.output / f"{name}.txt").read_text().splitlines(), [str(path)])

    def test_synthetic_curriculum_metadata_is_not_reported_as_real(self):
        for path in (self.training, self.development):
            manifest = json.loads(path.read_text())
            manifest["dataset_kind"] = "synthetic"
            path.write_text(json.dumps(manifest))
        prepared = self.prepare()
        self.assertEqual(prepared["dataset_kind"], "synthetic")
        self.assertEqual(prepared["splits"]["development"]["dataset_kind"], "synthetic")

    def test_invalid_bboxes_fail_before_output(self):
        for box in (None, [0, 0, 1], [False, 0, 5, 5], [0.0, 0, 5, 5],
                    [-1, 0, 10, 10], [0, 0, 201, 10], [5, 0, 5, 10], [0, 5, 10, 4]):
            with self.subTest(box=box):
                self.train["bbox"] = box
                self.save()
                with self.assertRaisesRegex(ValueError, "bbox"):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_non_target_empty_and_multiline_labels_fail(self):
        for text in ("Latin", "\U00011212", "", " ", "\U00011208\n", "\U00011208\t"):
            with self.subTest(text=text):
                self.train["text"] = text
                self.save()
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_split_page_id_and_full_text_overlap(self):
        original = dict(self.dev)
        for key in ("page", "id", "text"):
            with self.subTest(key=key):
                self.dev = {**original, key: self.train[key]}
                self.save()
                with self.assertRaisesRegex(ValueError, f"{key} overlap"):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_numeric_page_alias_is_overlap(self):
        self.dev["page"] = "020"
        self.save()
        with self.assertRaisesRegex(ValueError, "page overlap"):
            self.prepare()

    def test_image_overlap_with_different_ids_and_text(self):
        Path(self.dev["source_preview"]).write_bytes(Path(self.train["source_preview"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "image hash overlap"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_width_uses_actual_height_and_never_clips(self):
        with self.assertRaisesRegex(ValueError, "no clipping"):
            self.prepare(height=128, max_width=512)
        self.assertFalse(self.output.exists())
        prepared = self.prepare(height=96, max_width=416)
        self.assertEqual(prepared["splits"]["training"]["lines"][0]["normalized_width"], 416)

    def test_no_existing_directory_or_symlink_overwrite(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("unchanged")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(sentinel.read_text(), "unchanged")
        link = self.root / "link"
        link.symlink_to(self.root / "absent")
        with self.assertRaises(FileExistsError):
            prepare_data(self.training, self.development, link)

    def test_held_out_pages_rejected_before_images_open(self):
        for page in (1, 18, 19, 23, 24):
            with self.subTest(page=page):
                self.train["page"] = page
                self.save()
                with patch("PIL.Image.open", side_effect=AssertionError("must not open test image")):
                    with self.assertRaisesRegex(ValueError, "Held-out page"):
                        self.prepare()

    def test_explicit_evaluation_only_manifest_rejected_before_images_open(self):
        manifest = json.loads(self.training.read_text())
        manifest["training_allowed"] = False
        self.training.write_text(json.dumps(manifest))
        with patch("PIL.Image.open", side_effect=AssertionError("must not open evaluation image")):
            with self.assertRaisesRegex(ValueError, "Evaluation-only"):
                self.prepare()
        self.assertFalse(self.output.exists())

    def test_test_manifest_not_opened(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("must not open test labels")):
            with self.assertRaisesRegex(ValueError, "Test/gold"):
                prepare_data(self.root / "references.json", self.development, self.output)

    def test_unreviewed_or_unknown_input_fails_before_output(self):
        for key, value in (("dataset_kind", "unknown"), ("policy_version", "wrong")):
            self.save()
            manifest = json.loads(self.training.read_text())
            manifest[key] = value
            self.training.write_text(json.dumps(manifest))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.prepare()
        self.train["status"] = "draft"
        self.save()
        with self.assertRaisesRegex(ValueError, "reviewed"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_duplicate_full_text_within_split(self):
        manifest = json.loads(self.training.read_text())
        manifest["lines"].append({**self.train, "id": "different-id"})
        self.training.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "NFC full text"):
            self.prepare()

    def test_same_source_with_different_crops_is_not_a_new_page(self):
        self.dev.update(source_preview=self.train["source_preview"], bbox=[4, 0, 104, 25])
        self.save()
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            self.prepare()

    def test_codec_union_is_driven_by_training_only(self):
        prepared = {"splits": {"training": {"lines": [self.train]}, "development": {"lines": [self.dev]}}}
        mapping = {"\U00011208": [1], "\U0001122c": [2]}
        coverage = character_coverage(prepared, mapping)
        self.assertEqual(coverage["resize"], "fail")
        self.assertEqual(coverage["checkpoint_codec"], mapping)
        self.assertTrue(coverage["development_missing_from_checkpoint"])
        del mapping["\U0001122c"]
        self.assertEqual(character_coverage(prepared, mapping)["resize"], "union")

    def test_runner_existing_output_rejected_without_runtime(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("unchanged")
        with self.assertRaises(FileExistsError):
            run_adaptation(self.training, self.development, self.root / "absent.ckpt", self.output)
        self.assertEqual(sentinel.read_text(), "unchanged")

    def test_missing_checkpoint_writes_summary_and_trace_without_runtime(self):
        result = run_adaptation(self.training, self.development, self.root / "absent.ckpt", self.output)
        self.assertEqual(result["stop_reason"], "error")
        self.assertFalse(result["export_complete"])
        self.assertIn("FileNotFoundError", result["error"])
        self.assertIn("Traceback", (self.output / "error.log").read_text())
        self.assertEqual(json.loads((self.output / "summary.json").read_text()), result)


if __name__ == "__main__":
    unittest.main()