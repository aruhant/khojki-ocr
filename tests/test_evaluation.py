import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from khojki_ocr.core import POLICY_VERSION
from khojki_ocr.evaluation import evaluate, evaluate_detected_lines, main, normalize_predictions, predict_reference_lines, score_line, write_result

KA = "\U00011208"
KHA = "\U00011209"


def make_references(*texts, dataset_kind="real"):
    return {"policy_version": POLICY_VERSION, "dataset_kind": dataset_kind, "lines": [
        {"id": f"line-{index}", "page": 1, "text": text, "status": "reviewed"}
        for index, text in enumerate(texts, 1)]}


def make_predictions(*texts, provenance="model"):
    return [{"id": f"line-{index}", "text": text, "provenance": provenance}
            for index, text in enumerate(texts, 1)]


class CropRecognitionTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.image = self.root / "source.png"
        Image.new("RGB", (100, 30), "white").save(self.image)
        self.model = self.root / "model.safetensors"
        self.model.write_bytes(b"test model identity only")
        self.references = make_references(KA)
        self.references["lines"][0].update(source_preview=str(self.image), bbox=[10, 5, 90, 25])

    def test_recognizer_receives_crop_and_id_but_no_reference_text(self):
        calls = []

        def predictor(image, identifier):
            calls.append((image.size, identifier))
            return "English"

        predictions = predict_reference_lines(self.references, self.model, predictor=predictor)
        self.assertEqual(calls, [((80, 20), "line-1")])
        self.assertEqual(predictions["lines"][0]["text"], "English")
        self.assertEqual(predictions["scope"], "recognition_only_manual_crops")
        self.assertFalse(predictions["reference_text_given_to_model"])
        self.assertEqual(len(predictions["model_sha256"]), 64)
        self.assertGreater(evaluate(self.references, predictions)["errors"], 0)

    def test_bad_crop_is_rejected_instead_of_padding_or_clipping(self):
        for box in ([0, 0, 101, 20], [-1, 0, 80, 20], [1, 1, 1, 2], [False, 0, 80, 20]):
            self.references["lines"][0]["bbox"] = box
            with self.subTest(box=box), self.assertRaises(ValueError):
                predict_reference_lines(self.references, self.model, predictor=lambda image, identifier: KA)

    def test_cleanup_is_applied_before_recognition_without_changing_source(self):
        from PIL import Image

        with Image.new("RGB", (100, 30), (180, 170, 150)) as image:
            image.putpixel((20, 10), (0, 0, 0))
            image.save(self.image)
        original = self.image.read_bytes()
        observed = []

        def predictor(image, identifier):
            observed.append((image.getpixel((0, 0)), image.getpixel((10, 5))))
            return KA

        result = predict_reference_lines(self.references, self.model,
                                         predictor=predictor, cleanup_mode="normalized")
        self.assertGreater(observed[0][0][0], 180)
        self.assertEqual(observed[0][1], (0, 0, 0))
        self.assertEqual(self.image.read_bytes(), original)
        self.assertEqual(result["cleanup_mode"], "normalized")


class DetectedLineEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.references = make_references(KA, KHA)
        for index, line in enumerate(self.references["lines"]):
            line.update(page=18, source_size=[2000, 1000],
                        bbox=[200, 100 + 200 * index, 1800, 200 + 200 * index])
        self.reports = {"pages": [{"page": 18, "size": [4000, 2000], "provenance": "model",
                                    "segmentation": {"lines": []}, "lines": []}]}
        self.add_line("detected-first", KA, [400, 200, 3600, 400])
        self.add_line("detected-second", KHA, [400, 600, 3600, 800])

    def add_line(self, identifier, text, box):
        left, top, right, bottom = box
        page = self.reports["pages"][0]
        page["segmentation"]["lines"].append({"id": identifier,
            "baseline": [[left, (top + bottom) / 2], [right, (top + bottom) / 2]],
            "boundary": [[left, top], [right, top], [right, bottom], [left, bottom]]})
        page["lines"].append({"line_id": identifier, "raw_hypothesis": text})

    def test_scaled_geometry_matches_without_id_equality_and_does_not_mutate_inputs(self):
        original = copy.deepcopy((self.references, self.reports))
        result = evaluate_detected_lines(self.references, self.reports)
        self.assertEqual(result["cer"], 0)
        self.assertEqual(result["lines"][0]["matched_line_ids"], ["detected-first"])
        self.assertEqual(result["scope"], "end_to_end_annotated_regions")
        self.assertEqual(result["annotated_line_coverage"], 1)
        self.assertEqual((self.references, self.reports), original)

    def test_missing_annotated_line_is_retained_and_counted(self):
        self.reports["pages"][0]["segmentation"]["lines"].pop()
        self.reports["pages"][0]["lines"].pop()
        result = evaluate_detected_lines(self.references, self.reports)
        self.assertEqual(result["missing_annotated_line_count"], 1)
        self.assertEqual(result["deletions"], 1)
        self.assertEqual(result["cer"], 0.5)
        self.assertEqual(result["annotated_line_coverage"], 0.5)
        self.assertEqual(result["lines"][1]["prediction_text"], "")

    def test_fragments_are_combined_left_to_right_without_added_spaces(self):
        page = self.reports["pages"][0]
        page["segmentation"]["lines"] = []
        page["lines"] = []
        self.references["lines"][0]["text"] = KA + KHA
        self.add_line("right", KHA, [2200, 200, 3600, 400])
        self.add_line("left", KA, [400, 200, 1800, 400])
        self.add_line("other-row", KHA, [400, 600, 3600, 800])
        result = evaluate_detected_lines(self.references, self.reports)
        self.assertEqual(result["cer"], 0)
        self.assertEqual(result["split_count"], 1)
        self.assertEqual(result["lines"][0]["matched_line_ids"], ["left", "right"])
        self.assertEqual(result["raw_predictions"]["lines"][0]["text"], KA + KHA)

    def test_unlabelled_page_line_is_not_a_false_positive(self):
        self.add_line("unlabelled", "English", [400, 1400, 3600, 1600])
        result = evaluate_detected_lines(self.references, self.reports)
        self.assertEqual(result["cer"], 0)
        self.assertEqual(result["insertions"], 0)
        self.assertEqual(result["unmatched_detected_line_ids"], ["unlabelled"])
        self.assertNotIn("false_positive_count", result)

    def test_assignment_does_not_change_when_reference_strings_are_swapped(self):
        first = evaluate_detected_lines(self.references, self.reports)
        self.references["lines"][0]["text"], self.references["lines"][1]["text"] = KHA, KA
        second = evaluate_detected_lines(self.references, self.reports)
        self.assertEqual([line["matched_line_ids"] for line in first["lines"]],
                         [line["matched_line_ids"] for line in second["lines"]])
        self.assertEqual(second["cer"], 1)

    def test_ambiguous_line_is_not_reused_across_references(self):
        self.references["lines"][1]["bbox"] = self.references["lines"][0]["bbox"][:]
        result = evaluate_detected_lines(self.references, self.reports)
        self.assertEqual(result["missing_annotated_line_count"], 2)
        self.assertEqual(result["ambiguous_detected_line_ids"], ["detected-first"])
        self.assertTrue(all(not line["matched_line_ids"] for line in result["lines"]))

    def test_fragment_shared_between_regions_withholds_both_whole_regions(self):
        page = self.reports["pages"][0]
        page["segmentation"]["lines"] = []
        page["lines"] = []
        self.references["lines"][0]["bbox"] = [200, 100, 1200, 200]
        self.references["lines"][1]["bbox"] = [1000, 100, 1800, 200]
        self.add_line("left", KA, [400, 200, 1600, 400])
        self.add_line("shared", "", [1800, 200, 2600, 400])
        self.add_line("right", KHA, [2800, 200, 3600, 400])
        result = evaluate_detected_lines(self.references, self.reports)
        self.assertEqual(result["missing_annotated_line_count"], 2)
        self.assertEqual(result["cer"], 1)
        self.assertEqual(result["ambiguous_detected_line_ids"], ["shared"])
        self.assertEqual([line["candidate_line_ids"] for line in result["lines"]], [["left"], ["right"]])
        for line in result["lines"]:
            self.assertTrue(line["geometry_ambiguous"])
            self.assertEqual(line["shared_candidate_line_ids"], ["shared"])
            self.assertEqual(line["matched_line_ids"], [])

    def test_failed_recognition_is_flagged_and_scored_missing_not_clean(self):
        page = self.reports["pages"][0]
        page["lines"][1].update(raw_hypothesis="", recognition_error="RuntimeError: failed")
        self.add_line("unlabelled-failed", "", [400, 1400, 3600, 1600])
        page["lines"][-1]["recognition_error"] = "RuntimeError: failed"
        page.update(status="review", errors=["Line detected-second: RuntimeError: failed"])
        self.reports["status"] = "completed_with_errors"
        original = copy.deepcopy(self.reports)
        result = evaluate_detected_lines(self.references, self.reports)
        self.assertEqual(self.reports, original)
        self.assertEqual(result["recognition_failed_line_ids"], ["detected-second", "unlabelled-failed"])
        self.assertEqual(result["recognition_failed_line_count"], 2)
        self.assertEqual(result["recognition_failed_annotated_line_ids"], ["line-2"])
        self.assertEqual(result["unmatched_detected_line_ids"], ["unlabelled-failed"])
        self.assertEqual((result["missing_annotated_line_count"], result["errors"], result["insertions"]), (1, 1, 0))
        self.assertEqual(result["cer"], 0.5)
        first, second = result["lines"]
        self.assertFalse(first["recognition_failed"])
        self.assertEqual(first["matched_line_ids"], ["detected-first"])
        self.assertTrue(second["recognition_failed"] and second["missing_prediction"])
        self.assertEqual(second["recognition_failed_line_ids"], ["detected-second"])
        self.assertEqual((result["source_report_status"], result["pages_with_errors"]),
                         ("completed_with_errors", [18]))

    def test_recognition_run_without_detected_lines_scores_regions_missing(self):
        page = self.reports["pages"][0]
        page.update(provenance=None, status="review", lines=[], uncertainty_flags=["segmentation_failed"],
                    errors=["Segmentation: RuntimeError: failed"])
        page["segmentation"]["lines"] = []
        config = {"mode": "recognition", "model": "model.safetensors", "references_enabled": False}
        self.reports["config"] = config
        result = evaluate_detected_lines(self.references, self.reports)
        self.assertEqual((result["missing_annotated_line_count"], result["cer"]), (2, 1))
        self.assertEqual((result["no_detection_page_ids"], result["pages_with_errors"]), ([18], [18]))
        for change in ({"mode": "segment_only", "model": None}, {"model": ""}, {"references_enabled": True}):
            self.reports["config"] = {**config, **change}
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "model provenance"):
                evaluate_detected_lines(self.references, self.reports)
        self.reports["config"] = config
        for key, value in (("status", "segmented"), ("reference", {"text": KA})):
            reports = copy.deepcopy(self.reports)
            reports["pages"][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "model provenance"):
                evaluate_detected_lines(self.references, reports)

    def test_source_identity_is_propagated_only_when_provided(self):
        result = evaluate_detected_lines(self.references, self.reports)
        self.assertEqual((result["source_run_id"], result["model_sha256"]), (None, None))
        self.reports.update(run_id="run-1", config={"mode": "recognition", "model": "model.safetensors",
                                                    "references_enabled": False, "model_sha256": "f" * 64})
        result = evaluate_detected_lines(self.references, self.reports)
        self.assertEqual((result["source_run_id"], result["model_sha256"]), ("run-1", "f" * 64))

    def test_overlapping_fragments_are_not_combined_using_text(self):
        self.add_line("duplicate-region", KA, [400, 200, 3600, 400])
        result = evaluate_detected_lines(self.references, self.reports)
        self.assertEqual(result["missing_annotated_line_count"], 1)
        self.assertTrue(result["lines"][0]["geometry_ambiguous"])
        self.assertEqual(result["split_count"], 1)

    def test_overlap_gate_rejects_a_large_box_with_centre_inside_roi(self):
        page = self.reports["pages"][0]
        page["segmentation"]["lines"] = []
        page["lines"] = []
        self.add_line("oversized", KA, [0, 0, 4000, 1000])
        result = evaluate_detected_lines(self.references, self.reports)
        self.assertEqual(result["missing_annotated_line_count"], 2)

    def test_cleanup_inverse_affine_precedes_preview_scaling(self):
        page = self.reports["pages"][0]
        page["preprocessing"] = {"cleanup": {"original_size": [4000, 2000],
            "prepared_size": [3900, 1950], "to_original": [[1, 0, 100], [0, 1, 50], [0, 0, 1]]}}
        for geometry in page["segmentation"]["lines"]:
            for name in ("baseline", "boundary"):
                geometry[name] = [[point[0] - 100, point[1] - 50] for point in geometry[name]]
        self.assertEqual(evaluate_detected_lines(self.references, self.reports)["cer"], 0)

    def test_preview_dimensions_can_be_read_without_opening_reference_crops(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "preview.png"
            Image.new("RGB", (2000, 1000), "white").save(source)
            for line in self.references["lines"]:
                del line["source_size"]
                line["source_preview"] = str(source)
            self.assertEqual(evaluate_detected_lines(self.references, self.reports)["cer"], 0)

    def test_nonmodel_missing_page_duplicate_ids_and_bad_geometry_are_rejected(self):
        for kind in ("reference", "missing-page", "duplicate-page", "duplicate-prediction",
                     "duplicate-geometry", "bad-geometry", "nonfinite", "missing-affine"):
            reports = copy.deepcopy(self.reports)
            page = reports["pages"][0]
            if kind == "reference":
                page["provenance"] = "reference"
            elif kind == "missing-page":
                page["page"] = 19
            elif kind == "duplicate-page":
                reports["pages"].append(copy.deepcopy(page))
            elif kind == "duplicate-prediction":
                page["lines"].append(copy.deepcopy(page["lines"][0]))
            elif kind == "duplicate-geometry":
                page["segmentation"]["lines"].append(copy.deepcopy(page["segmentation"]["lines"][0]))
            elif kind == "bad-geometry":
                page["segmentation"]["lines"][0]["boundary"][0] = [-1, 0]
            elif kind == "nonfinite":
                page["segmentation"]["lines"][0]["baseline"][0][0] = float("nan")
            else:
                page["preprocessing"] = {"cleanup": {"mode": "normalized"}}
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                evaluate_detected_lines(self.references, reports)

    def test_duplicate_ids_across_pages_and_missing_recognition_provenance_are_rejected(self):
        second = copy.deepcopy(self.reports["pages"][0])
        second["page"] = 19
        self.reports["pages"].append(second)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate_detected_lines(self.references, self.reports)
        self.reports["pages"].pop()
        self.reports["pages"][0]["provenance"] = None
        with self.assertRaisesRegex(ValueError, "model provenance"):
            evaluate_detected_lines(self.references, self.reports)

    def test_genuine_blank_page_counts_all_annotated_lines_as_missing(self):
        page = self.reports["pages"][0]
        page.update(status="blank", provenance=None, lines=[])
        page["segmentation"]["lines"] = []
        result = evaluate_detected_lines(self.references, self.reports)
        self.assertEqual(result["missing_annotated_line_count"], 2)
        self.assertEqual(result["cer"], 1)
        page["status"] = "segmented"
        with self.assertRaisesRegex(ValueError, "model provenance"):
            evaluate_detected_lines(self.references, self.reports)

    def test_cli_saves_raw_region_predictions_and_sources_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = write_result(self.references, root / "references.json")
            reports = write_result(self.reports, root / "pages.json")
            output = root / "score.json"
            arguments = ["--references", str(references), "--page-report", str(reports), "--output", str(output)]
            self.assertEqual(main(arguments), 0)
            saved = json.loads((root / "score.predictions.json").read_text())
            self.assertEqual(saved["lines"][0]["matched_line_ids"], ["detected-first"])
            self.assertEqual(json.loads(output.read_text())["scope"], "end_to_end_annotated_regions")
            self.assertEqual(json.loads(output.read_text())["source_report_sha256"],
                             hashlib.sha256(reports.read_bytes()).hexdigest())
            for option in ("--predictions", "--model"):
                with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(arguments + [option, str(reports)])


class LineMetricTests(unittest.TestCase):
    def test_known_substitution_insertion_and_deletion(self):
        for reference, prediction, counts, pair in (
            (KA, KHA, (1, 0, 0), "U+11208 -> U+11209"),
            (KA, KA + KHA, (0, 1, 0), "<empty> -> U+11209"),
            (KA + KHA, KA, (0, 0, 1), "U+11209 -> <empty>"),
        ):
            with self.subTest(counts=counts):
                result = score_line(reference, prediction)
                self.assertEqual(tuple(result[name] for name in
                                       ("substitutions", "insertions", "deletions")), counts)
                self.assertEqual(result["errors"], 1)
                self.assertEqual(result["cer"], 1 / len(reference))
                self.assertEqual(result["confusion_pairs"], {pair: 1})

    def test_invalid_latin_is_retained_and_penalized(self):
        result = score_line(KA, KA + "English")
        self.assertEqual(result["prediction_text"], KA + "English")
        self.assertEqual(result["insertions"], 7)
        self.assertEqual(result["cer"], 7.0)
        self.assertIn("U+0045", result["invalid_codepoints"])

    def test_shared_nfc_and_line_ending_normalization(self):
        result = score_line("e\u0301\r\n" + KA + "\r", "\u00e9\n" + KA + "\n")
        self.assertEqual(result["reference_characters"], 4)
        self.assertEqual(result["prediction_characters"], 4)
        self.assertEqual(result["cer"], 0.0)
        self.assertTrue(result["exact_match"])

    def test_non_bmp_letters_are_one_code_point_each(self):
        result = score_line(KA + KHA, KA)
        self.assertEqual(result["reference_characters"], 2)
        self.assertEqual(result["prediction_characters"], 1)
        self.assertEqual(result["deletions"], 1)
        self.assertEqual(result["cer"], 0.5)

    def test_spaces_newlines_and_punctuation_count(self):
        result = score_line(KA + " \n\U00011238", KA)
        self.assertEqual(result["reference_characters"], 4)
        self.assertEqual(result["deletions"], 3)
        self.assertEqual(result["cer"], 0.75)
        self.assertIn("U+000A -> <empty>", result["confusion_pairs"])

    def test_empty_references(self):
        blank = score_line("", "")
        self.assertEqual(blank["cer"], 0.0)
        self.assertTrue(blank["exact_match"])
        inserted = score_line("", KA)
        self.assertIsNone(inserted["cer"])
        self.assertEqual(inserted["insertions"], 1)
        self.assertFalse(inserted["exact_match"])


class EvaluationTests(unittest.TestCase):
    def test_global_cer_uses_summed_counts_not_mean_line_cer(self):
        result = evaluate(make_references(KA, KA, KA * 4), make_predictions(KHA, KA + KHA, KA * 3))
        self.assertEqual(result["reference_characters"], 6)
        self.assertEqual(result["errors"], 3)
        self.assertEqual(result["cer"], 0.5)
        self.assertEqual(result["line_count"], 3)
        self.assertEqual(result["exact_line_accuracy"], 0.0)
        self.assertEqual((result["substitutions"], result["insertions"], result["deletions"]), (1, 1, 1))

    def test_missing_prediction_is_empty_not_excluded(self):
        result = evaluate(make_references(KA, KA + KHA), make_predictions(KA))
        self.assertEqual(result["line_count"], 2)
        self.assertEqual(result["missing_prediction_count"], 1)
        self.assertEqual(result["deletions"], 2)
        self.assertEqual(result["cer"], 2 / 3)
        self.assertEqual(result["exact_line_accuracy"], 0.5)
        self.assertTrue(result["lines"][1]["missing_prediction"])
        self.assertEqual(result["lines"][1]["prediction_text"], "")

    def test_ids_not_position_control_alignment(self):
        result = evaluate(make_references(KA, KHA), {"lines": list(reversed(make_predictions(KA, KHA)))})
        self.assertEqual(result["cer"], 0.0)
        self.assertEqual(result["exact_line_accuracy"], 1.0)

    def test_unknown_ids_including_blank_predictions_raise(self):
        for text in ("", KA):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "Unknown prediction IDs"):
                evaluate(make_references(KA), make_predictions(KA, text))

    def test_duplicate_reference_and_prediction_ids_raise(self):
        references = make_references(KA)
        references["lines"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate reference id"):
            evaluate(references, make_predictions(KA))
        with self.assertRaisesRegex(ValueError, "Duplicate prediction id"):
            evaluate(make_references(KA), make_predictions(KA) * 2)

    def test_unreviewed_and_pseudo_gold_references_raise(self):
        for status in ("needs_review", "pseudoGold", "pseudo_gold", "draft", None):
            references = make_references(KA)
            references["lines"][0]["status"] = status
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "reviewed or approved"):
                evaluate(references, make_predictions(KA))
        references["lines"][0]["status"] = "approved"
        self.assertEqual(evaluate(references, make_predictions(KA))["cer"], 0.0)

    def test_policy_version_is_required_and_must_match(self):
        for policy in (None, "different-policy"):
            references = make_references(KA)
            references["policy_version"] = policy
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "policy_version"):
                evaluate(references, make_predictions(KA))

    def test_malformed_lines_are_rejected(self):
        for line in (None, {"text": KA, "provenance": "model"},
                     {"id": "", "text": KA, "provenance": "model"},
                     {"id": "line-1", "text": None, "provenance": "model"}):
            with self.subTest(line=line), self.assertRaises(ValueError):
                evaluate(make_references(KA), [line])

    def test_provenance_is_required_even_with_override(self):
        for allow_reference in (False, True):
            for provenance in (None, "", " "):
                with self.subTest(provenance=provenance, override=allow_reference):
                    with self.assertRaisesRegex(ValueError, "provenance is required"):
                        evaluate(make_references(KA), make_predictions(KA, provenance=provenance),
                                 allow_reference=allow_reference)
            with self.assertRaisesRegex(ValueError, "provenance is required"):
                evaluate(make_references(KA), [{"id": "line-1", "text": KA}],
                         allow_reference=allow_reference)

    def test_reference_cache_blocked_and_override_labels_non_ocr(self):
        for provenance in ("reference", "cache"):
            predictions = make_predictions(KA, provenance=provenance)
            with self.subTest(provenance=provenance):
                with self.assertRaisesRegex(ValueError, "not OCR accuracy"):
                    evaluate(make_references(KA), predictions)
                result = evaluate(make_references(KA), predictions, allow_reference=True)
                self.assertEqual(result["result_mode"], "non_model_comparison")
                self.assertIn("not OCR accuracy", " ".join(result["warnings"]))
        mixed = make_predictions(KA, KHA)
        mixed[1]["provenance"] = "reference"
        result = evaluate(make_references(KA, KHA), mixed, allow_reference=True)
        self.assertEqual(result["result_mode"], "non_model_comparison")
        self.assertEqual(result["prediction_provenances"], ["model", "reference"])

    def test_container_provenance_cannot_be_overridden_by_model_lines(self):
        predictions = {"provenance": "reference", "lines": make_predictions(KA)}
        with self.assertRaisesRegex(ValueError, "Non-model provenance"):
            evaluate(make_references(KA), predictions)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            evaluate(make_references(KA), predictions, allow_reference=True)

    def test_evidence_kind_is_explicit_or_warned(self):
        references = make_references(KA)
        del references["dataset_kind"]
        result = evaluate(references, make_predictions(KA))
        self.assertEqual(result["dataset_kind"], "unspecified")
        self.assertIn("unspecified", " ".join(result["warnings"]))
        result = evaluate(make_references(KA, dataset_kind="synthetic"), make_predictions(KA))
        self.assertEqual(result["dataset_kind"], "synthetic")
        self.assertIn("real-only", " ".join(result["warnings"]))
        with self.assertRaisesRegex(ValueError, "dataset_kind"):
            evaluate(make_references(KA, dataset_kind="guessed"), make_predictions(KA))

    def test_zero_global_denominators_and_empty_dataset(self):
        empty = evaluate(make_references(), [])
        self.assertEqual(empty["line_count"], 0)
        self.assertEqual(empty["cer"], 0.0)
        self.assertIsNone(empty["exact_line_accuracy"])
        self.assertEqual(evaluate(make_references(""), make_predictions(""))["cer"], 0.0)
        self.assertIsNone(evaluate(make_references(""), make_predictions(KA))["cer"])
        result = evaluate(make_references("", KA), make_predictions(KA, ""))
        self.assertEqual(result["reference_characters"], 1)
        self.assertEqual(result["errors"], 2)
        self.assertEqual(result["cer"], 2.0)

    def test_confusion_frequency_invalid_codepoints_and_metadata(self):
        references = make_references(KA, KA, KA)
        references["lines"][0].update({"image": "line.png", "bbox": [0, 0, 20, 10]})
        predictions = make_predictions(KHA, KHA, KA + "A")
        originals = copy.deepcopy((references, predictions))
        result = evaluate(references, predictions)
        self.assertEqual(list(result["confusion_pairs"].items()),
                         [("U+11208 -> U+11209", 2), ("<empty> -> U+0041", 1)])
        self.assertEqual(result["invalid_codepoints"], ["U+0041"])
        self.assertEqual(result["lines"][0]["reference_metadata"]["bbox"], [0, 0, 20, 10])
        self.assertEqual((references, predictions), originals)


class PipelineNormalizerTests(unittest.TestCase):
    def test_raw_pipeline_aggregate_flattens_explicit_fields(self):
        predictions = {"pages": [
            {"page": 1, "provenance": "model", "lines": [
                {"line_id": "line-1", "raw_hypothesis": KA + "A", "normalized_hypothesis": KA}]},
            {"page": 2, "provenance": "model", "lines": [
                {"line_id": "line-2", "raw_hypothesis": KHA}]},
        ]}
        original = copy.deepcopy(predictions)
        normalized = normalize_predictions(predictions)
        self.assertEqual(normalized["input_format"], "pipeline_aggregate")
        self.assertEqual(normalized["lines"][0]["text"], KA + "A")
        self.assertEqual(normalized["lines"][1]["page"], 2)
        result = evaluate(make_references(KA, KHA), predictions)
        self.assertEqual(result["insertions"], 1)
        self.assertEqual(result["cer"], 0.5)
        self.assertEqual(predictions, original)

    def test_duplicate_pipeline_ids_across_pages_raise(self):
        page = {"page": 1, "provenance": "model", "lines": [
            {"line_id": "line-1", "raw_hypothesis": KA}]}
        with self.assertRaisesRegex(ValueError, "Duplicate prediction id"):
            normalize_predictions({"pages": [page, {**page, "page": 2}]})

    def test_empty_reference_cache_page_still_blocks_model_evaluation(self):
        predictions = {"pages": [{"page": 1, "provenance": "reference", "lines": [],
                                   "reference": {"text": KA}}]}
        with self.assertRaisesRegex(ValueError, "Non-model provenance"):
            evaluate(make_references(KA), predictions)
        result = evaluate(make_references(KA), predictions, allow_reference=True)
        self.assertEqual(result["result_mode"], "non_model_comparison")
        self.assertEqual(result["missing_prediction_count"], 1)

    def test_missing_or_conflicting_page_provenance_raises(self):
        for allow_reference in (False, True):
            with self.assertRaisesRegex(ValueError, "provenance is required"):
                normalize_predictions({"pages": [{"lines": []}]}, allow_reference=allow_reference)
        predictions = {"pages": [{"provenance": "reference", "lines": [
            {"line_id": "line-1", "raw_hypothesis": KA, "provenance": "model"}]}]}
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            normalize_predictions(predictions, allow_reference=True)

    def test_no_draft_fallback_or_ambiguous_shape(self):
        for predictions in ({"pages": [], "lines": []}, {"line-1": KA},
                            {"lines": {"line-1": KA}},
                            {"pages": [{"provenance": "model", "lines": [
                                {"line_id": "line-1", "draft_text": KA}]}]}):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                normalize_predictions(predictions)


class EvaluationCLITests(unittest.TestCase):
    def test_cli_writes_new_json_with_parents_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = write_result(make_references(KA), root / "references.json")
            predictions = write_result({"lines": make_predictions(KHA)}, root / "predictions.json")
            output = root / "new" / "result.json"
            arguments = ["--references", str(references), "--predictions", str(predictions),
                         "--output", str(output)]
            self.assertEqual(main(arguments), 0)
            original = output.read_bytes()
            result = json.loads(original)
            self.assertEqual(result["cer"], 1.0)
            self.assertEqual(result["result_mode"], "model_evaluation")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(arguments)
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(output.read_bytes(), original)

    def test_cli_validation_precedes_directory_creation_and_flag_changes_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = write_result(make_references(KA), root / "references.json")
            predictions = write_result({"lines": make_predictions(KA, provenance="reference")},
                                       root / "predictions.json")
            output = root / "new" / "result.json"
            arguments = ["--references", str(references), "--predictions", str(predictions),
                         "--output", str(output)]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(arguments)
            self.assertFalse(output.parent.exists())
            self.assertEqual(main(arguments + ["--allow-reference"]), 0)
            self.assertEqual(json.loads(output.read_text())["result_mode"], "non_model_comparison")

    def test_serialization_failure_creates_no_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new" / "result.json"
            with self.assertRaises(ValueError):
                write_result({"cer": float("nan")}, output)
            self.assertFalse(output.parent.exists())

    def test_dangling_output_symlink_is_not_followed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            target = Path(directory) / "missing.json"
            output.symlink_to(target)
            with self.assertRaises(FileExistsError):
                write_result({}, output)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()