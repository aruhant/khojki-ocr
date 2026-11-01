import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw, ImageFont

from khojki_ocr.core import POLICY_VERSION, raster_digest
from khojki_ocr.pipeline import (
    confirmed_blank, iter_pages, line_hypothesis, load_recognition_model, main, parse_pages, pdf_caption_regions,
    prepare_output, process_page, raster_statistics, recognize_page, run_pipeline,
    segment_page,
)


KA = "\U00011208"


class SegmentationModeTests(unittest.TestCase):
    def setUp(self):
        with Image.new("RGB", (16, 16), "white") as image:
            raster_statistics(image)
        self.binarization = SimpleNamespace(nlbin=Mock(side_effect=lambda image: image.copy()))
        self.pageseg = SimpleNamespace(segment=Mock(return_value=SimpleNamespace(lines=[])))
        self.config = Mock()
        self.model = Mock()
        self.model.predict.return_value = SimpleNamespace(type="baselines", lines=[], regions={})
        self.loader = Mock(return_value=self.model)
        self.input_error = type("KrakenInputException", (Exception,), {})
        modules = patch.dict("sys.modules", {
            "kraken": SimpleNamespace(binarization=self.binarization, pageseg=self.pageseg),
            "kraken.configs": SimpleNamespace(SegmentationInferenceConfig=self.config),
            "kraken.tasks": SimpleNamespace(SegmentationTaskModel=SimpleNamespace(load_model=self.loader)),
            "kraken.lib.exceptions": SimpleNamespace(KrakenInputException=self.input_error),
        })
        modules.start()
        self.addCleanup(modules.stop)

    def test_invalid_mode_is_rejected_before_processing(self):
        for mode in ("", "unknown", "AUTO", None, True, []):
            with self.subTest(mode=mode):
                image = Mock()
                with self.assertRaisesRegex(ValueError, "Segmentation mode"):
                    segment_page(image, self.model, segmentation_mode=mode)
                image.convert.assert_not_called()
                with patch("khojki_ocr.pipeline.prepare_output") as output, patch(
                        "khojki_ocr.pipeline.iter_pages") as pages:
                    with self.assertRaisesRegex(ValueError, "Segmentation mode"):
                        run_pipeline("missing.png", "unused", segmentation_mode=mode)
                    output.assert_not_called()
                    pages.assert_not_called()
        self.binarization.nlbin.assert_not_called()
        self.pageseg.segment.assert_not_called()
        self.loader.assert_not_called()
        self.model.predict.assert_not_called()

    def test_forced_baselines_bypasses_component_extraction(self):
        with Image.new("RGB", (64, 64), "black") as image:
            before = image.tobytes()
            result = segment_page(image, self.model, segmentation_mode="baselines")
            self.assertIs(result, self.model.predict.return_value)
            self.model.predict.assert_called_once_with(image, self.config.return_value)
            self.assertEqual(image.tobytes(), before)
        self.binarization.nlbin.assert_not_called()
        self.pageseg.segment.assert_not_called()
        self.loader.assert_not_called()

    def test_forced_baselines_retains_wide_image_padding_and_lazy_loading(self):
        with Image.new("RGB", (128, 16), "black") as image:
            before = image.tobytes()

            def predict(padded, config):
                self.assertEqual(padded.size, (128, 64))
                with padded.crop((0, 0, 128, 16)) as original:
                    self.assertEqual(original.tobytes(), before)
                self.assertEqual(padded.getpixel((0, 63)), (255, 255, 255))
                self.assertIs(config, self.config.return_value)
                return SimpleNamespace(type="baselines", lines=[], regions={})

            self.model.predict.side_effect = predict
            segment_page(image, segmentation_mode="baselines")
            self.assertEqual(image.tobytes(), before)
        self.loader.assert_called_once_with()
        self.model.predict.assert_called_once()
        self.binarization.nlbin.assert_not_called()
        self.pageseg.segment.assert_not_called()

    def test_components_without_rows_fails_without_learned_fallback(self):
        for color in ("white", "black"):
            with self.subTest(color=color), Image.new("RGB", (64, 64), color) as image:
                self.pageseg.segment.reset_mock()
                before = image.tobytes()
                with self.assertRaisesRegex(ValueError, "Component segmentation could not extract rows"):
                    segment_page(image, segmentation_mode="components")
                self.assertEqual(image.tobytes(), before)
                if color == "black":
                    self.pageseg.segment.assert_called_once()
                    self.assertEqual(self.pageseg.segment.call_args.kwargs,
                                     {"text_direction": "horizontal-lr", "black_colseps": True})
                else:
                    self.pageseg.segment.assert_not_called()
        self.loader.assert_not_called()
        self.model.predict.assert_not_called()

    def test_components_extraction_errors_are_explicit_and_auto_still_falls_back(self):
        for error in (ValueError("bad image"), self.input_error("bad rows")):
            with self.subTest(error=error), Image.new("RGB", (64, 64), "black") as image:
                self.pageseg.segment.side_effect = error
                with self.assertRaisesRegex(ValueError, "Component segmentation") as failure:
                    segment_page(image, self.model, segmentation_mode="components")
                self.assertIs(failure.exception.__cause__, error)
                self.model.predict.assert_not_called()
                self.assertIs(segment_page(image, self.model), self.model.predict.return_value)
                self.model.predict.assert_called_once()
                self.model.predict.reset_mock()
        self.loader.assert_not_called()

    def test_pipeline_forwards_mode_records_config_and_preserves_auto_callback(self):
        with tempfile.TemporaryDirectory() as directory, Image.new("RGB", (64, 64), "white") as image:
            root = Path(directory)
            source = root / "pages.tiff"
            with image.copy() as second:
                image.save(source, save_all=True, append_images=[second])
            for mode in ("auto", "components", "baselines"):
                with self.subTest(mode=mode):
                    self.loader.reset_mock()

                    def legacy_segment(source, model):
                        return SimpleNamespace(type="bbox", lines=[], regions={})

                    with patch("khojki_ocr.pipeline.segment_page", side_effect=(
                            legacy_segment if mode == "auto" else None)) as segmenter:
                        segmenter.return_value = SimpleNamespace(type="bbox", lines=[], regions={})
                        options = {} if mode == "auto" else {"segmentation_mode": mode}
                        result = run_pipeline(source, root / mode, segment_only=True, **options)
                        self.assertEqual(result["summary"]["processed_pages"], 2)
                        self.assertEqual(result["status"], "completed")
                        self.assertEqual(segmenter.call_count, 2)
                        for call in segmenter.call_args_list:
                            self.assertEqual(len(call.args), 2)
                            self.assertIs(call.args[1], None if mode == "components" else self.model)
                            self.assertEqual(call.kwargs, {} if mode == "auto" else {"segmentation_mode": mode})
                    self.assertEqual(result["config"]["segmentation_mode"], mode)
                    stored = json.loads((root / mode / "report.json").read_text(encoding="utf-8"))
                    self.assertEqual(stored["config"]["segmentation_mode"], mode)
                    self.assertEqual(result["models_loaded"]["segmentation"], mode != "components")
                    if mode == "components":
                        self.loader.assert_not_called()
                    else:
                        self.loader.assert_called_once_with()

    def test_pipeline_reports_component_failure_without_loading_detector(self):
        with tempfile.TemporaryDirectory() as directory, Image.new("RGB", (64, 64), "black") as image:
            root = Path(directory)
            source = root / "input.png"
            image.save(source)
            result = run_pipeline(source, root / "run", segment_only=True, segmentation_mode="components")
        self.assertEqual(result["status"], "completed_with_errors")
        self.assertIn("segmentation_failed", result["pages"][0]["uncertainty_flags"])
        self.assertIn("Component segmentation could not extract rows", result["pages"][0]["errors"][0])
        self.loader.assert_not_called()
        self.model.predict.assert_not_called()

    def test_cli_forwards_segmentation_mode_and_defaults_to_auto(self):
        for mode in (None, "auto", "components", "baselines"):
            with self.subTest(mode=mode), patch("khojki_ocr.pipeline.run_pipeline") as pipeline, patch("builtins.print"):
                pipeline.return_value = {"status": "completed", "summary": {}}
                arguments = ["--input", "input.png", "--output", "run", "--segment-only"]
                if mode is not None:
                    arguments.extend(["--segmentation-mode", mode])
                self.assertEqual(main(arguments), 0)
                self.assertEqual(pipeline.call_args.kwargs["segmentation_mode"], mode or "auto")
                self.assertTrue(pipeline.call_args.kwargs["segment_only"])

    def test_cli_rejects_invalid_mode_before_running_pipeline(self):
        with patch("khojki_ocr.pipeline.run_pipeline") as pipeline, patch("sys.stderr"):
            with self.assertRaises(SystemExit) as failure:
                main(["--input", "input.png", "--output", "run", "--segmentation-mode", "unknown"])
        self.assertEqual(failure.exception.code, 2)
        pipeline.assert_not_called()


class RecognitionInputContractTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not yet available")
        self.torch = torch
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.weights = Path(temporary.name) / "model.safetensors"
        payload = b"mocked recognition weights"
        self.weights.write_bytes(payload)
        self.sidecar = self.weights.with_suffix(".input.json")
        self.contract = {"schema": "khojki-ppocrv6-input-v1",
                         "model_sha256": hashlib.sha256(payload).hexdigest(),
                         "max_width": 64, "padding": 16}
        self.sidecar.write_text(json.dumps(self.contract), encoding="utf-8")
        self.model = Mock()
        self.loader = Mock(return_value=self.model)
        tasks = SimpleNamespace(RecognitionTaskModel=SimpleNamespace(load_model=self.loader))
        modules = patch.dict("sys.modules", {"kraken.tasks": tasks})
        modules.start()
        self.addCleanup(modules.stop)

    def test_without_contract_returns_original_model_without_hook(self):
        self.sidecar.unlink()
        self.assertIs(load_recognition_model(str(self.weights)), self.model)
        self.loader.assert_called_once_with(self.weights)
        self.model.net.nn.register_forward_pre_hook.assert_not_called()
        self.assertNotIn("input_contract", vars(self.model))

    def test_required_missing_contract_fails_without_registering_a_hook(self):
        self.sidecar.unlink()
        with self.assertRaisesRegex(ValueError, "Missing recognition input contract"):
            load_recognition_model(self.weights, require_contract=True)
        self.assertNotIn("input_contract", vars(self.model))

    def test_valid_contract_accepts_width_boundaries_and_registers_one_hook(self):
        for width in (64, 4096):
            with self.subTest(width=width):
                self.contract["max_width"] = width
                self.sidecar.write_text(json.dumps(self.contract), encoding="utf-8")
                self.model.reset_mock()
                self.assertIs(load_recognition_model(self.weights), self.model)
                self.assertEqual(self.model.input_contract, self.contract)
                self.model.net.nn.register_forward_pre_hook.assert_called_once()
                hook = self.model.net.nn.register_forward_pre_hook.call_args.args[0]
                self.assertTrue(callable(hook))

    def test_invalid_width_including_bool_is_rejected_without_hook(self):
        for width in (-1, 0, 63, 4097, True, False, 64.0, "64", None):
            with self.subTest(width=width):
                self.contract["max_width"] = width
                self.sidecar.write_text(json.dumps(self.contract), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "input contract"):
                    load_recognition_model(self.weights)
                self.model.net.nn.register_forward_pre_hook.assert_not_called()

    def test_schema_hash_and_padding_mismatches_are_rejected_without_hook(self):
        for field, value in (("schema", "unsupported-v2"), ("schema", None),
                             ("model_sha256", "0" * 64), ("model_sha256", None),
                             ("padding", 0), ("padding", 15), ("padding", 17),
                             ("padding", True), ("padding", "16"), ("padding", None)):
            with self.subTest(field=field, value=value):
                contract = {**self.contract, field: value}
                self.sidecar.write_text(json.dumps(contract), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "input contract"):
                    load_recognition_model(self.weights)
                self.model.net.nn.register_forward_pre_hook.assert_not_called()

    def test_changed_model_bytes_invalidate_previously_matching_hash(self):
        self.weights.write_bytes(b"tampered recognition weights")
        with self.assertRaisesRegex(ValueError, "input contract"):
            load_recognition_model(self.weights)
        self.model.net.nn.register_forward_pre_hook.assert_not_called()

    def test_hook_right_pads_zeros_preserving_prefix_and_sequence_lengths(self):
        load_recognition_model(self.weights)
        hook = self.model.net.nn.register_forward_pre_hook.call_args.args[0]
        for width in (31, 64):
            with self.subTest(width=width):
                image = self.torch.arange(1, 2 * 3 * width + 1, dtype=self.torch.float32).reshape(2, 1, 3, width)
                original = image.clone()
                lengths = self.torch.tensor([width, width - 3])
                original_lengths = lengths.clone()
                extra_argument = object()
                arguments = hook(self.model.net.nn, (image, lengths, extra_argument))
                self.assertEqual(len(arguments), 3)
                padded = arguments[0]
                self.assertEqual(tuple(padded.shape), (2, 1, 3, 64))
                self.assertEqual(padded.dtype, image.dtype)
                self.assertEqual(padded.device, image.device)
                self.assertTrue(self.torch.equal(padded[..., :width], original))
                self.assertEqual(self.torch.count_nonzero(padded[..., width:]).item(), 0)
                self.assertTrue(self.torch.equal(image, original))
                self.assertIs(arguments[1], lengths)
                self.assertTrue(self.torch.equal(lengths, original_lengths))
                self.assertIs(arguments[2], extra_argument)

    def test_oversize_input_fails_before_padding_or_clipping(self):
        load_recognition_model(self.weights)
        hook = self.model.net.nn.register_forward_pre_hook.call_args.args[0]
        image = self.torch.ones((1, 1, 3, 65))
        lengths = self.torch.tensor([65])
        with patch("torch.nn.functional.pad") as pad:
            with self.assertRaisesRegex(ValueError, "refusing to clip"):
                hook(self.model.net.nn, (image, lengths))
            pad.assert_not_called()
        self.assertEqual(tuple(image.shape), (1, 1, 3, 65))
        self.assertTrue(self.torch.equal(image, self.torch.ones_like(image)))
        self.assertEqual(lengths.tolist(), [65])


class PipelinePolicyTests(unittest.TestCase):
    def test_clear_printed_rows_keep_complete_glyphs_and_reading_order(self):
        font_path = Path(__file__).resolve().parents[1] / "web/assets/NotoSansKhojki-Regular.ttf"
        font = ImageFont.truetype(str(font_path), 64, layout_engine=ImageFont.Layout.RAQM)
        for framed in (False, True):
            with self.subTest(framed=framed), Image.new("RGB", (1000, 720), "white") as image:
                model = Mock()
                draw = ImageDraw.Draw(image)
                if framed:
                    draw.rectangle((0, 0, 999, 719), outline="black", width=12)
                boxes = []
                for row, base in enumerate((0x1120E, 0x11219, 0x11226, 0x11229)):
                    text = "\U0001123A".join(
                        chr(base) + (chr(vowel) if vowel is not None else "")
                        for vowel in (None, 0x1122C, 0x1122D, 0x1122F, 0x11230, 0x11232)
                    )
                    position = (80, 160 + row * 135)
                    boxes.append(draw.textbbox(position, text, font=font, anchor="ls", direction="ltr"))
                    draw.text(position, text, font=font, anchor="ls", direction="ltr", fill="black")
                original = image.tobytes()
                segmentation = segment_page(image, model)
                self.assertEqual(segmentation.type, "bbox")
                self.assertEqual(len(segmentation.lines), 4)
                for line, expected in zip(segmentation.lines, boxes):
                    left, top, right, bottom = line.bbox
                    self.assertLessEqual(left, expected[0] + 2)
                    self.assertLessEqual(top, expected[1] + 2)
                    self.assertGreaterEqual(right, expected[2] - 2)
                    self.assertGreaterEqual(bottom, expected[3] - 2)
                self.assertEqual(image.tobytes(), original)
                model.predict.assert_not_called()

    def test_empty_classical_segmentation_uses_learned_fallback(self):
        model = Mock()
        result = SimpleNamespace(lines=[])
        model.predict.return_value = result
        with Image.new("RGB", (200, 200), "white") as image, patch(
                "kraken.binarization.nlbin", return_value=image.copy()), patch(
                "kraken.pageseg.segment", return_value=SimpleNamespace(lines=[])):
            self.assertIs(segment_page(image, model), result)
        model.predict.assert_called_once()

    def test_bounding_box_lines_support_reporting_recognition_and_crop_export(self):
        line = SimpleNamespace(id="complete-row", bbox=(8, 9, 58, 31))
        segmentation = SimpleNamespace(type="bbox", lines=[line], regions={})
        record = SimpleNamespace(id=line.id, prediction=KA, confidences=[0.9])
        with tempfile.TemporaryDirectory() as directory, Image.new("RGB", (64, 64), "white") as image:
            image.putpixel((8, 9), (0, 0, 0))
            result = process_page(image, 1, segmenter=lambda source: segmentation,
                                  recognizer=lambda source, bounds: iter([record]),
                                  line_crops=directory)
            self.assertEqual(result["segmentation"]["lines"][0]["bbox"], [8, 9, 58, 31])
            self.assertEqual(result["raw_hypothesis"], KA)
            self.assertEqual(result["errors"], [])
            self.assertEqual(len(result["line_crops"]), 1)
            with Image.open(result["line_crops"][0]["path"]) as crop:
                self.assertEqual(crop.size, (50, 22))
                self.assertEqual(crop.getpixel((0, 0)), (0, 0, 0))

    def test_overwide_row_preserves_all_pixels_and_returns_one_line(self):
        from kraken.containers import BBoxLine, Segmentation

        line = BBoxLine(id="wide-row", bbox=(0, 0, 600, 80), base_dir="L")
        segmentation = Segmentation(type="bbox", imagename="test", lines=[line],
                                    text_direction="horizontal-lr", script_detection=False)
        contract = {"schema": "khojki-ppocrv6-input-v1", "max_width": 256, "padding": 16}
        fragments = []

        def predict(source, bounds, config):
            fragment = bounds.lines[0]
            fragments.append(fragment.bbox)
            return [SimpleNamespace(prediction=KA, confidences=[0.9])]

        model = SimpleNamespace(input_contract=contract, predict=predict)
        with Image.new("RGB", (600, 80), "white") as image:
            draw = ImageDraw.Draw(image)
            for start in range(10, 590, 40):
                draw.rectangle((start, 20, start + 15, 60), fill="black")
            original = image.tobytes()
            records = list(recognize_page(image, segmentation, model))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].id, line.id)
            self.assertGreater(len(fragments), 1)
            self.assertEqual(records[0].prediction, KA * len(fragments))
            self.assertEqual(fragments[0][0], 0)
            self.assertEqual(fragments[-1][2], 600)
            for before, after in zip(fragments, fragments[1:]):
                self.assertEqual(before[2], after[0])
                self.assertEqual(image.crop((before[2], 0, before[2] + 1, 80)).convert("L").getextrema(), (255, 255))
            self.assertTrue(all(right - left <= 158 for left, top, right, bottom in fragments))
            self.assertEqual(image.tobytes(), original)

    def test_overwide_row_without_whitespace_fails_instead_of_clipping(self):
        from kraken.containers import BBoxLine, Segmentation

        line = BBoxLine(id="uncuttable", bbox=(0, 0, 600, 80), base_dir="L")
        segmentation = Segmentation(type="bbox", imagename="test", lines=[line],
                                    text_direction="horizontal-lr", script_detection=False)
        model = SimpleNamespace(input_contract={"schema": "khojki-ppocrv6-input-v1",
                                                "max_width": 256, "padding": 16}, predict=Mock())
        with Image.new("RGB", (600, 80), "black") as image:
            with self.assertRaisesRegex(ValueError, "no whitespace split"):
                list(recognize_page(image, segmentation, model))
        model.predict.assert_not_called()

    def test_wide_crop_segmentation_preserves_ink_and_original_coordinates(self):
        configs = SimpleNamespace(SegmentationInferenceConfig=Mock())
        tasks = SimpleNamespace(SegmentationTaskModel=Mock())
        original = Image.new("RGB", (600, 60), "white")
        self.addCleanup(original.close)
        original.putpixel((42, 18), (12, 34, 56))
        before = original.tobytes()
        result = SimpleNamespace(lines=[])

        def predict(padded, config):
            self.assertEqual(padded.size, (600, 300))
            self.assertEqual(padded.crop((0, 0, 600, 60)).tobytes(), before)
            self.assertEqual(padded.getpixel((42, 18)), (12, 34, 56))
            self.assertEqual(padded.getpixel((42, 299)), (255, 255, 255))
            return result

        with patch.dict("sys.modules", {"kraken.configs": configs, "kraken.tasks": tasks}):
            self.assertIs(segment_page(original, SimpleNamespace(predict=predict)), result)
        self.assertEqual(original.tobytes(), before)

    def test_kraken_adapters_use_verified_task_signatures(self):
        segment_config = Mock()
        recognition_config = Mock()
        configs = SimpleNamespace(SegmentationInferenceConfig=segment_config,
                                  RecognitionInferenceConfig=recognition_config)
        task_class = Mock()
        tasks = SimpleNamespace(SegmentationTaskModel=task_class)
        model = Mock()
        segmentation = SimpleNamespace(type="baselines", lines=[], regions={})
        with Image.new("RGB", (16, 16), "white") as image, patch.dict(
                "sys.modules", {"kraken.configs": configs, "kraken.tasks": tasks}):
            segment_page(image, model)
            model.predict.assert_called_once_with(image, segment_config.return_value)
            self.assertTrue(segment_config.call_args.kwargs["raise_on_error"])
            task_class.load_model.assert_not_called()
            model.reset_mock()
            recognize_page(image, segmentation, model)
            model.predict.assert_called_once_with(image, segmentation, recognition_config.return_value)
            self.assertEqual(recognition_config.call_args.kwargs["num_line_workers"], 0)
            self.assertFalse(recognition_config.call_args.kwargs["bidi_reordering"])

    def test_partial_recognition_and_crop_results_survive_later_failures(self):
        line = SimpleNamespace(id="line1", baseline=[(4, 20), (50, 20)],
                               boundary=[(4, 4), (50, 4), (50, 24), (4, 24)])
        segmentation = SimpleNamespace(type="baselines", lines=[line], regions={})

        def recognize(image, bounds):
            yield SimpleNamespace(id="line1", prediction=KA, confidences=[0.9])
            raise RuntimeError("second line failed")

        def crops(*args):
            yield {"line_id": "line1", "path": "private/line.png", "size": [12, 8]}
            raise RuntimeError("second crop failed")

        with Image.new("RGB", (64, 64), "white") as image, patch(
                "khojki_ocr.pipeline.extract_line_crops", side_effect=crops):
            result = process_page(image, 1, segmenter=Mock(return_value=segmentation),
                                  recognizer=recognize, line_crops="unused")
            self.assertEqual(result["raw_hypothesis"], KA)
            self.assertEqual(result["line_crops"][0]["path"], "private/line.png")
            self.assertIn("recognition_failed", result["uncertainty_flags"])
            self.assertIn("line_crop_export_failed", result["uncertainty_flags"])
            self.assertEqual(result["status"], "review")

    def test_unapproved_or_changed_policy_reference_never_exports(self):
        with tempfile.TemporaryDirectory() as directory, Image.new("RGB", (16, 16), "white") as image:
            digest = raster_digest(image.tobytes(), image.width, image.height)
            entry = {"status": "approved", "reviewed_by": "reviewer", "decision": "transcription", "text": KA}
            manifests = [
                {"policy_version": "other", "pages": {digest: entry}},
                {"policy_version": POLICY_VERSION, "pages": {digest: {**entry, "reviewed_by": ""}}},
                {"policy_version": POLICY_VERSION, "pages": {digest: {**entry, "status": "draft"}}},
            ]
            for manifest in manifests:
                segmenter = Mock(return_value=SimpleNamespace(type="baselines", lines=[], regions={}))
                result = process_page(image, 1, segmenter=segmenter, manifest=manifest, output_dir=directory)
                self.assertIsNone(result["provenance"])
                self.assertIsNone(result["transcript_path"])
                segmenter.assert_called_once()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_failed_region_does_not_hide_following_recognition(self):
        lines = [SimpleNamespace(id=name, baseline=[(4, 20), (50, 20)],
                                 boundary=[(4, 4), (50, 4), (50, 24), (4, 24)])
                 for name in ("bad", "good")]
        bounds = SimpleNamespace(type="baselines", lines=lines, regions={})
        records = [SimpleNamespace(id="bad", prediction="", confidences=[], recognition_error="too wide"),
                   SimpleNamespace(id="good", prediction=KA, confidences=[0.9])]
        with Image.new("RGB", (64, 64), "white") as image:
            result = process_page(image, 1, segmenter=lambda image: bounds,
                                  recognizer=lambda image, segmentation: iter(records))
        self.assertEqual(len(result["lines"]), 2)
        self.assertEqual(result["lines"][1]["raw_hypothesis"], KA)
        self.assertIn("line_recognition_failed", result["uncertainty_flags"])
        self.assertEqual(result["errors"], ["Line bad: too wide"])
        self.assertEqual(result["recognition_failure_line_ids"], ["bad"])
        self.assertEqual(result["recognition_missing_line_ids"], [])

    def test_bad_confidences_do_not_discard_the_raw_hypothesis(self):
        for scores in ([None], ["bad"], None):
            result = line_hypothesis(SimpleNamespace(prediction=KA + "A", confidences=scores))
            self.assertEqual(result["raw_hypothesis"], KA + "A")
            self.assertIn("U+0041", result["invalid_codepoints"])

    def test_blank_page_and_approved_no_target_create_no_page_files(self):
        with tempfile.TemporaryDirectory() as directory, Image.new("RGB", (64, 64), "white") as image:
            output = Path(directory) / "export"
            crops = Path(directory) / "private"
            segmenter = Mock(return_value=SimpleNamespace(type="baselines", lines=[], regions={}))
            result = process_page(image, 1, segmenter=segmenter, output_dir=output,
                                  line_crops=crops, export_drafts=True)
            self.assertEqual(result["status"], "blank")
            self.assertFalse(output.exists())
            self.assertFalse(crops.exists())
            image.putpixel((32, 32), (0, 0, 0))
            digest = raster_digest(image.tobytes(), image.width, image.height)
            manifest = {"policy_version": POLICY_VERSION, "pages": {
                digest: {"status": "approved", "reviewed_by": "reviewer", "decision": "no_target_text"}}}
            segmenter.reset_mock()
            result = process_page(image, 1, segmenter=segmenter, manifest=manifest,
                                  output_dir=output, line_crops=crops, export_drafts=True)
            self.assertEqual(result["status"], "no_target_text")
            self.assertEqual(result["provenance"], "reference")
            segmenter.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(crops.exists())

    def test_stain_without_lines_or_failed_segmentation_is_not_blank(self):
        with Image.new("RGB", (64, 64), "white") as image:
            failing = Mock(side_effect=RuntimeError("unavailable model"))
            result = process_page(image, 1, segmenter=failing)
            self.assertEqual(result["status"], "review")
            self.assertIn("segmentation_failed", result["uncertainty_flags"])
            image.putpixel((32, 32), (20, 20, 20))
            empty = Mock(return_value=SimpleNamespace(type="baselines", lines=[], regions={}))
            result = process_page(image, 1, segmenter=empty)
            self.assertEqual(result["status"], "review")
            self.assertIn("no_detected_lines_with_content", result["uncertainty_flags"])

    def test_model_hypotheses_are_preserved_but_no_final_text_is_written(self):
        line = SimpleNamespace(id="line1", baseline=[(4, 20), (50, 20)],
                               boundary=[(4, 4), (50, 4), (50, 24), (4, 24)])
        segmenter = Mock(return_value=SimpleNamespace(type="baselines", lines=[line], regions={}))
        for raw in (KA, KA + " English", ""):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory, Image.new("RGB", (64, 64), "white") as image:
                record = SimpleNamespace(id="line1", prediction=raw, confidences=[0.999] * len(raw))
                recognizer = Mock(return_value=iter([record]))
                result = process_page(image, 1, segmenter=segmenter, recognizer=recognizer,
                                      output_dir=Path(directory) / "export")
                self.assertEqual(result["status"], "review")
                self.assertEqual(result["provenance"], "model")
                self.assertEqual(result["raw_hypothesis"], raw)
                self.assertEqual(result["draft_text"], raw)
                self.assertFalse((Path(directory) / "export").exists())
                if "English" in raw:
                    self.assertIn("U+0045", result["invalid_codepoints"])
                if not raw:
                    self.assertIn("suspicious_empty_recognition", result["uncertainty_flags"])

    def test_explicit_raw_draft_export_is_not_an_approved_transcript(self):
        line = SimpleNamespace(id="line1", baseline=[(4, 20), (50, 20)],
                               boundary=[(4, 4), (50, 4), (50, 24), (4, 24)])
        bounds = SimpleNamespace(type="baselines", lines=[line], regions={})
        for raw in (KA, KA + "\r\n" + KA, KA + " English", "", " "):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory, Image.new("RGB", (64, 64), "white") as image:
                record = SimpleNamespace(id="line1", prediction=raw, confidences=[0.9] * len(raw))
                output = Path(directory) / "export"
                result = process_page(image, 18, segmenter=lambda image: bounds,
                                      recognizer=lambda image, segmentation: iter([record]),
                                      output_dir=output, export_drafts=True)
                self.assertEqual(result["status"], "review")
                self.assertIsNone(result["transcript_path"])
                self.assertEqual(result["raw_hypothesis"], raw)
                if raw.strip() and "English" not in raw:
                    path = output / "drafts" / "page-0018.txt"
                    self.assertEqual(result["draft_path"], str(path))
                    self.assertEqual(path.read_bytes(), raw.encode("utf-8"))
                else:
                    self.assertIsNone(result["draft_path"])
                    self.assertFalse(output.exists())

    def test_segmentation_only_does_not_call_recognizer(self):
        line = SimpleNamespace(id="line1", baseline=[(4, 20), (50, 20)],
                               boundary=[(4, 4), (50, 4), (50, 24), (4, 24)])
        segmenter = Mock(return_value=SimpleNamespace(type="baselines", lines=[line], regions={}))
        recognizer = Mock(side_effect=AssertionError("must not recognize"))
        with Image.new("RGB", (64, 64), "white") as image:
            for options in ({"segment_only": True, "recognizer": recognizer}, {}):
                result = process_page(image, 1, segmenter=segmenter, **options)
                self.assertEqual(result["status"], "segmented")
                self.assertEqual(result["segmentation"]["line_count"], 1)
                self.assertIsNone(result["raw_hypothesis"])
            recognizer.assert_not_called()

    def test_reference_hash_uses_full_page_before_caption_cleanup(self):
        with tempfile.TemporaryDirectory() as directory, Image.new("RGB", (64, 64), "white") as image:
            image.putpixel((1, 1), (0, 0, 0))
            digest = raster_digest(image.tobytes(), image.width, image.height)
            manifest = {"policy_version": POLICY_VERSION, "pages": {
                digest: {"status": "approved", "reviewed_by": "reviewer",
                         "decision": "transcription", "text": KA}}}
            metadata = {"captions": {"regions": [{"text": "Source", "pixel_bbox": [0, 0, 8, 8]}]}}
            segmenter = Mock(side_effect=AssertionError("reference must bypass segmentation"))
            result = process_page(image, 1, segmenter=segmenter, manifest=manifest,
                                  metadata=metadata, output_dir=directory)
            self.assertEqual(result["raster_digest"], digest)
            self.assertEqual(result["provenance"], "reference")
            self.assertEqual(Path(result["transcript_path"]).read_bytes(), KA.encode("utf-8"))
            self.assertEqual(image.getpixel((1, 1)), (0, 0, 0))
            segmenter.assert_not_called()

    def test_cleanup_preserves_original_and_cannot_confirm_masked_page_blank(self):
        with Image.new("RGB", (64, 64), "white") as image:
            image.putpixel((1, 1), (0, 0, 0))
            digest = raster_digest(image.tobytes(), image.width, image.height)
            metadata = {"captions": {"regions": [{"text": "Source", "pixel_bbox": [0, 0, 8, 8]}]}}

            def segmenter(working):
                self.assertEqual(working.getpixel((1, 1)), (255, 255, 255))
                return SimpleNamespace(type="baselines", lines=[], regions={})

            result = process_page(image, 1, segmenter=segmenter, metadata=metadata)
            self.assertEqual(result["raster_digest"], digest)
            self.assertEqual(result["status"], "review")
            self.assertEqual(image.getpixel((1, 1)), (0, 0, 0))
            self.assertIsNone(result["preprocessing"]["crop"])

    def test_nonempty_output_is_refused_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "existing.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_output(directory)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_reference_only_run_writes_aggregate_and_disable_references_works(self):
        with tempfile.TemporaryDirectory() as directory, Image.new("RGB", (16, 16), "white") as image:
            root = Path(directory)
            source = root / "input.png"
            image.save(source)
            digest = raster_digest(image.tobytes(), image.width, image.height)
            manifest = {"policy_version": POLICY_VERSION, "pages": {
                digest: {"status": "approved", "reviewed_by": "reviewer", "decision": "no_target_text"}}}
            references = root / "references.json"
            references.write_text(json.dumps(manifest), encoding="utf-8")
            result = run_pipeline(source, root / "run1", references=references, line_crops=root / "private")
            self.assertEqual(result["summary"]["reference_pages"], 1)
            self.assertEqual(list((root / "run1").iterdir()), [root / "run1" / "report.json"])
            self.assertFalse((root / "private").exists())
            stored = json.loads((root / "run1" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["pages"][0]["status"], "no_target_text")
            with patch("khojki_ocr.pipeline.process_page") as processing:
                processing.return_value = result["pages"][0]
                run_pipeline(source, root / "run2", references=root / "does-not-exist.json", disable_references=True)
                self.assertIsNone(processing.call_args.kwargs["manifest"])
                run_pipeline(source, root / "run3", references=references, segment_only=True)
                self.assertIsNone(processing.call_args.kwargs["manifest"])

    def test_image_selection_and_native_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pages.tiff"
            with Image.new("RGB", (12, 8), "white") as first, Image.new("RGB", (12, 8), "black") as second:
                first.save(source, save_all=True, append_images=[second])
            pages = iter_pages(source, [2], dpi=600)
            number, image, metadata = next(pages)
            self.assertEqual(number, 2)
            self.assertEqual(image.size, (12, 8))
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
            self.assertEqual(metadata["resolution"], "native_pixels")
            pages.close()
            with self.assertRaises(ValueError):
                next(iter_pages(source, [3]))

    def test_transparent_image_is_composited_on_white(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "transparent.png"
            with Image.new("RGBA", (8, 8), (0, 0, 0, 0)) as transparent:
                transparent.save(source)
            for _, image, _ in iter_pages(source):
                self.assertTrue(raster_statistics(image)["high_whiteness"])

    def test_pdf_ingestion_uses_render_scale_and_releases_handles(self):
        page = Mock()
        page.get_rotation.return_value = 0
        bitmap = page.render.return_value
        bitmap.to_pil.return_value = Image.new("RGB", (10, 20), "white")
        document = Mock()
        document.__len__ = Mock(return_value=3)
        document.__getitem__ = Mock(return_value=page)
        context = Mock()
        context.__enter__ = Mock(return_value=document)
        context.__exit__ = Mock(return_value=False)
        pdfium = SimpleNamespace(PdfDocument=Mock(return_value=context))
        with patch.dict("sys.modules", {"pypdfium2": pdfium}):
            pages = iter_pages("input.pdf", [2], dpi=144)
            number, image, metadata = next(pages)
            self.assertEqual(number, 2)
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
            page.render.assert_called_once_with(scale=2, rotation=0, crop=(0, 0, 0, 0))
            document.__getitem__.assert_called_once_with(1)
            bitmap.close.assert_called_once()
            page.close.assert_called_once()
            self.assertEqual(metadata["captions"]["regions"], [])
            pages.close()

    def test_caption_mask_requires_exact_text_and_no_scan_overlap(self):
        page = Mock()
        textpage = page.get_textpage.return_value
        textpage.count_rects.return_value = 1
        textpage.get_rect.return_value = (10, 10, 80, 20)
        textpage.get_text_bounded.return_value = "Source: Archive"
        bitmap = Mock(width=100, height=100)
        bitmap.get_posconv.return_value.to_bitmap.side_effect = lambda horizontal, vertical: (horizontal, 100 - vertical)
        image_object = Mock(type=3)
        image_object.get_bounds.return_value = (0, 0, 100, 100)
        pdfium = SimpleNamespace(raw=SimpleNamespace(FPDF_PAGEOBJ_TEXT=1))
        with patch.dict("sys.modules", {"pypdfium2": pdfium}):
            page.get_objects.return_value = [image_object]
            result = pdf_caption_regions(page, bitmap, ["Source: Archive"])
            self.assertEqual(result["regions"], [])
            self.assertEqual(result["rejected"][0]["reason"], "overlaps_nontext_object")
            page.get_objects.return_value = []
            result = pdf_caption_regions(page, bitmap, ["Source: Archive"])
            self.assertEqual(result["regions"][0]["pixel_bbox"], [9, 79, 82, 92])
            self.assertEqual(result["regions"][0]["raster_guard_pixels"], 1)
            with Image.new("RGB", (100, 100), "white") as image:
                edge_pixels = [(9, 80), (10, 79), (81, 90), (80, 91)]
                for point in edge_pixels:
                    image.putpixel(point, (40, 40, 40))

                def segmenter(working):
                    self.assertTrue(raster_statistics(working)["high_whiteness"])
                    return SimpleNamespace(type="baselines", lines=[], regions={})

                processed = process_page(image, 1, segmenter=segmenter,
                                         metadata={"captions": result})
                self.assertEqual(processed["status"], "review")
                self.assertTrue(all(image.getpixel(point) == (40, 40, 40) for point in edge_pixels))
            image_object.get_bounds.return_value = (0, 0, 9, 100)
            page.get_objects.return_value = [image_object]
            guarded = pdf_caption_regions(page, bitmap, ["Source: Archive"])
            self.assertEqual(guarded["regions"], [])
            self.assertEqual(guarded["rejected"][0]["reason"], "overlaps_nontext_object")
            page.get_objects.return_value = []
            textpage.get_rect.return_value = (0, 10, 80, 20)
            guarded = pdf_caption_regions(page, bitmap, ["Source: Archive"])
            self.assertEqual(guarded["regions"], [])
            self.assertEqual(guarded["rejected"][0]["reason"], "out_of_bounds")
            self.assertEqual(pdf_caption_regions(page, bitmap, ["Other caption"])["regions"], [])
            with self.assertRaises(ValueError):
                pdf_caption_regions(page, bitmap, [KA])

    def test_page_selection_is_one_based_and_preserves_order(self):
        self.assertIsNone(parse_pages(None))
        self.assertEqual(parse_pages("2, 1"), [2, 1])
        for value in ("", "0", "1,1", "1,", "-1", "1-3", "a", "1.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_pages(value)

    def test_white_rgb_requires_no_detected_lines(self):
        with Image.new("RGB", (64, 64), "white") as image:
            statistics = raster_statistics(image)
        self.assertTrue(confirmed_blank(statistics, 0))
        self.assertFalse(confirmed_blank(statistics, 1))

    def test_one_stain_pixel_or_faint_content_requires_review(self):
        for color in ((0, 0, 0), (249, 249, 249), (255, 245, 255)):
            with self.subTest(color=color), Image.new("RGB", (64, 64), "white") as image:
                image.putpixel((32, 32), color)
                self.assertFalse(confirmed_blank(raster_statistics(image), 0))

    def test_invalid_hypothesis_is_not_stripped(self):
        raw = KA + " English\t\U00011212"
        result = line_hypothesis(SimpleNamespace(prediction=raw, confidences=[0.99] * len(raw)))
        self.assertEqual(result["raw_hypothesis"], raw)
        self.assertIn("U+11212", result["invalid_codepoints"])
        self.assertIn("U+0009", result["invalid_codepoints"])
        self.assertIn("invalid_codepoints", result["uncertainty_flags"])

    def test_high_confidence_target_text_does_not_prove_image_script(self):
        result = line_hypothesis(SimpleNamespace(prediction=KA, confidences=[0.999]))
        self.assertEqual(result["invalid_codepoints"], [])
        self.assertIn("script_identity_unverified", result["uncertainty_flags"])

    def test_warning_threshold_is_opt_in_and_nonfinite_scores_are_flagged(self):
        record = SimpleNamespace(prediction=KA, confidences=[0.1])
        self.assertNotIn("below_warning_confidence", line_hypothesis(record)["uncertainty_flags"])
        self.assertIn("below_warning_confidence", line_hypothesis(record, 0.5)["uncertainty_flags"])
        result = line_hypothesis(SimpleNamespace(prediction=KA, confidences=[float("nan")]))
        self.assertEqual(result["character_confidences"], [None])
        self.assertIn("invalid_confidences", result["uncertainty_flags"])


if __name__ == "__main__":
    unittest.main()