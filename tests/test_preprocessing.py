import json
import math
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

from khojki_ocr.preprocessing import (
    assess_text_presence,
    crop_line_region,
    is_likely_text_region,
    prepare_page,
)


def ruled_text_page():
    image = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(image)
    for top in (90, 180, 270, 360):
        for left in (100, 300, 500):
            draw.line((left, top, left + 120, top), fill="black", width=2)
            for offset in (8, 32, 56, 80, 104):
                draw.rectangle((left + offset, top, left + offset + 8, top + 18), fill="black")
                draw.ellipse((left + offset, top - 11, left + offset + 3, top - 8), fill="black")
    return image


class PreparePageTests(unittest.TestCase):
    def test_raw_default_is_independent_rgb_and_ignores_opt_in_flags(self):
        source = ruled_text_page()
        original = source.tobytes()
        output, metadata = prepare_page(source, deskew=True, crop_border=True)
        self.assertEqual(output.mode, "RGB")
        self.assertIsNot(output, source)
        self.assertEqual(output.tobytes(), original)
        self.assertFalse(metadata["changed"])
        self.assertEqual(metadata["to_original"], np.eye(3).tolist())
        self.assertEqual(metadata["from_original"], np.eye(3).tolist())
        json.dumps(metadata, allow_nan=False)
        output.putpixel((0, 0), (0, 0, 0))
        self.assertEqual(source.tobytes(), original)

    def test_unsigned_float_and_alpha_inputs(self):
        for pixels in (np.array([[0, 65535]], dtype=np.uint16),
                       np.array([[0, 255]], dtype=np.uint8),
                       np.array([[0.0, 1.0]], dtype=np.float32),
                       np.array([[0.0, 255.0]], dtype=np.float64)):
            with self.subTest(dtype=pixels.dtype):
                original = pixels.copy()
                output, _ = prepare_page(pixels)
                self.assertEqual(list(output.getdata()), [(0, 0, 0), (255, 255, 255)])
                np.testing.assert_array_equal(pixels, original)
        transparent = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        self.assertEqual(prepare_page(transparent)[0].getpixel((0, 0)), (255, 255, 255))
        single_channel = np.full((3, 5, 1), 255, dtype=np.uint8)
        self.assertEqual(prepare_page(single_channel)[0].size, (5, 3))

    def test_invalid_inputs_fail_explicitly(self):
        for pixels in (np.zeros((0, 4), dtype=np.uint8), np.zeros((3, 4, 2), dtype=np.uint8),
                       np.array([[-1]], dtype=np.int16), np.array([[float("nan")]]),
                       np.array([[float("inf")]]), np.array([[256.0]])):
            with self.subTest(shape=pixels.shape, dtype=pixels.dtype), self.assertRaises(ValueError):
                prepare_page(pixels)
        with self.assertRaises(ValueError):
            prepare_page(Image.new("RGB", (10, 10)), mode="automatic")
        with self.assertRaises(ValueError):
            prepare_page(Image.new("RGB", (10, 10)), deskew=cast(Any, 1))

    def test_white_margin_and_brown_full_scan_do_not_crop(self):
        for color in ("white", (190, 150, 105)):
            with self.subTest(color=color):
                source = Image.new("RGB", (200, 160), color)
                ImageDraw.Draw(source).rectangle((8, 40, 20, 55), fill="black")
                output, metadata = prepare_page(source, mode="conservative", crop_border=True)
                self.assertEqual(output.tobytes(), source.tobytes())
                self.assertFalse(metadata["crop"]["applied"])

    def test_substantial_black_frame_crop_and_inverse_mapping(self):
        source = Image.new("RGB", (200, 160), "black")
        ImageDraw.Draw(source).rectangle((10, 8, 189, 151), fill="white")
        original = source.tobytes()
        output, metadata = prepare_page(source, mode="conservative", crop_border=True)
        self.assertTrue(metadata["crop"]["applied"])
        self.assertEqual(metadata["crop"]["bbox"], [10, 8, 190, 152])
        self.assertEqual(output.size, (180, 144))
        np.testing.assert_allclose(np.array(metadata["to_original"]) @ [0, 0, 1], [10, 8, 1])
        np.testing.assert_allclose(np.array(metadata["to_original"]) @ np.array(metadata["from_original"]), np.eye(3))
        self.assertEqual(source.tobytes(), original)

    def test_dot_next_to_candidate_band_rejects_entire_crop(self):
        source = Image.new("RGB", (200, 160), "white")
        draw = ImageDraw.Draw(source)
        draw.rectangle((0, 0, 9, 159), fill="black")
        draw.point((12, 70), fill="black")
        output, metadata = prepare_page(source, mode="conservative", crop_border=True)
        self.assertFalse(metadata["crop"]["applied"])
        self.assertIn("left", metadata["crop"]["protected_sides"])
        self.assertEqual(output.tobytes(), source.tobytes())

    def test_crop_rejects_15_percent_and_larger_bands(self):
        for depth in (30, 45, 200):
            with self.subTest(depth=depth):
                source = Image.new("RGB", (200, 160), "white")
                ImageDraw.Draw(source).rectangle((0, 0, depth - 1, 159), fill="black")
                output, metadata = prepare_page(source, mode="conservative", crop_border=True)
                self.assertEqual(output.tobytes(), source.tobytes())
                self.assertFalse(metadata["crop"]["applied"])

    def test_normalization_preserves_every_dark_dot_and_diacritic(self):
        pixels = np.empty((120, 200, 3), dtype=np.uint8)
        for column in range(200):
            pixels[:, column] = (170 + column // 4, 160 + column // 4, 140 + column // 4)
        source = Image.fromarray(pixels)
        draw = ImageDraw.Draw(source)
        draw.rectangle((50, 50, 65, 65), fill=(35, 30, 25))
        draw.point((51, 43), fill="black")
        draw.ellipse((69, 42, 72, 45), fill=(45, 40, 35))
        original = np.asarray(source).copy()
        output, metadata = prepare_page(source, mode="normalized")
        dark = original.max(axis=2) <= 110
        np.testing.assert_array_equal(np.asarray(output)[dark], original[dark])
        np.testing.assert_array_equal(np.asarray(source), original)
        self.assertTrue(metadata["background"]["applied"])
        self.assertEqual(metadata["background"]["components_removed"], 0)
        self.assertGreater(output.getpixel((10, 10))[2], source.getpixel((10, 10))[2])
        json.dumps(metadata, allow_nan=False)

    def test_weak_ornament_and_single_rule_do_not_deskew(self):
        for ornament in (False, True):
            source = Image.new("RGB", (400, 300), "white")
            draw = ImageDraw.Draw(source)
            if ornament:
                draw.ellipse((110, 80, 290, 220), outline="black", width=3)
            else:
                draw.line((30, 140, 370, 151), fill="black", width=3)
            output, metadata = prepare_page(source, mode="conservative", deskew=True)
            self.assertFalse(metadata["deskew"]["confident"])
            self.assertFalse(metadata["deskew"]["applied"])
            self.assertEqual(output.tobytes(), source.tobytes())

    def test_weak_two_degree_scan_is_conservatively_rejected(self):
        source = ruled_text_page().rotate(2, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
        original = source.tobytes()
        output, metadata = prepare_page(source, mode="conservative", deskew=True)
        evidence = metadata["deskew"]
        self.assertFalse(evidence["confident"], evidence)
        self.assertFalse(evidence["applied"])
        self.assertEqual(evidence["reason"], "inconsistent_stroke_angles")
        self.assertGreater(evidence["confidence"], 0)
        self.assertLess(evidence["confidence"], 0.75)
        self.assertAlmostEqual(evidence["estimated_degrees"], -2.0, delta=0.4)
        self.assertEqual(output.tobytes(), original)
        self.assertEqual(source.tobytes(), original)
        np.testing.assert_array_equal(metadata["from_original"], np.eye(3))
        np.testing.assert_array_equal(metadata["to_original"], np.eye(3))

    def test_computed_deskew_row_support_is_json_serializable(self):
        source = ruled_text_page().rotate(2, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
        _, metadata = prepare_page(source, mode="conservative", deskew=True)
        evidence = metadata["deskew"]
        self.assertGreater(evidence["supported_rows"], 0, evidence)
        self.assertIs(type(evidence["supported_rows"]), int)
        serialized = json.loads(json.dumps(metadata, allow_nan=False))
        self.assertEqual(serialized["deskew"]["supported_rows"], evidence["supported_rows"])

    def test_known_two_degree_scan_estimate(self):
        page = Image.new("RGB", (1200, 700), "white")
        draw = ImageDraw.Draw(page)
        for top in (100, 240, 380, 520):
            for left in (100, 650):
                draw.line((left, top, left + 420, top), fill="black", width=2)
                for offset in range(8, 408, 24):
                    draw.rectangle((left + offset, top, left + offset + 8, top + 18), fill="black")
                    draw.ellipse((left + offset, top - 11, left + offset + 3, top - 8), fill="black")
        source = page.rotate(2, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
        original = source.tobytes()
        output, metadata = prepare_page(source, mode="conservative", deskew=True)
        self.assertTrue(metadata["deskew"]["confident"], metadata["deskew"])
        self.assertTrue(metadata["deskew"]["applied"])
        self.assertGreaterEqual(metadata["deskew"]["confidence"], 0.75)
        self.assertGreaterEqual(metadata["deskew"]["supported_rows"], 3)
        self.assertAlmostEqual(metadata["deskew"]["estimated_degrees"], -2.0, delta=0.4)
        self.assertGreater(output.width, source.width)
        self.assertGreater(output.height, source.height)
        self.assertEqual(source.tobytes(), original)
        json.dumps(metadata, allow_nan=False)

    def test_expanded_rotation_exact_position_corners_and_inverse(self):
        source = ruled_text_page()
        evidence = {"confident": True, "applied": False, "reason": "supported_text_strokes",
                    "estimated_degrees": 2.0, "confidence": 1.0}
        original = source.tobytes()
        with patch("khojki_ocr.preprocessing._deskew_evidence", return_value=evidence):
            output, metadata = prepare_page(source, mode="conservative", deskew=True)
        forward, inverse = np.array(metadata["from_original"]), np.array(metadata["to_original"])
        np.testing.assert_allclose(forward @ inverse, np.eye(3), atol=1e-12)
        corners = np.array([[-0.5, -0.5, 1], [799.5, -0.5, 1], [799.5, 499.5, 1], [-0.5, 499.5, 1]])
        mapped = corners @ forward.T
        self.assertGreaterEqual(mapped[:, 0].min(), -0.5 - 1e-10)
        self.assertGreaterEqual(mapped[:, 1].min(), -0.5 - 1e-10)
        self.assertLessEqual(mapped[:, 0].max(), output.width - 0.5 + 1e-10)
        self.assertLessEqual(mapped[:, 1].max(), output.height - 0.5 + 1e-10)
        np.testing.assert_allclose(mapped @ inverse.T, corners, atol=1e-10)
        angle = math.radians(2)
        np.testing.assert_allclose(forward[:2, :2], [[math.cos(angle), math.sin(angle)],
                                                   [-math.sin(angle), math.cos(angle)]])
        point = forward @ [108, 100, 1]
        horizontal, vertical = round(point[0]), round(point[1])
        self.assertLess(np.asarray(output)[vertical - 1:vertical + 2,
                          horizontal - 1:horizontal + 2].min(), 90)
        self.assertEqual(output.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(source.tobytes(), original)

    def test_crop_then_rotate_composes_in_the_documented_order(self):
        source = Image.new("RGB", (220, 180), "black")
        ImageDraw.Draw(source).rectangle((10, 10, 209, 169), fill="white")
        evidence = {"confident": True, "applied": False, "reason": "supported_text_strokes",
                    "estimated_degrees": -2.0, "confidence": 1.0}
        with patch("khojki_ocr.preprocessing._deskew_evidence", return_value=evidence):
            _, metadata = prepare_page(source, mode="conservative", deskew=True, crop_border=True)
        forward = np.array(metadata["from_original"])
        cropped_rotation = cv2.getRotationMatrix2D((99.5, 79.5), -2, 1)
        np.testing.assert_allclose(forward[:2, :2], cropped_rotation[:, :2])
        np.testing.assert_allclose(np.array(metadata["to_original"]) @ forward @ [10, 10, 1], [10, 10, 1])
        self.assertEqual(metadata["crop"]["bbox"], [10, 10, 210, 170])


class RegionEvidenceTests(unittest.TestCase):
    def test_polygon_baseline_bbox_and_clipped_geometry(self):
        source = Image.new("RGB", (100, 80), "white")
        polygon = {"boundary": [[10, 20], [40, 20], [40, 30], [10, 30]]}
        crop, metadata = crop_line_region(source, polygon)
        self.assertEqual(crop.size, (30, 10))
        self.assertEqual(metadata["bbox"], [10, 20, 40, 30])
        crop, metadata = crop_line_region(source, SimpleNamespace(baseline=[(10, 25), (40, 25)], boundary=[]))
        self.assertGreater(crop.height, 0)
        self.assertTrue(metadata["baseline_only"])
        crop, metadata = crop_line_region(source, {"bbox": [-4, 10, 30, 25]})
        self.assertEqual(crop.size, (30, 15))
        self.assertTrue(metadata["geometry_clipped"])
        np.testing.assert_allclose(np.array(metadata["to_page"]) @ [0, 0, 1], [0, 10, 1])
        for invalid in ([1, 1, 1, 4], [0, 0, float("nan"), 5], [110, 0, 120, 5]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                crop_line_region(source, invalid)

    def test_repeated_generic_glyphs_score_without_any_script_claim(self):
        source = Image.new("RGB", (400, 100), "white")
        draw = ImageDraw.Draw(source)
        for left in (40, 65, 90, 115, 140, 165):
            draw.rectangle((left, 40, left + 8, 58), fill="black")
            draw.point((left + 3, 32), fill="black")
        original = source.tobytes()
        result = is_likely_text_region(source, [25, 20, 200, 70])
        self.assertEqual(result["decision"], "likely_text")
        self.assertIsNone(result["script_label"])
        self.assertFalse(result["strict_artifact"])
        self.assertFalse(result["automatic_removal"])
        self.assertEqual(result["reading_direction"], "ltr")
        self.assertEqual(source.tobytes(), original)
        json.dumps(result, allow_nan=False)

    def test_only_page_spanning_single_solid_rule_is_strict_frame(self):
        source = Image.new("RGB", (400, 120), "white")
        draw = ImageDraw.Draw(source)
        draw.line((10, 60, 390, 60), fill="black", width=2)
        result = is_likely_text_region(source, [0, 45, 400, 75])
        self.assertTrue(result["strict_artifact"])
        self.assertEqual(result["decision"], "artifact_candidate")
        self.assertFalse(is_likely_text_region(source, [-2, 45, 400, 75])["strict_artifact"])
        self.assertFalse(is_likely_text_region(source, {"baseline": [[10, 60], [390, 60]]})["strict_artifact"])
        draw.point((100, 50), fill="black")
        self.assertFalse(is_likely_text_region(source, [0, 45, 400, 75])["strict_artifact"])
        shorter = Image.new("RGB", (400, 120), "white")
        ImageDraw.Draw(shorter).line((100, 60, 220, 60), fill="black", width=2)
        self.assertFalse(is_likely_text_region(shorter, [90, 45, 230, 75])["strict_artifact"])

    def test_dot_groups_and_isolated_specks_are_not_hard_rejected(self):
        for count in (1, 4):
            with self.subTest(count=count):
                source = Image.new("RGB", (200, 80), "white")
                draw = ImageDraw.Draw(source)
                for number in range(count):
                    draw.ellipse((30 + number * 20, 30, 32 + number * 20, 32), fill="black")
                result = is_likely_text_region(source, [20, 20, 120, 45])
                self.assertFalse(result["strict_artifact"])
                self.assertEqual(result["decision"], "ambiguous")
                if count == 4:
                    self.assertTrue(result["dot_group_possible"])


class TextPresenceTests(unittest.TestCase):
    def test_empty_requires_every_channel_nearwhite_and_explicit_no_proposals(self):
        source = Image.new("RGB", (30, 30), "white")
        self.assertFalse(assess_text_presence(source)["confirmed_empty"])
        self.assertFalse(assess_text_presence(source, proposals=[{}])["confirmed_empty"])
        result = assess_text_presence(source, proposals=[])
        self.assertTrue(result["confirmed_empty"])
        self.assertEqual(result["status"], "confirmed_empty")
        json.dumps(result, allow_nan=False)

    def test_brown_paper_faint_dot_and_heavy_blemishes_remain_review(self):
        paper = Image.new("RGB", (80, 60), (190, 150, 110))
        faint = Image.new("RGB", (80, 60), "white")
        faint.putpixel((40, 30), (252, 252, 252))
        blemished = Image.new("RGB", (80, 60), "white")
        ImageDraw.Draw(blemished).rectangle((2, 2, 60, 55), fill=(80, 70, 60))
        for source in (paper, faint, blemished):
            result = assess_text_presence(source, proposals=[])
            self.assertFalse(result["confirmed_empty"])
            self.assertEqual(result["status"], "review")


if __name__ == "__main__":
    unittest.main()