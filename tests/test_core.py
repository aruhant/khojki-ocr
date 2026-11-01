import tempfile
import unittest
from pathlib import Path

from khojki_ocr.core import (
    POLICY_VERSION, approved_reference, invalid_codepoints, raster_digest,
    validate_text, write_transcript,
)

KA = "\U00011208"
VOWEL_I = "\U0001122d"


class OutputPolicyTests(unittest.TestCase):
    def test_base_and_dependent_vowel_are_preserved(self):
        self.assertEqual(validate_text(KA + VOWEL_I), KA + VOWEL_I)

    def test_english_reserved_digit_and_replacement_character_are_rejected(self):
        for character in ("A", "1", "\U00011212", "\ufffd", "\t"):
            with self.subTest(character=character), self.assertRaises(ValueError):
                validate_text(KA + character)

    def test_wrong_script_is_reported_not_silently_stripped(self):
        self.assertEqual(invalid_codepoints(KA + "English"),
                         ["U+0045", "U+0067", "U+0068", "U+0069", "U+006C", "U+006E", "U+0073"])

    def test_empty_page_creates_no_file_or_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs" / "page.txt"
            self.assertFalse(write_transcript(output, " \n"))
            self.assertFalse(output.parent.exists())

    def test_output_is_utf8_without_english_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "page.txt"
            self.assertTrue(write_transcript(output, KA + VOWEL_I))
            self.assertEqual(output.read_bytes(), (KA + VOWEL_I).encode("utf-8"))
            with self.assertRaises(FileExistsError):
                write_transcript(output, "")

    def test_digest_changes_with_pixels_or_dimensions(self):
        self.assertNotEqual(raster_digest(bytes(6), 2, 1), raster_digest(bytes(6), 1, 2))
        self.assertNotEqual(raster_digest(bytes(3), 1, 1), raster_digest(b"\x01\x00\x00", 1, 1))

    def test_draft_and_wrong_policy_references_are_not_replayed(self):
        manifest = {"policy_version": POLICY_VERSION, "pages": {
            "digest": {"status": "draft", "reviewed_by": "reviewer",
                       "decision": "transcription", "text": KA}}}
        self.assertIsNone(approved_reference(manifest, "digest"))
        manifest["pages"]["digest"]["status"] = "approved"
        self.assertEqual(approved_reference(manifest, "digest")["text"], KA)
        manifest["policy_version"] = "different-policy"
        self.assertIsNone(approved_reference(manifest, "digest"))

    def test_approved_skip_cannot_hide_text(self):
        manifest = {"policy_version": POLICY_VERSION, "pages": {
            "digest": {"status": "approved", "reviewed_by": "reviewer",
                       "decision": "no_target_text", "text": KA}}}
        with self.assertRaises(ValueError):
            approved_reference(manifest, "digest")


if __name__ == "__main__":
    unittest.main()