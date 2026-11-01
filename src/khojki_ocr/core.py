"""Output policy and exact, explicitly approved reference retrieval."""

import hashlib
import struct
import unicodedata
from pathlib import Path

POLICY_VERSION = "khojki-only-nfc-v1"
KHOJKI_CODEPOINTS = frozenset(range(0x11200, 0x11242)) - {0x11212}


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def invalid_codepoints(text: str) -> list[str]:
    return sorted({f"U+{ord(character):04X}" for character in text
                   if ord(character) not in KHOJKI_CODEPOINTS and character not in " \n"})


def validate_text(text: str) -> str:
    normalized = normalize_text(text)
    invalid = invalid_codepoints(normalized)
    if invalid:
        raise ValueError("Non-target or unassigned code points: " + ", ".join(invalid))
    return normalized


def raster_digest(pixels: bytes, width: int, height: int, mode: str = "RGB") -> str:
    if width <= 0 or height <= 0 or mode != "RGB" or len(pixels) != width * height * 3:
        raise ValueError("Expected a nonempty, tightly packed RGB raster")
    header = b"khojki-raster-v1\0RGB\0" + struct.pack(">QQ", width, height)
    return hashlib.sha256(header + pixels).hexdigest()


def approved_reference(manifest: dict, digest: str) -> dict | None:
    if manifest.get("policy_version") != POLICY_VERSION:
        return None
    entry = manifest.get("pages", {}).get(digest)
    if not entry or entry.get("status") != "approved" or not entry.get("reviewed_by"):
        return None
    decision = entry.get("decision")
    if decision == "no_target_text":
        if entry.get("text", "").strip():
            raise ValueError("A no-target reference cannot contain a transcription")
        return {**entry, "text": ""}
    if decision != "transcription":
        raise ValueError("Unknown approved reference decision")
    text = validate_text(entry.get("text", ""))
    if not text.strip():
        raise ValueError("An approved transcription must contain target text")
    return {**entry, "text": text}


def write_transcript(path: Path, text: str) -> bool:
    normalized = validate_text(text)
    if not normalized.strip():
        if path.exists():
            raise FileExistsError("Refusing to leave a stale transcript for an empty page")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(normalized)
    return True