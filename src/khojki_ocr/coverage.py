"""Generate a checked basic Khojki combination inventory, not linguistic text."""

import argparse
from pathlib import Path

from khojki_ocr.core import POLICY_VERSION
from khojki_ocr.training import (
    CONSONANTS, INDEPENDENT_VOWELS, VOWEL_SIGNS, file_sha256, load_font,
    render_line, write_json,
)


def build_coverage(font_path: Path, output: Path, *, development: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(output)
    font, supported, renderer = load_font(font_path, 48)
    consonants = sorted(CONSONANTS & supported)
    if development:
        consonants = [base for base in consonants if base in (0x11208, 0x1121F, 0x11229)]
    if not consonants:
        raise ValueError("Font lacks the consonants required for coverage")
    vowels = sorted(VOWEL_SIGNS & supported)
    independent = sorted(INDEPENDENT_VOWELS & supported)
    missing = (CONSONANTS | VOWEL_SIGNS | INDEPENDENT_VOWELS) - supported
    combinations = [(chr(base), base, None) for base in consonants]
    combinations += [(chr(base) + chr(vowel), base, vowel) for base in consonants for vowel in vowels]
    combinations += [(chr(vowel), vowel, None) for vowel in independent]
    output.mkdir(parents=True)
    records = []
    for index, (target, base, vowel) in enumerate(combinations):
        next_base = chr(consonants[(index + 7) % len(consonants)])
        text = (next_base + "\U0001123a" + target + "\U0001123a" if development
            else target + "\U0001123a" + next_base + "\U0001122c" + target + "\U0001123a")
        image = render_line(text, font)
        path = output / f"coverage-{index:04d}.png"
        image.save(path)
        path.with_suffix(".gt.txt").write_text(text, encoding="utf-8")
        records.append({"id": f"{'development' if development else 'coverage'}-{index:04d}",
                "page": "synthetic-development" if development else "synthetic-coverage",
                        "status": "reviewed", "example_kind": "synthetic",
                        "source_preview": str(path.resolve()), "bbox": [0, 0, *image.size],
                        "text": text, "covered_base": f"U+{base:04X}",
                        "covered_vowel": f"U+{vowel:04X}" if vowel else None,
                        "image_sha256": file_sha256(path)})
        image.close()
    report = {"policy_version": POLICY_VERSION, "dataset_kind": "synthetic",
              "label_provenance": "Programmatically known Unicode strings rendered with RAQM; basic orthography only",
              "renderer": renderer, "font_sha256": file_sha256(font_path),
              "missing_font_codepoints": [f"U+{value:04X}" for value in sorted(missing)],
              "coverage_scope": ("Synthetic-only development on KA, PA, SA and independent vowels; not real OCR evidence"
                                 if development else "Every font-supported consonant with each basic dependent vowel, plus bare consonants and independent vowels; not conjuncts or handwriting"),
              "consonants": len(consonants), "dependent_vowels": len(vowels), "lines": records}
    write_json(output / "coverage.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    result = build_coverage(args.font, args.output, development=args.development)
    print({"examples": len(result["lines"]), "consonants": result["consonants"],
           "dependent_vowels": result["dependent_vowels"], "missing": result["missing_font_codepoints"]})


if __name__ == "__main__":
    main()