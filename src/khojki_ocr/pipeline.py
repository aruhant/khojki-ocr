"""Conservative page inference; model hypotheses are drafts, not script proof."""

import argparse
import hashlib
import json
import math
import sys
import uuid
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from khojki_ocr.core import (
    POLICY_VERSION, approved_reference, invalid_codepoints, normalize_text,
    raster_digest, validate_text, write_transcript,
)


def _selected_pages(pages: Sequence[int] | None, count: int) -> Sequence[int]:
    if pages is None:
        return range(1, count + 1)
    if not pages or any(type(number) is not int or not 1 <= number <= count for number in pages):
        raise ValueError(f"Selected pages must be between 1 and {count}")
    if len(set(pages)) != len(pages):
        raise ValueError("Duplicate page selection")
    return pages


def _caption_text(text: str) -> str:
    return " ".join(text.split())


def _validate_caption_texts(texts: Sequence[str]) -> None:
    for text in texts:
        if (not text or any(not (32 <= ord(character) <= 126 or character in "\r\n")
                            for character in text)
                or not any(character.isascii() and character.isalpha() for character in text)):
            raise ValueError("Caption text must be explicit printable ASCII text containing letters")


def pdf_caption_regions(page, bitmap, allowed_texts: Sequence[str] = ()) -> dict:
    """Select exact, user-named ASCII captions outside all non-text objects.

    PDF text alone is not evidence of English or of a visible caption: hidden
    OCR layers and any overlap with a scan, drawing, or form are rejected.
    Coordinates use PDFium's own page-to-bitmap transform, including rotation.
    A one-pixel raster guard covers rounding/antialiasing before safety checks.
    Nothing is erased here. Any inspection failure leaves all pixels intact.
    """
    _validate_caption_texts(allowed_texts)
    result = {"mode": "explicit_ascii_captions" if allowed_texts else "disabled",
              "allowed_texts": list(allowed_texts), "regions": [], "rejected": [],
              "uncertainty_flags": []}
    if not allowed_texts:
        return result
    from pypdfium2 import raw as pdfium_raw

    try:
        converter = bitmap.get_posconv(page)
        obstacles = [obj.get_bounds() for obj in page.get_objects(max_depth=1)
                     if obj.type != pdfium_raw.FPDF_PAGEOBJ_TEXT]
        allowed = {_caption_text(text) for text in allowed_texts}
        textpage = page.get_textpage()
        try:
            for index in range(textpage.count_rects()):
                left, bottom, right, top = textpage.get_rect(index)
                text = textpage.get_text_bounded(left, bottom, right, top, errors="strict")
                if _caption_text(text) not in allowed:
                    continue
                _validate_caption_texts([text])
                region = {"text": text, "pdf_bbox": [left, bottom, right, top],
                          "raster_guard_pixels": 1}
                if not all(math.isfinite(value) for value in (left, bottom, right, top)):
                    raise ValueError("Nonfinite PDF caption bounds")
                points = [converter.to_bitmap(horizontal, vertical)
                          for horizontal in (left, right) for vertical in (bottom, top)]
                box = [
                    min(point[0] for point in points) - 1,
                    min(point[1] for point in points) - 1,
                    max(point[0] for point in points) + 2,
                    max(point[1] for point in points) + 2,
                ]
                region["pixel_bbox"] = box
                if not (0 <= box[0] < box[2] <= bitmap.width
                        and 0 <= box[1] < box[3] <= bitmap.height):
                    result["rejected"].append({**region, "reason": "out_of_bounds"})
                    continue
                overlaps = False
                for obstacle in obstacles:
                    corners = [converter.to_bitmap(horizontal, vertical)
                               for horizontal in (obstacle[0], obstacle[2])
                               for vertical in (obstacle[1], obstacle[3])]
                    if (box[0] <= max(point[0] for point in corners)
                            and box[2] >= min(point[0] for point in corners)
                            and box[1] <= max(point[1] for point in corners)
                            and box[3] >= min(point[1] for point in corners)):
                        overlaps = True
                        break
                if overlaps:
                    result["rejected"].append({**region, "reason": "overlaps_nontext_object"})
                else:
                    result["regions"].append(region)
        finally:
            textpage.close()
    except Exception as error:
        result["regions"] = []
        result["error"] = f"{type(error).__name__}: {error}"
        result["uncertainty_flags"].append("pdf_caption_inspection_failed")
    return result


def iter_pages(input_path: str | Path, pages: Sequence[int] | None = None,
               dpi: float = 300, *, pdf_caption_texts: Sequence[str] = ()) -> Iterator[tuple]:
    """Yield (one_based_number, RGB_image, metadata), one page at a time.

    The yielded image is borrowed until the next iteration or generator close;
    consumers retaining it must copy it. PDFs render at dpi/72 with no crop.
    Raster images retain native resolution; multi-frame TIFFs are supported.
    RGB conversion/compositing defines the raw raster, before any cleanup.
    """
    from PIL import Image

    if not math.isfinite(dpi) or dpi <= 0:
        raise ValueError("DPI must be finite and positive")
    _validate_caption_texts(pdf_caption_texts)
    input_path = Path(input_path)
    if input_path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        with pdfium.PdfDocument(input_path) as document:
            for number in _selected_pages(pages, len(document)):
                page = document[number - 1]
                try:
                    bitmap = page.render(scale=dpi / 72, rotation=0, crop=(0, 0, 0, 0))
                    try:
                        with bitmap.to_pil() as rendered:
                            image = rendered.convert("RGB")
                        metadata = {"source_kind": "pdf", "dpi": dpi,
                                    "pdf_rotation": page.get_rotation(),
                                    "captions": pdf_caption_regions(page, bitmap, pdf_caption_texts)}
                    finally:
                        bitmap.close()
                finally:
                    page.close()
                try:
                    yield number, image, metadata
                finally:
                    image.close()
    else:
        with Image.open(input_path) as source:
            for number in _selected_pages(pages, getattr(source, "n_frames", 1)):
                source.seek(number - 1)
                if "A" in source.getbands() or "transparency" in source.info:
                    with source.convert("RGBA") as foreground:
                        image = Image.new("RGB", foreground.size, "white")
                        with foreground.getchannel("A") as alpha:
                            image.paste(foreground, mask=alpha)
                else:
                    image = source.convert("RGB")
                try:
                    yield number, image, {"source_kind": "image", "dpi": None,
                                          "resolution": "native_pixels",
                                          "captions": {"mode": "not_applicable", "regions": [],
                                                       "uncertainty_flags": []}}
                finally:
                    image.close()


def _validate_segmentation_mode(segmentation_mode: str) -> None:
    if segmentation_mode not in ("auto", "components", "baselines"):
        raise ValueError("Segmentation mode must be auto, components, or baselines")


def segment_page(image, model=None, *, segmentation_mode: str = "auto"):
    """Extract horizontal rows or learned baselines; auto tries rows first."""
    _validate_segmentation_mode(segmentation_mode)
    from kraken import binarization, pageseg
    from kraken.lib.exceptions import KrakenInputException

    if segmentation_mode != "baselines":
        try:
            with image.convert("L") as grayscale, binarization.nlbin(grayscale).convert("1") as binary:
                if binary.histogram()[0] >= 16:
                    segmentation = pageseg.segment(binary, text_direction="horizontal-lr",
                                                   black_colseps=True)
                    if segmentation.lines:
                        lines = [replace(line, bbox=(max(0, line.bbox[0] - 8),
                                                     max(0, line.bbox[1] - 8),
                                                     min(image.width, line.bbox[2] + 8),
                                                     min(image.height, line.bbox[3] + 8)))
                                 for line in segmentation.lines]
                        return replace(segmentation, lines=lines)
        except (ValueError, KrakenInputException) as error:
            if segmentation_mode == "components":
                raise ValueError("Component segmentation could not extract rows") from error
        if segmentation_mode == "components":
            raise ValueError("Component segmentation could not extract rows")

    from kraken.configs import SegmentationInferenceConfig
    from kraken.tasks import SegmentationTaskModel

    if model is None:
        model = SegmentationTaskModel.load_model()
    config = SegmentationInferenceConfig(precision="32-true", raise_on_error=True)
    if image.width > 2 * image.height:
        from PIL import Image

        with Image.new("RGB", (image.width, math.ceil(image.width / 2)), "white") as padded:
            padded.paste(image, (0, 0))
            return model.predict(padded, config)
    return model.predict(image, config)


def load_recognition_model(path, *, require_contract=False):
    """Restore the static tensor padding used by this project's trained models."""
    from kraken.tasks import RecognitionTaskModel
    import torch.nn.functional as functional

    path = Path(path)
    model = RecognitionTaskModel.load_model(path)
    contract_path = path.with_suffix(".input.json")
    if not contract_path.exists():
        if require_contract:
            raise ValueError("Missing recognition input contract; keep weights and .input.json together")
        return model
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    width = contract.get("max_width")
    if (contract.get("schema") != "khojki-ppocrv6-input-v1" or contract.get("model_sha256") != digest
            or type(width) is not int or not 64 <= width <= 4096 or contract.get("padding") != 16):
        raise ValueError("Invalid or mismatched recognition input contract")

    def pad_input(module, arguments):
        image = arguments[0]
        if image.shape[-1] > width:
            raise ValueError("Input exceeds the model's trained tensor width; refusing to clip it")
        return (functional.pad(image, (0, width - image.shape[-1])), *arguments[1:])

    model.net.nn.register_forward_pre_hook(pad_input)
    model.input_contract = contract
    return model


def _recognize_box_lines(image, segmentation, model, config, contract):
    """Split overwide rows at whitespace and return one record per source row."""
    import numpy as np
    from kraken import binarization

    for line in segmentation.lines:
        left, top, right, bottom = line.bbox
        limit = max(1, int((contract["max_width"] - 2 * contract["padding"])
                           * (bottom - top) / 96 * 0.85))
        if right - left <= limit:
            yield from model.predict(image, replace(segmentation, lines=[line]), config)
            continue
        with image.crop(tuple(line.bbox)) as crop, crop.convert("L") as grayscale:
            minimum, maximum = grayscale.getextrema()
            if minimum == maximum:
                raise ValueError("Overwide row has no whitespace split; refusing to clip text")
            with binarization.nlbin(grayscale).convert("1") as binary:
                whitespace = np.all(np.asarray(binary, dtype=bool), axis=0)
        edges = np.diff(np.pad(whitespace.astype(np.int8), (1, 1)))
        gaps = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
        cuts = [0]
        while right - left - cuts[-1] > limit:
            start = cuts[-1]
            candidates = [(int(begin), int(end)) for begin, end in gaps
                          if start < (begin + end) // 2 <= start + limit]
            preferred = [gap for gap in candidates
                         if (gap[0] + gap[1]) // 2 >= start + limit // 2]
            if not candidates:
                raise ValueError("Overwide row has no whitespace split; refusing to clip text")
            begin, end = max(preferred or candidates, key=lambda gap: (gap[1] - gap[0], gap[1]))
            cuts.append((begin + end) // 2)
        cuts.append(right - left)
        texts = []
        scores = []
        for part, (start, end) in enumerate(zip(cuts, cuts[1:])):
            fragment = replace(line, id=f"{line.id}-part-{part}",
                               bbox=(left + start, top, left + end, bottom))
            records = list(model.predict(image, replace(segmentation, lines=[fragment]), config))
            if len(records) != 1 or not records[0].prediction:
                raise ValueError("Incomplete recognition of an overwide row")
            texts.append(records[0].prediction)
            scores.extend(records[0].confidences if records[0].confidences is not None else [])
        yield SimpleNamespace(id=line.id, prediction="".join(texts), confidences=scores)


def recognize_page(image, segmentation, model):
    """Yield the custom model's raw records without filtering decoder symbols."""
    from kraken.configs import RecognitionInferenceConfig

    config = RecognitionInferenceConfig(precision="32-true", batch_size=1,
                                        num_line_workers=0, raise_on_error=True,
                                        bidi_reordering=False, no_legacy_polygons=True)
    contract = getattr(model, "input_contract", None)
    if (segmentation.type == "bbox" and isinstance(contract, dict)
            and contract.get("schema") == "khojki-ppocrv6-input-v1"):
        return _recognize_box_lines(image, segmentation, model, config, contract)
    return model.predict(image, segmentation, config)


def extract_line_crops(image, segmentation, directory: str | Path, page_number: int,
                       digest: str) -> Iterator[dict]:
    """Yield each saved private crop immediately so partial exports are auditable."""
    from kraken.lib.segmentation import extract_polygons

    if segmentation.type == "bbox":
        crops = ((image.crop(tuple(line.bbox)), line) for line in segmentation.lines)
    else:
        crops = extract_polygons(image, segmentation, legacy=False)
    for number, (crop, line) in enumerate(crops, 1):
        try:
            path = Path(directory) / f"page-{page_number:04d}-{digest[:12]}-line-{number:04d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                crop.save(stream, format="PNG")
            yield {"line_id": line.id, "path": str(path), "size": list(crop.size)}
        finally:
            crop.close()


def parse_pages(value: str | None) -> list[int] | None:
    """Parse unique, positive, one-based pages, preserving requested order."""
    if value is None:
        return None
    parts = value.split(",")
    if any(not part.strip().isascii() or not part.strip().isdecimal() for part in parts):
        raise ValueError("Pages must be comma-separated positive integers")
    pages = [int(part) for part in parts]
    if not pages or min(pages) < 1 or len(set(pages)) != len(pages):
        raise ValueError("Pages must be unique and one-based")
    return pages


def raster_statistics(image) -> dict:
    """Use every RGB pixel; even a small stain prevents automatic blank status."""
    import numpy as np

    if image.mode != "RGB" or image.width < 1 or image.height < 1:
        raise ValueError("Expected a nonempty RGB image")
    darkest = np.asarray(image).min(axis=2)
    return {
        "near_white_fraction": float(np.mean(darkest >= 250)),
        "minimum_channel": int(darkest.min()),
        "high_whiteness": bool(np.all(darkest >= 250)),
        "near_white_floor": 250,
    }


def confirmed_blank(statistics: dict, line_count: int) -> bool:
    """A conservative raster heuristic, never a learned no-target classifier."""
    return line_count == 0 and statistics["high_whiteness"] is True


def line_hypothesis(record, warn_confidence: float | None = None) -> dict:
    """Preserve raw decoder output and expose uncalibrated character scores."""
    if warn_confidence is not None and not 0 <= warn_confidence <= 1:
        raise ValueError("Warning confidence must be between zero and one")
    raw = record.prediction
    normalized = normalize_text(raw)
    scores = []
    raw_scores = [] if record.confidences is None else list(record.confidences)
    for score in raw_scores:
        try:
            numeric = float(score)
        except (TypeError, ValueError, OverflowError):
            numeric = math.nan
        scores.append(numeric)
    valid_scores = [score for score in scores if math.isfinite(score) and 0 <= score <= 1]
    flags = ["script_identity_unverified"]
    if invalid_codepoints(normalized):
        flags.append("invalid_codepoints")
    if not normalized.strip():
        flags.append("empty_hypothesis")
    if not scores:
        flags.append("missing_confidences")
    if len(valid_scores) != len(scores):
        flags.append("invalid_confidences")
    if len(scores) != len(raw):
        flags.append("confidence_length_mismatch")
    if warn_confidence is not None and any(score < warn_confidence for score in valid_scores):
        flags.append("below_warning_confidence")
    return {
        "raw_hypothesis": raw,
        "normalized_hypothesis": normalized,
        "raw_invalid_codepoints": invalid_codepoints(raw),
        "invalid_codepoints": invalid_codepoints(normalized),
        "character_confidences": [score if math.isfinite(score) else None for score in scores],
        "unserializable_confidences": {str(index): repr(raw_scores[index])
                           for index, score in enumerate(scores) if not math.isfinite(score)},
        "minimum_confidence": min(valid_scores) if valid_scores else None,
        "mean_confidence": sum(valid_scores) / len(valid_scores) if valid_scores else None,
        "confidence_note": "Decoder scores are not calibrated accuracy or script probabilities",
        "uncertainty_flags": flags,
    }


def _line_geometry(line, image) -> dict:
    box = getattr(line, "bbox", None)
    if box is not None:
        if len(box) != 4 or any(not math.isfinite(value) for value in box):
            raise ValueError("Invalid line bounding box")
        left, top, right, bottom = box
        if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
            raise ValueError("Line bounding box outside page")
        flags = ["line_touches_page_edge"] if (left == 0 or top == 0 or right == image.width
                                               or bottom == image.height) else []
        return {"id": str(line.id), "bbox": list(box), "baseline": [],
                "boundary": [[left, top], [right - 1, top],
                             [right - 1, bottom - 1], [left, bottom - 1]],
                "uncertainty_flags": flags}
    baseline = [[float(horizontal), float(vertical)] for horizontal, vertical in (line.baseline or [])]
    boundary = [[float(horizontal), float(vertical)] for horizontal, vertical in (line.boundary or [])]
    if any(not math.isfinite(value) for point in baseline + boundary for value in point):
        raise ValueError("Nonfinite line coordinates")
    flags = []
    if len(baseline) < 2 or len(boundary) < 3:
        flags.append("missing_line_geometry")
    if any(not (0 <= horizontal < image.width and 0 <= vertical < image.height)
           for horizontal, vertical in baseline + boundary):
        flags.append("line_geometry_out_of_bounds")
    if any(horizontal in (0, image.width - 1) or vertical in (0, image.height - 1)
           for horizontal, vertical in boundary):
        flags.append("line_touches_page_edge")
    return {"id": str(line.id), "baseline": baseline, "boundary": boundary,
            "uncertainty_flags": flags}


def process_page(image, page_number: int, *, segmenter, recognizer=None,
                 manifest: dict | None = None, metadata: dict | None = None,
                 segment_only: bool = False, warn_confidence: float | None = None,
                 output_dir: str | Path | None = None,
                 line_crops: str | Path | None = None, cleanup_mode: str = "none",
                 deskew: bool = False, crop_border: bool = False,
                 export_drafts: bool = False) -> dict:
    """Process one borrowed RGB page using injectable inference functions.

    segmenter(image) returns a Kraken Segmentation; recognizer(image, bounds)
    yields Kraken records. Only explicitly approved exact-raster references
    can currently become final transcripts. Unverified model output stays in
    the aggregate as raw_hypothesis/draft_text, even when every symbol is valid.
    Explicit segment_only disables reference replay as well as recognition.
    """
    if page_number < 1:
        raise ValueError("Page number must be one-based")
    if warn_confidence is not None and not 0 <= warn_confidence <= 1:
        raise ValueError("Warning confidence must be between zero and one")
    statistics = raster_statistics(image)
    digest = raster_digest(image.tobytes(), image.width, image.height, mode="RGB")
    metadata = metadata or {}
    report = {
        "page": page_number, "raster_digest": digest, "size": list(image.size),
        "source": metadata, "status": "review", "review_required": True,
        "provenance": None, "raw_raster_statistics": statistics,
        "preprocessing": {"border_handling": "preserve_full_page", "crop": None,
                          "caption_masks": [], "mask_note": "Disabled unless exact ASCII captions are explicitly named"},
        "segmentation": {"line_count": None, "region_count": None, "lines": []},
        "lines": [], "raw_hypothesis": None, "draft_text": None,
        "invalid_codepoints": [], "raw_invalid_codepoints": [],
        "transcript_path": None, "draft_path": None,
        "line_crops": [], "uncertainty_flags": [], "errors": [],
    }
    if manifest is not None and not segment_only:
        try:
            reference = approved_reference(manifest, digest)
        except (ValueError, TypeError, AttributeError) as error:
            report["uncertainty_flags"].append("invalid_reference")
            report["errors"].append(f"Reference: {type(error).__name__}: {error}")
            return report
        if reference is not None:
            report["provenance"] = "reference"
            report["reference"] = reference
            report["reference_policy_version"] = POLICY_VERSION
            report["review_required"] = False
            if reference["decision"] == "no_target_text":
                report["status"] = "no_target_text"
            else:
                text = validate_text(reference["text"])
                if output_dir is not None:
                    path = Path(output_dir) / "transcripts" / f"page-{page_number:04d}.txt"
                    write_transcript(path, text)
                    report["transcript_path"] = str(path)
                report["status"] = "transcribed"
            return report

    with ExitStack() as stack:
        working = stack.enter_context(image.copy())
        captions = metadata.get("captions", {})
        report["uncertainty_flags"].extend(captions.get("uncertainty_flags", []))
        for region in captions.get("regions", []):
            box = region["pixel_bbox"]
            _validate_caption_texts([region["text"]])
            if not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height):
                raise ValueError("Caption mask extends beyond the page")
            working.paste((255, 255, 255), tuple(box))
            report["preprocessing"]["caption_masks"].append(region)
        if report["preprocessing"]["caption_masks"]:
            report["preprocessing"]["mask_note"] = (
                "Only exact user-named ASCII PDF caption rectangles outside non-text objects were whitened; no margins cropped"
            )
        if cleanup_mode != "none":
            from khojki_ocr.preprocessing import prepare_page

            prepared, cleanup = prepare_page(working, mode=cleanup_mode, deskew=deskew, crop_border=crop_border)
            working = stack.enter_context(prepared)
            report["preprocessing"]["cleanup"] = cleanup
        try:
            segmentation = segmenter(working)
            if segmentation.type not in ("baselines", "bbox"):
                raise ValueError("Expected baseline or bounding-box segmentation")
            geometries = [_line_geometry(line, working) for line in segmentation.lines]
            report["segmentation"] = {
                "line_count": len(geometries), "lines": geometries,
                "region_count": sum(len(regions) for regions in (segmentation.regions or {}).values()),
                "reading_order": "Kraken segmentation.lines order; not manually verified",
            }
        except Exception as error:
            report["uncertainty_flags"].append("segmentation_failed")
            report["errors"].append(f"Segmentation: {type(error).__name__}: {error}")
            return report

        if not segmentation.lines:
            if confirmed_blank(statistics, 0) and not report["uncertainty_flags"]:
                report["status"] = "blank"
                report["review_required"] = False
            else:
                report["uncertainty_flags"].append("no_detected_lines_with_content")
            return report

        report["uncertainty_flags"].extend(["script_identity_unverified", "layout_unverified"])
        for geometry in geometries:
            report["uncertainty_flags"].extend(geometry["uncertainty_flags"])
        if line_crops is not None:
            try:
                for crop in extract_line_crops(working, segmentation, line_crops, page_number, digest):
                    report["line_crops"].append(crop)
            except Exception as error:
                report["uncertainty_flags"].append("line_crop_export_failed")
                report["errors"].append(f"Line crops: {type(error).__name__}: {error}")

        if segment_only or recognizer is None:
            report["status"] = "segmented"
            report["uncertainty_flags"].append("recognition_not_run")
            report["uncertainty_flags"] = sorted(set(report["uncertainty_flags"]))
            return report

        report["provenance"] = "model"
        try:
            for record in recognizer(working, segmentation):
                hypothesis = line_hypothesis(record, warn_confidence)
                hypothesis["line_id"] = str(getattr(record, "id", ""))
                if getattr(record, "recognition_error", None):
                    hypothesis["recognition_error"] = record.recognition_error
                    hypothesis["uncertainty_flags"].append("line_recognition_failed")
                    report["errors"].append(f"Line {hypothesis['line_id']}: {record.recognition_error}")
                report["lines"].append(hypothesis)
                report["uncertainty_flags"].extend(hypothesis["uncertainty_flags"])
        except Exception as error:
            report["uncertainty_flags"].append("recognition_failed")
            report["errors"].append(f"Recognition: {type(error).__name__}: {error}")
        if [line["line_id"] for line in report["lines"]] != [line["id"] for line in geometries]:
            report["uncertainty_flags"].append("recognition_line_mismatch")
        raw = "\n".join(line["raw_hypothesis"] for line in report["lines"])
        report["raw_hypothesis"] = raw
        report["draft_text"] = normalize_text(raw)
        report["recognition_failure_line_ids"] = [line["line_id"] for line in report["lines"]
                              if line.get("recognition_error")]
        returned_ids = {line["line_id"] for line in report["lines"]}
        report["recognition_missing_line_ids"] = [line["id"] for line in geometries
                              if line["id"] not in returned_ids]
        report["invalid_codepoints"] = invalid_codepoints(report["draft_text"])
        report["raw_invalid_codepoints"] = invalid_codepoints(raw)
        try:
            validate_text(report["draft_text"])
        except ValueError:
            report["uncertainty_flags"].append("invalid_codepoints")
        if not report["draft_text"].strip():
            report["uncertainty_flags"].append("suspicious_empty_recognition")
        if export_drafts and output_dir is not None and raw.strip() and not report["invalid_codepoints"]:
            path = Path(output_dir) / "drafts" / f"page-{page_number:04d}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="") as stream:
                stream.write(raw)
            report["draft_path"] = str(path)
            report["draft_review_note"] = "Unverified raw model text; empty rows may be failed recognition. Consult this report before reuse."
        report["uncertainty_flags"] = sorted(set(report["uncertainty_flags"]))
        return report


def prepare_output(directory: str | Path) -> Path:
    """Refuse existing nonempty runs; never delete or overwrite user outputs."""
    directory = Path(directory)
    if directory.is_symlink():
        raise FileExistsError("Output must not be a symbolic link")
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError("Output is nonempty; choose a new run directory")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_pipeline(input_path: str | Path, output_dir: str | Path, *,
                 pages: Sequence[int] | None = None, model_path: str | Path | None = None,
                 references: str | Path | None = None, disable_references: bool = False,
                 segment_only: bool = False, dpi: float = 300,
                 line_crops: str | Path | None = None, warn_confidence: float | None = None,
                 pdf_caption_texts: Sequence[str] = (), cleanup_mode: str = "none",
                 deskew: bool = False, crop_border: bool = False,
                 export_drafts: bool = False, segmentation_mode: str = "auto") -> dict:
    """Run local page inference and write one aggregate report.json.

    Debug crops go into a unique, lazily created run below line_crops, outside
    the export directory. Optional draft files preserve raw model text and do
    not become approved transcripts.
    Models are loaded once, on demand, so reference-only runs need no Kraken.
    No trained script classifier is supplied; model pages require review.
    """
    _validate_segmentation_mode(segmentation_mode)
    if not Path(input_path).is_file():
        raise FileNotFoundError(input_path)
    if not math.isfinite(dpi) or dpi <= 0:
        raise ValueError("DPI must be finite and positive")
    if warn_confidence is not None and not 0 <= warn_confidence <= 1:
        raise ValueError("Warning confidence must be between zero and one")
    _validate_caption_texts(pdf_caption_texts)
    if model_path is not None and not segment_only and not Path(model_path).is_file():
        raise FileNotFoundError(model_path)
    model_digest = None
    contract_digest = None
    if model_path is not None and not segment_only:
        with Path(model_path).open("rb") as stream:
            model_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        contract_path = Path(model_path).with_suffix(".input.json")
        if contract_path.is_file():
            with contract_path.open("rb") as stream:
                contract_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = None
    if references is not None and not disable_references and not segment_only:
        with Path(references).open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("pages"), dict):
            raise ValueError("Reference manifest must be an object with a pages object")
    if line_crops is not None:
        export = Path(output_dir).resolve()
        private = Path(line_crops).resolve()
        if export == private or export in private.parents or private in export.parents:
            raise ValueError("Line crops must be outside and separate from the export directory")
    output = prepare_output(output_dir)
    run_id = uuid.uuid4().hex
    crop_directory = Path(line_crops) / run_id if line_crops is not None else None
    models = {"segmentation": None, "recognition": None}

    def segment(image):
        if segmentation_mode != "components" and models["segmentation"] is None:
            from kraken.tasks import SegmentationTaskModel

            models["segmentation"] = SegmentationTaskModel.load_model()
        if segmentation_mode == "auto":
            return segment_page(image, models["segmentation"])
        return segment_page(image, models["segmentation"], segmentation_mode=segmentation_mode)

    def recognize(image, segmentation):
        if models["recognition"] is None:
            models["recognition"] = load_recognition_model(model_path, require_contract=True)
        for line in segmentation.lines:
            try:
                yield from recognize_page(image, replace(segmentation, lines=[line]), models["recognition"])
            except Exception as error:
                yield SimpleNamespace(id=line.id, prediction="", confidences=[],
                                      recognition_error=f"{type(error).__name__}: {error}")

    report = {
        "run_id": run_id, "input": str(Path(input_path).resolve()),
        "policy_version": POLICY_VERSION, "status": "completed", "pages": [], "errors": [],
        "config": {"pages": list(pages) if pages is not None else None, "dpi": dpi,
                   "mode": "segment_only" if segment_only or model_path is None else "recognition",
                   "segmentation_mode": segmentation_mode,
                   "model": str(model_path) if model_path is not None else None,
                   "model_sha256": model_digest,
                   "input_contract_sha256": contract_digest,
                   "references": str(references) if manifest is not None else None,
                   "references_enabled": manifest is not None,
                   "export_drafts": export_drafts,
                   "warn_confidence": warn_confidence, "precision": "32-true",
                   "pdf_caption_texts": list(pdf_caption_texts),
                   "line_crops": str(crop_directory) if crop_directory is not None else None},
        "limitations": [
            "No trained Khojki/non-target image classifier; valid decoder symbols do not prove image script",
            "Model hypotheses always require review and are withheld from final transcripts",
            "BLLA layout and reading order are pretrained, not validated on these manuscripts",
            "Character confidences and raster heuristics are not measured OCR accuracy",
            "Caption masking is opt-in, exact ASCII text only, outside non-text objects; embedded scan captions remain",
            "PDF scale assumes 72 canvas units per inch; unusual PDF UserUnit values need review",
        ],
    }
    with (output / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        try:
            with closing(iter_pages(input_path, pages, dpi, pdf_caption_texts=pdf_caption_texts)) as page_iterator:
                for number, image, metadata in page_iterator:
                    report["pages"].append(process_page(
                        image, number, segmenter=segment,
                        recognizer=recognize if model_path is not None and not segment_only else None,
                        manifest=manifest, metadata=metadata, segment_only=segment_only,
                        warn_confidence=warn_confidence, output_dir=output, line_crops=crop_directory,
                        cleanup_mode=cleanup_mode, deskew=deskew, crop_border=crop_border,
                        export_drafts=export_drafts,
                    ))
        except Exception as error:
            report["status"] = "failed"
            report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            counts = Counter(page["status"] for page in report["pages"])
            report["summary"] = {
                "processed_pages": len(report["pages"]), "statuses": dict(counts),
                "detected_lines": sum(page["segmentation"]["line_count"] or 0 for page in report["pages"]),
                "reference_pages": sum(page["provenance"] == "reference" for page in report["pages"]),
                "model_pages": sum(page["provenance"] == "model" for page in report["pages"]),
                "review_pages": sum(page["review_required"] for page in report["pages"]),
                "transcripts_written": sum(page["transcript_path"] is not None for page in report["pages"]),
                "drafts_written": sum(page.get("draft_path") is not None for page in report["pages"]),
                "pages_with_errors": sum(bool(page["errors"]) for page in report["pages"]),
            }
            if report["status"] != "failed":
                if report["summary"]["pages_with_errors"]:
                    report["status"] = "completed_with_errors"
                elif report["summary"]["review_pages"]:
                    report["status"] = "completed_with_review"
            report["models_loaded"] = {name: model is not None for name, model in models.items()}
            json.dump(report, stream, ensure_ascii=True, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local Khojki page segmentation and reviewable recognition")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pages", help="Unique one-based page numbers, e.g. 17,18")
    parser.add_argument("--model", type=Path, help="Trained custom Kraken recognition model; optional")
    parser.add_argument("--references", type=Path, help="Explicitly approved exact-raster reference manifest")
    parser.add_argument("--disable-references", action="store_true")
    parser.add_argument("--segment-only", action="store_true", help="Disable recognition and reference transcript replay")
    parser.add_argument("--segmentation-mode", choices=("auto", "components", "baselines"), default="auto",
                        help="auto tries classical rows first; components forbids fallback; baselines forces the learned detector")
    parser.add_argument("--export-drafts", action="store_true", help="Write nonempty target-only raw model text under drafts/, still requiring review")
    parser.add_argument("--dpi", type=float, default=300)
    parser.add_argument("--cleanup", choices=("none", "conservative", "normalized"), default="none")
    parser.add_argument("--deskew", action="store_true")
    parser.add_argument("--crop-border", action="store_true")
    parser.add_argument("--line-crops", type=Path, help="Private crop root, separate from transcript export")
    parser.add_argument("--warn-confidence", type=float, help="Optional uncalibrated warning threshold in [0,1]")
    parser.add_argument("--pdf-caption-text", action="append", default=[],
                        help="Exact known external ASCII caption; repeat for multiple captions; default leaves all text intact")
    args = parser.parse_args(argv)
    try:
        report = run_pipeline(
            args.input, args.output, pages=parse_pages(args.pages), model_path=args.model,
            references=args.references, disable_references=args.disable_references,
            segment_only=args.segment_only, dpi=args.dpi, line_crops=args.line_crops,
            cleanup_mode=args.cleanup, deskew=args.deskew, crop_border=args.crop_border,
            warn_confidence=args.warn_confidence, pdf_caption_texts=args.pdf_caption_text,
            export_drafts=args.export_drafts, segmentation_mode=args.segmentation_mode,
        )
    except (OSError, ValueError, ImportError) as error:
        print(f"khojki-ocr: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], **report["summary"],
                      "report": str(args.output / "report.json")}))
    return 1 if report["status"] in ("failed", "completed_with_errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())