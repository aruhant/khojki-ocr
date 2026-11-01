"""Code-point CER against reviewed lines, never silently filtering predictions.

References: {policy_version, dataset_kind?, lines: [{id, text, status, ...}]}.
Predictions: [{id, text, provenance, ...}] or {lines: [...]}.
The explicit pipeline adapter accepts {pages: [{page, provenance, lines:
[{line_id, raw_hypothesis, ...}]}]}; IDs must be globally unique and match gold.
Metadata such as page, image, and bbox is retained on reference result lines.
Confusion keys use U+XXXX labels and <empty>, ordered by descending frequency.
"""

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from khojki_ocr.core import POLICY_VERSION, invalid_codepoints, normalize_text


def _index_lines(lines: object, kind: str) -> dict[str, dict]:
    if not isinstance(lines, list):
        raise ValueError(f"{kind} lines must be a list")
    indexed = {}
    for line in lines:
        if not isinstance(line, dict):
            raise ValueError(f"Each {kind.lower()} line must be an object")
        identifier = line.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"Each {kind.lower()} line requires a nonempty string id")
        if identifier in indexed:
            raise ValueError(f"Duplicate {kind.lower()} id: {identifier!r}")
        if not isinstance(line.get("text"), str):
            raise ValueError(f"{kind} {identifier!r} requires string text")
        indexed[identifier] = dict(line)
    return indexed


def _provenance(value: object, allow_reference: bool) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Prediction provenance is required, including for empty pipeline pages")
    if value != "model" and not allow_reference:
        raise ValueError("Non-model provenance requires --allow-reference; it is not OCR accuracy")
    return value


def normalize_predictions(predictions: dict | list, *, allow_reference: bool = False) -> dict:
    """Return canonical lines, observed provenances, and the explicit input format.

Flat records require their own provenance; pipeline records inherit the required
page provenance. Conflicting container/line provenance is always an error.
No positional matching, ID synthesis, text filtering, or draft-text fallback.
"""
    provenances = set()
    container_provenance = None
    if isinstance(predictions, dict) and "provenance" in predictions:
        container_provenance = _provenance(predictions["provenance"], allow_reference)
        provenances.add(container_provenance)
    if isinstance(predictions, dict) and "pages" in predictions:
        input_format = "pipeline_aggregate"
        if "lines" in predictions or not isinstance(predictions["pages"], list):
            raise ValueError("Pipeline predictions require pages as a list and no top-level lines")
        lines = []
        for page in predictions["pages"]:
            if not isinstance(page, dict) or not isinstance(page.get("lines"), list):
                raise ValueError("Each pipeline page requires a lines list")
            provenance = _provenance(page.get("provenance"), allow_reference)
            if container_provenance is not None and provenance != container_provenance:
                raise ValueError("Conflicting container/page provenance")
            provenances.add(provenance)
            for line in page["lines"]:
                if not isinstance(line, dict):
                    raise ValueError("Each pipeline line must be an object")
                if "provenance" in line and line["provenance"] != provenance:
                    raise ValueError("Conflicting page/line provenance")
                lines.append({"id": line.get("line_id"), "text": line.get("raw_hypothesis"),
                              "provenance": provenance, "page": page.get("page")})
    elif isinstance(predictions, dict) and "lines" in predictions:
        input_format = "lines_object"
        lines = predictions["lines"]
    elif isinstance(predictions, list):
        input_format = "lines_list"
        lines = predictions
    else:
        raise ValueError("Predictions must be a lines list, a lines object, or a pipeline aggregate")
    indexed = _index_lines(lines, "Prediction")
    for line in indexed.values():
        provenance = _provenance(line.get("provenance"), allow_reference)
        if container_provenance is not None and provenance != container_provenance:
            raise ValueError("Conflicting container/line provenance")
        provenances.add(provenance)
        line["text"] = normalize_text(line["text"])
    return {"lines": list(indexed.values()), "provenances": sorted(provenances),
            "input_format": input_format}


def _cer(errors: int, reference_characters: int) -> float | None:
    if reference_characters:
        return errors / reference_characters
    return None if errors else 0.0


def _ordered_confusions(confusions: Counter) -> dict[str, int]:
    return dict(sorted(confusions.items(), key=lambda pair: (-pair[1], pair[0])))


def score_line(reference: str, prediction: str) -> dict:
    """Score NFC/line-ending-normalized Unicode strings, counting all characters.

CER is null for an empty reference with insertions, zero for two empty strings.
Edit positions and S/I/D counts follow RapidFuzz's optimal edit alignment.
"""
    if not isinstance(reference, str) or not isinstance(prediction, str):
        raise ValueError("Reference and prediction text must be strings")
    reference = normalize_text(reference)
    prediction = normalize_text(prediction)
    counts = Counter()
    confusions = Counter()
    for operation in Levenshtein.editops(reference, prediction):
        counts[operation.tag] += 1
        source = ("<empty>" if operation.tag == "insert"
                  else f"U+{ord(reference[operation.src_pos]):04X}")
        target = ("<empty>" if operation.tag == "delete"
                  else f"U+{ord(prediction[operation.dest_pos]):04X}")
        confusions[f"{source} -> {target}"] += 1
    errors = sum(counts.values())
    return {
        "reference_text": reference, "prediction_text": prediction,
        "reference_characters": len(reference), "prediction_characters": len(prediction),
        "substitutions": counts["replace"], "insertions": counts["insert"],
        "deletions": counts["delete"], "errors": errors,
        "cer": _cer(errors, len(reference)), "exact_match": reference == prediction,
        "invalid_codepoints": invalid_codepoints(prediction),
        "reference_invalid_codepoints": invalid_codepoints(reference),
        "confusion_pairs": _ordered_confusions(confusions),
    }


def evaluate(references: dict, predictions: dict | list, *, allow_reference: bool = False) -> dict:
    """Evaluate every reviewed/approved reference ID; missing predictions are empty.

Unknown predictions, duplicate IDs, unreviewed references, and incompatible
policy versions are errors. An empty dataset has null exact-line accuracy.
Non-model inputs label the entire result non_model_comparison, even if mixed.
"""
    if not isinstance(references, dict) or references.get("policy_version") != POLICY_VERSION:
        raise ValueError(f"References require policy_version {POLICY_VERSION!r}")
    dataset_kind = references.get("dataset_kind", "unspecified")
    if dataset_kind not in ("real", "synthetic", "mixed", "unspecified"):
        raise ValueError("dataset_kind must be real, synthetic, mixed, or unspecified")
    gold = _index_lines(references.get("lines"), "Reference")
    for identifier, line in gold.items():
        if line.get("status") not in ("reviewed", "approved"):
            raise ValueError(f"Reference {identifier!r} must be reviewed or approved")
    normalized = normalize_predictions(predictions, allow_reference=allow_reference)
    hypotheses = {line["id"]: line for line in normalized["lines"]}
    unknown = sorted(hypotheses.keys() - gold.keys())
    if unknown:
        raise ValueError(f"Unknown prediction IDs: {unknown!r}")
    warnings = []
    if dataset_kind == "unspecified":
        warnings.append("dataset_kind is unspecified; do not infer real or synthetic evidence")
    elif dataset_kind in ("synthetic", "mixed"):
        warnings.append("Synthetic or mixed evidence does not establish real-only OCR accuracy")
    non_model = any(provenance != "model" for provenance in normalized["provenances"])
    if non_model:
        warnings.append("Non-model comparison only; these scores are not OCR accuracy")
    totals = Counter()
    confusions = Counter()
    invalid = set()
    reference_invalid = set()
    lines = []
    for identifier, reference in gold.items():
        prediction = hypotheses.get(identifier)
        metrics = score_line(reference["text"], prediction["text"] if prediction is not None else "")
        for name in ("reference_characters", "prediction_characters", "substitutions",
                     "insertions", "deletions", "errors"):
            totals[name] += metrics[name]
        totals["exact_line_count"] += int(metrics["exact_match"])
        totals["missing_prediction_count"] += int(prediction is None)
        confusions.update(metrics["confusion_pairs"])
        invalid.update(metrics["invalid_codepoints"])
        reference_invalid.update(metrics["reference_invalid_codepoints"])
        lines.append({"id": identifier, "reference_metadata": {
            key: value for key, value in reference.items() if key not in ("id", "text")},
            "provenance": prediction["provenance"] if prediction is not None else None,
            "missing_prediction": prediction is None, **metrics})
    if totals["missing_prediction_count"]:
        warnings.append("Missing prediction IDs were scored as empty strings")
    if reference_invalid:
        warnings.append("Reviewed references contain invalid code points; they were retained and counted")
    return {
        "policy_version": POLICY_VERSION, "dataset_kind": dataset_kind,
        "result_mode": "non_model_comparison" if non_model else "model_evaluation",
        "allow_reference": allow_reference, "prediction_provenances": normalized["provenances"],
        "prediction_format": normalized["input_format"], "metric": "codepoint_cer",
        "normalization": "core.normalize_text: NFC and CRLF/CR to LF",
        "whitespace": "counted", "punctuation": "counted", "warnings": warnings,
        "line_count": len(gold), "prediction_count": len(hypotheses),
        **{name: totals[name] for name in ("reference_characters", "prediction_characters",
           "substitutions", "insertions", "deletions", "errors", "exact_line_count",
           "missing_prediction_count")},
        "cer": _cer(totals["errors"], totals["reference_characters"]),
        "exact_line_accuracy": totals["exact_line_count"] / len(gold) if gold else None,
        "invalid_codepoints": sorted(invalid), "reference_invalid_codepoints": sorted(reference_invalid),
        "confusion_pairs": _ordered_confusions(confusions), "lines": lines,
    }


def evaluate_detected_lines(references: dict, page_reports: dict) -> dict:
    """Score annotated ROIs after page OCR, assigning by geometry, never text.

    Coordinates are mapped through cleanup's inverse affine, then scaled from
    the original raster to source_preview (or explicit source_size). At least
    half the detected box must overlap the ROI and a baseline/box centre must
    lie inside it. A region sharing any fragment with another region is withheld
    whole. Fragments are concatenated without invented whitespace only for a
    clear LTR row. Regions with failed line recognition are flagged and missing.
    Recognition runs whose pages detected no lines score those regions as missing.
    Other page lines are unscored because these annotations are not exhaustive.
    """
    evaluate(references, [])
    gold = _index_lines(references["lines"], "Reference")

    def numbers(value, count, name):
        if not isinstance(value, (list, tuple)) or len(value) != count or any(
                type(number) not in (int, float) or not math.isfinite(number) for number in value):
            raise ValueError(f"Invalid {name}")
        return list(value)

    def size(value, name):
        dimensions = numbers(value, 2, name)
        if any(number <= 0 for number in dimensions):
            raise ValueError(f"Invalid {name}")
        return dimensions

    if not isinstance(page_reports, dict) or not isinstance(page_reports.get("pages"), list):
        raise ValueError("Page reports require an aggregate pages list")
    if "lines" in page_reports or ("provenance" in page_reports and
                                  page_reports["provenance"] != "model"):
        raise ValueError("Page reports require model provenance and no top-level lines")
    config = page_reports.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Invalid page report config")
    model_intended = (config.get("mode") == "recognition" and isinstance(config.get("model"), str)
                      and bool(config["model"].strip()) and config.get("references_enabled") is False)
    identity = {}
    for name, key in (("source_run_id", "run_id"), ("model_sha256", "model_sha256")):
        value = config.get(key, page_reports.get(key))
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Invalid {key}")
        identity[name] = value
    pages = {}
    geometry_ids = set()
    prediction_ids = set()
    detected = []
    failed_ids = []
    no_detection_page_ids = []
    for page in page_reports["pages"]:
        if not isinstance(page, dict) or type(page.get("page")) is not int or page["page"] < 1:
            raise ValueError("Each report requires a one-based page ID")
        page_id = page["page"]
        if page_id in pages:
            raise ValueError(f"Duplicate page ID: {page_id}")
        original_size = size(page.get("size"), "report size")
        segmentation = page.get("segmentation")
        if not isinstance(segmentation, dict) or not isinstance(segmentation.get("lines"), list):
            raise ValueError("Each page requires segmentation.lines")
        geometries = segmentation["lines"]
        if not isinstance(page.get("lines"), list):
            raise ValueError("Each page requires recognized lines")
        empty = page.get("provenance") is None and not geometries and not page["lines"]
        blank = (empty and page.get("status") == "blank" and not page.get("errors")
                 and not page.get("uncertainty_flags"))
        undetected = empty and model_intended and page.get("status") == "review"
        if page.get("reference") is not None or (page.get("provenance") != "model"
                                                  and not blank and not undetected):
            raise ValueError("Page evaluation requires model provenance (or a genuine blank page)")
        if undetected:
            no_detection_page_ids.append(page_id)
        preprocessing = page.get("preprocessing", {})
        if not isinstance(preprocessing, dict):
            raise ValueError("Invalid preprocessing geometry")
        cleanup = preprocessing.get("cleanup", {})
        if not isinstance(cleanup, dict):
            raise ValueError("Invalid cleanup geometry")
        affine = preprocessing.get("cleanup_to_original", cleanup.get("to_original"))
        if affine is None:
            if cleanup or preprocessing.get("crop") is not None:
                raise ValueError("Cleanup geometry requires an inverse affine")
            affine = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        if not isinstance(affine, (list, tuple)) or len(affine) not in (2, 3):
            raise ValueError("Invalid inverse affine")
        affine = [numbers(row, 3, "inverse affine") for row in affine]
        if len(affine) == 3 and affine[2] != [0, 0, 1]:
            raise ValueError("Invalid inverse affine bottom row")
        if abs(affine[0][0] * affine[1][1] - affine[0][1] * affine[1][0]) < 1e-12:
            raise ValueError("Singular inverse affine")
        working_size = size(cleanup.get("prepared_size", original_size), "prepared size")
        if "original_size" in cleanup and size(cleanup["original_size"], "original size") != original_size:
            raise ValueError("Conflicting original raster sizes")
        hypotheses = {}
        for line in page["lines"]:
            if not isinstance(line, dict):
                raise ValueError("Invalid recognized line")
            identifier = line.get("line_id")
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError("Recognized lines require nonempty line_id")
            if identifier in prediction_ids:
                raise ValueError(f"Duplicate prediction id: {identifier!r}")
            prediction_ids.add(identifier)
            if not isinstance(line.get("raw_hypothesis"), str):
                raise ValueError("Recognized lines require raw_hypothesis text")
            if "provenance" in line and line["provenance"] != "model":
                raise ValueError("Recognized lines require model provenance")
            hypotheses[identifier] = line["raw_hypothesis"]
            if line.get("recognition_error"):
                failed_ids.append(identifier)
        local_ids = set()
        for geometry in geometries:
            if not isinstance(geometry, dict):
                raise ValueError("Invalid line geometry")
            identifier = geometry.get("id")
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError("Geometry requires a nonempty id")
            if identifier in geometry_ids:
                raise ValueError(f"Duplicate geometry id: {identifier!r}")
            geometry_ids.add(identifier)
            local_ids.add(identifier)
            transformed = {}
            for name, minimum in (("boundary", 3), ("baseline", 2)):
                points = geometry.get(name, [])
                if not isinstance(points, list) or (len(points) < minimum and
                                                   (name == "boundary" or points)):
                    raise ValueError(f"Invalid {name} geometry")
                transformed[name] = []
                for point in points:
                    horizontal, vertical = numbers(point, 2, name)
                    if not (0 <= horizontal <= working_size[0] and 0 <= vertical <= working_size[1]):
                        raise ValueError("Line geometry is out of bounds")
                    mapped = [row[0] * horizontal + row[1] * vertical + row[2] for row in affine[:2]]
                    if any(not math.isfinite(value) for value in mapped) or not (
                            0 <= mapped[0] <= original_size[0] and 0 <= mapped[1] <= original_size[1]):
                        raise ValueError("Mapped line geometry is out of bounds")
                    transformed[name].append(mapped)
            boundary = transformed["boundary"]
            box = [min(point[0] for point in boundary), min(point[1] for point in boundary),
                   max(point[0] for point in boundary), max(point[1] for point in boundary)]
            area = abs(sum(point[0] * following[1] - following[0] * point[1]
                           for point, following in zip(boundary, boundary[1:] + boundary[:1]))) / 2
            if box[0] >= box[2] or box[1] >= box[3] or area <= 0:
                raise ValueError("Degenerate line geometry")
            centres = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]]
            baseline = transformed["baseline"]
            if baseline:
                centres.append([(min(point[0] for point in baseline) + max(point[0] for point in baseline)) / 2,
                                sum(point[1] for point in baseline) / len(baseline)])
            detected.append({"id": identifier, "page": page_id, "box": box, "centres": centres,
                             "text": hypotheses.get(identifier),
                             "recognition_failed": identifier in failed_ids})
        if hypotheses.keys() - local_ids:
            raise ValueError("Recognized line IDs lack segmentation geometry")
        pages[page_id] = page

    regions = {}
    preview_sizes = {}
    for identifier, reference in gold.items():
        page_id = reference.get("page")
        if type(page_id) is not int or page_id not in pages:
            raise ValueError(f"Missing page report for reference {identifier!r}")
        dimensions = reference.get("source_size", pages[page_id].get("source_size"))
        if dimensions is None:
            from PIL import Image

            source = reference.get("source_preview")
            if not isinstance(source, str) or not source:
                raise ValueError("References require source_preview or source_size")
            with Image.open(source) as image:
                dimensions = image.size
        dimensions = size(dimensions, "source preview size")
        if page_id in preview_sizes and preview_sizes[page_id] != dimensions:
            raise ValueError("Conflicting source preview sizes on one page")
        preview_sizes[page_id] = dimensions
        box = numbers(reference.get("bbox"), 4, "reference bbox")
        if not (0 <= box[0] < box[2] <= dimensions[0] and 0 <= box[1] < box[3] <= dimensions[1]):
            raise ValueError("Reference bbox is out of bounds")
        regions[identifier] = {"page": page_id, "box": box, "size": dimensions}

    assigned = {identifier: [] for identifier in gold}
    shared = {identifier: [] for identifier in gold}
    ambiguous_ids = []
    unmatched_ids = []
    for line in detected:
        candidates = []
        for identifier, region in regions.items():
            if region["page"] != line["page"]:
                continue
            scale = [region["size"][axis] / pages[line["page"]]["size"][axis] for axis in (0, 1)]
            box = [value * scale[index % 2] for index, value in enumerate(line["box"])]
            centres = [[point[axis] * scale[axis] for axis in (0, 1)] for point in line["centres"]]
            roi = region["box"]
            overlap = max(0, min(box[2], roi[2]) - max(box[0], roi[0])) * max(
                0, min(box[3], roi[3]) - max(box[1], roi[1]))
            inside = any(roi[0] <= point[0] < roi[2] and roi[1] <= point[1] < roi[3] for point in centres)
            if inside and overlap / ((box[2] - box[0]) * (box[3] - box[1])) >= 0.5:
                candidates.append((identifier, {**line, "box": box, "centres": centres}))
        if len(candidates) == 1:
            identifier, candidate = candidates[0]
            assigned[identifier].append(candidate)
        elif candidates:
            ambiguous_ids.append(line["id"])
            for identifier, _ in candidates:
                shared[identifier].append(line)
        else:
            unmatched_ids.append(line["id"])

    predictions = []
    matches = {}
    for identifier, fragments in assigned.items():
        fragments.sort(key=lambda fragment: (fragment["box"][0], fragment["id"]))
        clear = True
        for left, right in zip(fragments, fragments[1:]):
            first, second = left["box"], right["box"]
            height = min(first[3] - first[1], second[3] - second[1])
            vertical_overlap = min(first[3], second[3]) - max(first[1], second[1])
            baseline_distance = abs(left["centres"][-1][1] - right["centres"][-1][1])
            clear = clear and first[2] <= second[0] and vertical_overlap >= height / 2 and baseline_distance <= height / 4
        failed = [line["id"] for line in fragments + shared[identifier] if line["recognition_failed"]]
        usable = (bool(fragments) and clear and not shared[identifier] and not failed
                  and all(fragment["text"] is not None for fragment in fragments))
        matched_ids = [fragment["id"] for fragment in fragments] if usable else []
        matches[identifier] = {"matched_line_ids": matched_ids,
                               "candidate_line_ids": [fragment["id"] for fragment in fragments],
                               "shared_candidate_line_ids": [line["id"] for line in shared[identifier]],
                               "split_reference": len(fragments) > 1,
                               "geometry_ambiguous": not clear or bool(shared[identifier]),
                               "recognition_failed": bool(failed),
                               "recognition_failed_line_ids": failed}
        if usable:
            predictions.append({"id": identifier, "text": "".join(fragment["text"] for fragment in fragments),
                                "provenance": "model", "matched_line_ids": matched_ids})
    result = evaluate(references, {"provenance": "model", "lines": predictions})
    for line in result["lines"]:
        line.update(matches[line["id"]])
    result.update(scope="end_to_end_annotated_regions", prediction_format="geometry_matched_page_reports",
                  assignment="geometry_only", reference_text_used_for_assignment=False, model_rerun=False,
                  missing_annotated_line_count=result["missing_prediction_count"],
                  split_count=sum(match["split_reference"] for match in matches.values()),
                  matched_annotated_line_count=len(predictions),
                  annotated_line_coverage=len(predictions) / len(gold) if gold else None,
                  matched_detected_line_count=sum(len(match["matched_line_ids"]) for match in matches.values()),
                  detected_line_count=len(detected), unmatched_detected_line_ids=unmatched_ids,
                  ambiguous_detected_line_ids=ambiguous_ids,
                  recognition_failed_line_ids=failed_ids, recognition_failed_line_count=len(failed_ids),
                  recognition_failed_annotated_line_ids=[
                      identifier for identifier, match in matches.items() if match["recognition_failed"]],
                  no_detection_page_ids=no_detection_page_ids,
                  pages_with_errors=[page_id for page_id, page in pages.items() if page.get("errors")],
                  source_report_status=page_reports.get("status"), **identity,
                  raw_predictions={"provenance": "model", "lines": predictions,
                                   "scope": "end_to_end_annotated_regions"})
    result["warnings"].extend([
        "Only annotated regions are scored; unlabelled page lines are not assumed false positives",
        "CER includes recognition and Unicode semantic mapping; valid code points do not prove correct mapping",
        "Fragments are concatenated without inferred spaces; ambiguous geometry is scored as missing",
    ])
    if failed_ids:
        result["warnings"].append("Lines with recognition errors are flagged; their regions are scored as missing")
    if no_detection_page_ids:
        result["warnings"].append("Recognition pages without detected lines score their regions as missing")
    if page_reports.get("status") not in (None, "completed") or result["pages_with_errors"]:
        result["warnings"].append(f"Source report status {page_reports.get('status')!r} with page errors; "
                                  "failures are not clean output")
    return result


def write_result(result: dict, output: str | Path) -> Path:
    """Serialize before making directories, then exclusively create a new JSON file."""
    serialized = json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    return output


def predict_reference_lines(references: dict, model_path: Path, *, predictor=None,
                            cleanup_mode: str = "none") -> dict:
    """Measure recognition on manual crops; this does not score line detection."""
    from PIL import Image

    if cleanup_mode not in ("none", "conservative", "normalized"):
        raise ValueError("Unknown cleanup mode")
    records = _index_lines(references.get("lines"), "Reference")
    started = time.monotonic()
    if predictor is None:
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        import torch
        from kraken.configs import RecognitionInferenceConfig
        from kraken.containers import BBoxLine, Segmentation
        from khojki_ocr.pipeline import load_recognition_model

        torch.set_num_threads(2)
        model = load_recognition_model(model_path, require_contract=True)
        config = RecognitionInferenceConfig(precision="32-true", batch_size=1,
                                            num_line_workers=0, raise_on_error=True,
                                            bidi_reordering=False)

        def predictor(image, identifier):
            line = BBoxLine(id=identifier, bbox=(0, 0, image.width, image.height), base_dir="L")
            bounds = Segmentation(type="bbox", imagename=identifier,
                                  text_direction="horizontal-lr", script_detection=False,
                                  lines=[line])
            predictions = list(model.predict(image, bounds, config))
            if len(predictions) != 1:
                raise ValueError(f"Expected one prediction for {identifier}")
            return predictions[0].prediction

    lines = []
    for identifier, reference in records.items():
        source = Path(reference["source_preview"])
        box = reference.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box):
            raise ValueError(f"Invalid source crop for {identifier}")
        with Image.open(source) as original:
            left, top, right, bottom = box
            if not 0 <= left < right <= original.width or not 0 <= top < bottom <= original.height:
                raise ValueError(f"Out-of-bounds source crop for {identifier}")
            with original.crop(tuple(box)).convert("RGB") as crop:
                if cleanup_mode == "none":
                    prediction = predictor(crop, identifier)
                else:
                    from khojki_ocr.preprocessing import prepare_page

                    prepared, _ = prepare_page(crop, mode=cleanup_mode)
                    with prepared:
                        prediction = predictor(prepared, identifier)
        if not isinstance(prediction, str):
            raise TypeError("Recognizer must return text")
        lines.append({"id": identifier, "text": prediction, "provenance": "model",
                      "source_preview": str(source), "bbox": box})
    with Path(model_path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"lines": lines, "model_sha256": digest,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "scope": "recognition_only_manual_crops", "cleanup_mode": cleanup_mode,
            "reference_text_given_to_model": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path)
    source.add_argument("--model", type=Path, help="Recognize each reference crop before scoring")
    source.add_argument("--page-report", type=Path, help="Match existing page OCR to annotated ROIs by geometry")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cleanup", choices=("none", "conservative", "normalized"), default="none")
    parser.add_argument("--allow-reference", action="store_true",
                        help="Permit explicit non-model provenance as a non-OCR comparison")
    args = parser.parse_args(argv)
    try:
        with args.references.open(encoding="utf-8") as stream:
            references = json.load(stream)
        if args.page_report is not None:
            if args.allow_reference:
                raise ValueError("--allow-reference is not permitted with --page-report")
            raw_report = args.page_report.read_bytes()
            result = evaluate_detected_lines(references, json.loads(raw_report.decode("utf-8")))
            result["source_report_sha256"] = hashlib.sha256(raw_report).hexdigest()
            write_result(result["raw_predictions"], args.output.with_name(args.output.stem + ".predictions.json"))
        elif args.model is not None:
            predictions = predict_reference_lines(references, args.model, cleanup_mode=args.cleanup)
            write_result(predictions, args.output.with_name(args.output.stem + ".predictions.json"))
        else:
            with args.predictions.open(encoding="utf-8") as stream:
                predictions = json.load(stream)
        if args.page_report is None:
            result = evaluate(references, predictions, allow_reference=args.allow_reference)
        if args.model is not None:
            result["inference"] = {key: value for key, value in predictions.items() if key != "lines"}
            result["label_provenance"] = references.get("label_provenance", "unspecified")
        write_result(result, args.output)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())