"""Opt-in raster cleanup and advisory, script-agnostic region evidence.

Coordinates use pixel centres, x right and y down. ``to_original`` maps
prepared-image coordinates to the untouched input; ``from_original`` is its
inverse. No function identifies Khojki, removes captions, or certifies OCR.
"""

import math
from collections.abc import Mapping

import cv2
import numpy as np
from PIL import Image


def _rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        if image.width < 1 or image.height < 1:
            raise ValueError("Expected a nonempty image")
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            return Image.alpha_composite(Image.new("RGBA", image.size, "white"), rgba).convert("RGB")
        if image.mode not in ("I", "F") and not image.mode.startswith("I;16"):
            return image.convert("RGB").copy()
        image = np.asarray(image)
    pixels = np.asarray(image)
    if pixels.ndim not in (2, 3) or min(pixels.shape[:2], default=0) < 1:
        raise ValueError("Expected a nonempty grayscale, RGB, or RGBA image")
    if pixels.ndim == 3 and pixels.shape[2] not in (1, 3, 4):
        raise ValueError("Expected one, three, or four channels")
    if pixels.dtype.kind == "f":
        if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 255:
            raise ValueError("Floating pixels must be finite and in [0, 1] or [0, 255]")
        scale = 255.0 if pixels.max() <= 1 else 1.0
        pixels = np.rint(pixels.astype(np.float64) * scale).astype(np.uint8)
    elif pixels.dtype.kind == "u":
        scale = 255.0 / np.iinfo(pixels.dtype).max
        pixels = np.rint(pixels.astype(np.float64) * scale).astype(np.uint8)
    else:
        raise ValueError("Pixels must have unsigned integer or floating dtype")
    if pixels.ndim == 3 and pixels.shape[2] == 1:
        pixels = pixels[:, :, 0]
    converted = Image.fromarray(pixels)
    if converted.mode == "RGBA":
        return Image.alpha_composite(Image.new("RGBA", converted.size, "white"), converted).convert("RGB")
    return converted.convert("RGB")


def _gray(pixels):
    return cv2.cvtColor(np.ascontiguousarray(pixels), cv2.COLOR_RGB2GRAY)


def _odd(value, limit):
    maximum = max(1, int(limit) // 2 * 2 - 1)
    return min(maximum, max(1, int(math.ceil(value)) // 2 * 2 + 1))


def _ink(gray):
    kernel_size = _odd(max(9, min(gray.shape) * 0.025), min(gray.shape))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))
    contrast = background.astype(np.int16) - gray.astype(np.int16)
    return ((contrast >= 18) | (gray <= 110)).astype(np.uint8)


def _components(mask):
    _, labels, stats, centres = cv2.connectedComponentsWithStats(np.ascontiguousarray(mask), connectivity=8)
    return labels, stats[1:], centres[1:]


def _text_scale(gray):
    _, stats, _ = _components(_ink(gray))
    heights = [int(height) for _, _, width, height, area in stats
               if area >= 5 and 3 <= height <= gray.shape[0] * 0.12
               and width <= gray.shape[1] * 0.3 and width / height < 12]
    return float(np.median(heights)) if heights else max(3.0, min(gray.shape) * 0.008)


def _border_crop(pixels):
    height, width = pixels.shape[:2]
    dark = (pixels.max(axis=2) <= 36).astype(np.uint8)
    _, labels, _, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    candidates = {}
    reasons = {}
    for side, edge, axis_size in (
            ("left", labels[:, 0], width), ("right", labels[:, -1], width),
            ("top", labels[0], height), ("bottom", labels[-1], height)):
        edge_labels = np.unique(edge[edge != 0])
        if not len(edge_labels):
            reasons[side] = "no_near_black_edge_band"
            continue
        connected = np.isin(labels, edge_labels)
        profile = connected.mean(axis=0 if side in ("left", "right") else 1)
        if side in ("right", "bottom"):
            profile = profile[::-1]
        sparse = np.flatnonzero(profile < 0.92)
        depth = int(sparse[0]) if len(sparse) else axis_size
        minimum = max(2, int(math.ceil(axis_size * 0.003)))
        if depth < minimum:
            reasons[side] = "not_a_substantial_band"
        elif depth >= axis_size * 0.15:
            reasons[side] = "band_reaches_15_percent_limit"
        else:
            candidates[side] = depth
            reasons[side] = "candidate_edge_connected_band"
    bounds = [candidates.get("left", 0), candidates.get("top", 0),
              width - candidates.get("right", 0), height - candidates.get("bottom", 0)]
    left, top, right, bottom = bounds
    guard = max(4, int(math.ceil(min(width, height) * 0.006)))
    ink = _ink(_gray(pixels))
    protected = []
    strips = {"left": ink[top:bottom, left:min(right, left + guard)],
              "right": ink[top:bottom, max(left, right - guard):right],
              "top": ink[top:min(bottom, top + guard), left:right],
              "bottom": ink[max(top, bottom - guard):bottom, left:right]}
    for side in candidates:
        if strips[side].any():
            protected.append(side)
            reasons[side] = "protected_ink_near_candidate_cut"
    accepted = bool(candidates) and not protected
    if not accepted:
        bounds = [0, 0, width, height]
    metadata = {"applied": accepted, "bbox": bounds, "candidate_depths": candidates,
                "side_reasons": reasons, "protected_sides": protected,
                "guard_pixels": guard, "max_fraction_exclusive": 0.15,
                "reason": "edge_band_only" if accepted else
                ("crop_rejected_due_to_nearby_ink" if protected else "no_safe_border")}
    return bounds, metadata


def _deskew_evidence(pixels):
    """Erase stroke pixels only in an evidence mask, never in the output raster."""
    height, width = pixels.shape[:2]
    scale = min(1.0, 1600.0 / max(height, width))
    sampled = cv2.resize(pixels, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA) if scale < 1 else pixels
    gray = _gray(sampled)
    sample_height, sample_width = gray.shape
    evidence = {"applied": False, "estimated_degrees": None, "confidence": 0.0,
                "confident": False, "reason": "insufficient_text_strokes",
                "angle_range_degrees": [-5.0, 5.0], "stroke_count": 0,
                "row_count": 0, "body_component_count": 0,
                "method": "hough_strokes_with_repeated_body_support"}
    if min(gray.shape) < 32:
        return evidence
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=max(12, int(sample_width * 0.018)),
                            minLineLength=max(24, int(sample_width * 0.05)),
                            maxLineGap=max(2, int(sample_width * 0.01)))
    if lines is None:
        return evidence
    angles, lengths, midpoints, strokes = [], [], [], []
    margin = max(3, int(min(gray.shape) * 0.02))
    for endpoints in np.asarray(lines).reshape(-1, 4):
        left, top, right, bottom = map(int, endpoints)
        if right < left:
            left, top, right, bottom = right, bottom, left, top
        horizontal = right - left
        if horizontal == 0 or horizontal >= sample_width * 0.85:
            continue
        angle = math.degrees(math.atan2(bottom - top, horizontal))
        if abs(angle) > 5 or min(left, right) < margin or max(left, right) >= sample_width - margin:
            continue
        if min(top, bottom) < margin or max(top, bottom) >= sample_height - margin:
            continue
        angles.append(angle)
        lengths.append(float(math.hypot(horizontal, bottom - top)))
        midpoints.append((top + bottom) / 2)
        strokes.append((left, top, right, bottom))
    evidence["stroke_count"] = len(strokes)
    if len(strokes) < 5:
        return evidence
    order = np.argsort(angles)
    sorted_angles = np.asarray(angles)[order]
    sorted_weights = np.asarray(lengths)[order]
    middle = int(np.searchsorted(np.cumsum(sorted_weights), sorted_weights.sum() / 2))
    angle = float(sorted_angles[middle])
    residuals = np.abs(np.asarray(angles) - angle)
    inliers = residuals <= 0.4
    support = float(np.asarray(lengths)[inliers].sum() / sum(lengths))
    rows = []
    row_gap = max(5.0, sample_height * 0.012)
    for midpoint in sorted(np.asarray(midpoints)[inliers].tolist()):
        if not rows or midpoint - rows[-1] >= row_gap:
            rows.append(midpoint)
    body_mask = _ink(gray)
    for stroke, inlier in zip(strokes, inliers):
        if inlier:
            cv2.line(body_mask, stroke[:2], stroke[2:], 0, max(3, int(min(gray.shape) * 0.004)))
    _, stats, centres = _components(body_mask)
    body = [(stat, centre) for stat, centre in zip(stats, centres)
            if stat[4] >= 5 and 3 <= stat[3] <= sample_height * 0.08
            and stat[2] / stat[3] < 8 and margin <= centre[0] < sample_width - margin
            and margin <= centre[1] < sample_height - margin]
    supported_rows = int(sum(sum(abs(centre[1] - row) <= max(12, stat[3] * 2)
                                 for stat, centre in body) >= 4 for row in rows))
    evidence.update(estimated_degrees=angle, confidence=support, row_count=len(rows),
                    body_component_count=len(body), supported_rows=supported_rows)
    if support < 0.75:
        evidence["reason"] = "inconsistent_stroke_angles"
    elif len(rows) < 3 or max(rows) - min(rows) < sample_height * 0.10:
        evidence["reason"] = "insufficient_independent_rows"
    elif len(body) < 12 or supported_rows < 3:
        evidence["reason"] = "no_repeated_glyph_body_support"
    else:
        evidence["confident"] = True
        evidence["reason"] = "already_level" if abs(angle) < 0.15 else "supported_text_strokes"
    return evidence


def _rotate_expanded(pixels, angle):
    height, width = pixels.shape[:2]
    affine = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1.0)
    corners = np.array([[-0.5, -0.5, 1], [width - 0.5, -0.5, 1],
                        [width - 0.5, height - 0.5, 1], [-0.5, height - 0.5, 1]])
    rotated = corners @ affine.T
    minimum, maximum = rotated.min(axis=0), rotated.max(axis=0)
    output_width, output_height = np.ceil(maximum - minimum).astype(int)
    affine[:, 2] += np.array([-0.5, -0.5]) - minimum
    output = cv2.warpAffine(pixels, affine, (int(output_width), int(output_height)),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))
    return output, np.vstack((affine, [0.0, 0.0, 1.0]))


def _normalize_background(pixels):
    gray = _gray(pixels)
    text_height = _text_scale(gray)
    kernel_size = _odd(max(9, text_height * 3), min(gray.shape))
    median_size = _odd(max(3, text_height * 0.35), min(gray.shape))
    background = cv2.morphologyEx(pixels, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))
    if median_size > 1:
        background = cv2.medianBlur(background, median_size)
    contrast = _gray(background).astype(np.float32) - gray.astype(np.float32)
    protection = np.clip(contrast / 20.0, 0.0, 1.0)
    protection[gray <= 110] = 1.0
    addition = 0.65 * (255.0 - background.astype(np.float32)) * (1.0 - protection[:, :, None])
    normalized = np.rint(np.clip(pixels.astype(np.float32) + addition, 0, 255)).astype(np.uint8)
    return normalized, {"applied": not np.array_equal(pixels, normalized),
                        "method": "morphological_close_median_weighted_flattening",
                        "estimated_text_height": text_height, "kernel_size": kernel_size,
                        "median_size": median_size, "strength": 0.65,
                        "foreground_protection_contrast": 20, "components_removed": 0}


def prepare_page(image, *, mode="none", deskew=False, crop_border=False):
    """Return an independent RGB PIL image and JSON-serializable evidence.

    ``none`` bypasses all cleanup, even requested flags. ``conservative`` only
    attempts the explicitly requested crop/deskew; ``normalized`` additionally
    flattens background. Compare on development data before enabling either.
    NumPy unsigned arrays use their dtype range; floats use [0,1] or [0,255].
    """
    if mode not in ("none", "conservative", "normalized"):
        raise ValueError("mode must be none, conservative, or normalized")
    if not isinstance(deskew, bool) or not isinstance(crop_border, bool):
        raise ValueError("deskew and crop_border must be booleans")
    source = _rgb(image)
    pixels = np.array(source, copy=True)
    transform = np.eye(3)
    metadata = {"schema": "khojki-page-cleanup-v1", "mode": mode,
                "original_size": list(source.size), "reading_direction": "ltr",
                "coordinate_convention": "pixel_centres_x_right_y_down",
                "operation_order": ["crop", "deskew", "background"],
                "crop": {"applied": False, "reason": "mode_none" if mode == "none" else "not_requested"},
                "deskew": {"applied": False, "confident": False, "confidence": 0.0,
                           "reason": "mode_none" if mode == "none" else "not_requested"},
                "background": {"applied": False, "reason": "not_requested"},
                "limitations": ["heuristic_not_learned", "no_script_or_English_classification",
                                "ornament_and_faint_ink_can_be_ambiguous",
                                "requires_development_comparison_before_default_use"]}
    if mode != "none" and crop_border:
        bounds, metadata["crop"] = _border_crop(pixels)
        left, top, right, bottom = bounds
        pixels = pixels[top:bottom, left:right].copy()
        transform[:2, 2] = [-left, -top]
    if mode != "none" and deskew:
        evidence = _deskew_evidence(pixels)
        metadata["deskew"] = evidence
        if evidence["confident"] and evidence["reason"] != "already_level":
            pixels, rotation = _rotate_expanded(pixels, evidence["estimated_degrees"])
            transform = rotation @ transform
            evidence["applied"] = True
            evidence["applied_opencv_degrees"] = evidence["estimated_degrees"]
            evidence["fill_rgb"] = [255, 255, 255]
    if mode == "normalized":
        pixels, metadata["background"] = _normalize_background(pixels)
    metadata.update(prepared_size=[int(pixels.shape[1]), int(pixels.shape[0])],
                    from_original=transform.tolist(), to_original=np.linalg.inv(transform).tolist(),
                    changed=not (pixels.shape == np.asarray(source).shape and
                                 np.array_equal(pixels, np.asarray(source))))
    return Image.fromarray(pixels), metadata


def crop_line_region(image, linegeometry, *, padding=0):
    """Crop full-raster geometry without masking pixels or changing its input.

    Accept an xyxy bbox, or a mapping/object with bbox, boundary/polygon or
    baseline. Baseline-only bounds receive a conservative height allowance;
    they cannot recover a segmenter's missing polygon exactly.
    """
    source = _rgb(image)
    if not isinstance(padding, (int, float)) or not math.isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and nonnegative")
    if isinstance(linegeometry, Mapping):
        geometry = linegeometry
    elif hasattr(linegeometry, "baseline") or hasattr(linegeometry, "boundary"):
        geometry = {key: getattr(linegeometry, key, None) for key in ("bbox", "boundary", "baseline")}
    else:
        geometry = {"bbox": linegeometry}
    baseline_only = False
    if geometry.get("bbox") is not None:
        bounds = np.asarray(geometry["bbox"], dtype=np.float64)
        if bounds.shape != (4,):
            raise ValueError("bbox must be [left, top, right, bottom]")
    else:
        points = geometry.get("boundary")
        if points is None or len(points) == 0:
            points = geometry.get("polygon")
        if points is None or len(points) == 0:
            points = geometry.get("baseline")
            baseline_only = True
        points = np.asarray(points, dtype=np.float64)
        minimum_points = 2 if baseline_only else 3
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum_points:
            raise ValueError("Expected a polygon or baseline geometry")
        bounds = np.array([*points.min(axis=0), *points.max(axis=0)])
    if not np.isfinite(bounds).all():
        raise ValueError("Geometry must be finite")
    if bounds[2] <= bounds[0] or bounds[3] < bounds[1] or (bounds[3] == bounds[1] and not baseline_only):
        raise ValueError("Geometry must have positive extent")
    horizontal_pad = float(padding)
    vertical_pad = horizontal_pad + (max(4, source.height * 0.008) if baseline_only else 0)
    requested = [math.floor(bounds[0] - horizontal_pad), math.floor(bounds[1] - vertical_pad),
                 math.ceil(bounds[2] + horizontal_pad), math.ceil(bounds[3] + vertical_pad)]
    bounded = [max(0, requested[0]), max(0, requested[1]),
               min(source.width, requested[2]), min(source.height, requested[3])]
    if bounded[2] <= bounded[0] or bounded[3] <= bounded[1]:
        raise ValueError("Geometry does not overlap the raster")
    return source.crop(tuple(bounded)), {"bbox": bounded, "geometry_clipped": bounded != requested,
                                        "baseline_only": baseline_only,
                                        "to_page": [[1.0, 0.0, float(bounded[0])],
                                                    [0.0, 1.0, float(bounded[1])], [0.0, 0.0, 1.0]]}


def is_likely_text_region(image, linegeometry):
    """Return advisory evidence, never a script label or an automatic removal.

    Only a single near-solid, page-spanning thin rule with no other ink is a
    strict artifact candidate. Dot groups and specks remain reviewable.
    """
    source = _rgb(image)
    crop, geometry = crop_line_region(source, linegeometry)
    pixels = np.asarray(crop)
    mask = _ink(_gray(pixels))
    _, stats, centres = _components(mask)
    crop_height, crop_width = mask.shape
    body = [(stat, centre) for stat, centre in zip(stats, centres)
            if stat[4] >= 5 and stat[3] >= 5 and stat[2] / stat[3] < 12]
    text_height = float(np.median([stat[3] for stat, _ in body])) if body else 4.0
    small = [(stat, centre) for stat, centre in zip(stats, centres)
             if stat[4] <= max(16, text_height ** 2 * 0.25)
             and max(stat[2], stat[3]) <= max(4, text_height * 0.6)]
    groups = 0
    aligned = sorted(body, key=lambda item: item[1][0])
    gaps = []
    for (previous, previous_centre), (current, current_centre) in zip(aligned, aligned[1:]):
        gap = float(current[0] - previous[0] - previous[2])
        if 0 <= gap <= text_height * 4 and abs(current_centre[1] - previous_centre[1]) <= text_height:
            groups += 1
            gaps.append(gap)
    dot_group = False
    if len(small) >= 3:
        dot_centres = np.asarray([centre for _, centre in small])
        dot_centres = dot_centres[np.argsort(dot_centres[:, 0])]
        dot_gaps = np.diff(dot_centres[:, 0])
        dot_group = bool(dot_gaps.min() > 0 and np.std(dot_gaps) / np.mean(dot_gaps) <= 0.5
                         and np.ptp(dot_centres[:, 1]) <= max(4, text_height))
    strict_frame = False
    if len(stats) == 1 and not geometry["geometry_clipped"] and not geometry["baseline_only"]:
        _, _, width, height, area = map(int, stats[0])
        horizontal_rule = (width >= source.width * 0.60 and width >= crop_width * 0.85
                           and width / max(height, 1) >= 25 and height <= max(3, crop_height * 0.035))
        vertical_rule = (height >= source.height * 0.60 and height >= crop_height * 0.85
                         and height / max(width, 1) >= 25 and width <= max(3, crop_width * 0.035))
        strict_frame = bool((horizontal_rule or vertical_rule) and area / (width * height) >= 0.65)
    score = min(1.0, len(body) / 6 * 0.45 + groups / 4 * 0.45 + (0.10 if dot_group and body else 0))
    if strict_frame:
        decision, score = "artifact_candidate", 0.0
    elif len(body) >= 3 and groups >= 2:
        decision = "likely_text"
    else:
        decision = "ambiguous"
    flags = []
    if dot_group:
        flags.append("dot_group_preserve_for_review")
    if len(stats) and not body:
        flags.append("isolated_or_small_ink_not_proof_of_artifact")
    if geometry["baseline_only"]:
        flags.append("baseline_bounds_are_approximate")
    if geometry["geometry_clipped"]:
        flags.append("geometry_clipped")
    return {"schema": "khojki-region-evidence-v1", **geometry, "decision": decision,
            "text_score": float(score), "strict_artifact": strict_frame,
            "component_count": len(stats), "glyph_body_count": len(body),
            "repeated_glyph_pairs": groups, "small_component_count": len(small),
            "dot_group_possible": dot_group, "ink_fraction": float(mask.mean()),
            "uncertainty_flags": flags, "script_label": None, "reading_direction": "ltr",
            "automatic_removal": False,
            "limitations": ["generic_geometry_not_a_learned_model",
                            "English_and_Khojki_are_not_distinguished",
                            "connected_headlines_and_ornaments_may_be_ambiguous"]}


def assess_text_presence(image, *, proposals=None):
    """Confirm only a near-white raster with an explicit empty proposal list.

    Missing proposals, brown paper, faint ink, or blemishes remain reviewable.
    This is raster evidence, not proof that a page has no target-script text.
    """
    source = _rgb(image)
    pixels = np.asarray(source)
    all_near_white = bool((pixels >= 254).all())
    count = None if proposals is None else len(proposals)
    confirmed = all_near_white and count == 0
    return {"schema": "khojki-text-presence-v1", "confirmed_empty": confirmed,
            "status": "confirmed_empty" if confirmed else "review",
            "all_pixels_near_white": all_near_white, "near_white_channel_threshold": 254,
            "proposal_count": count, "minimum_channel_value": int(pixels.min()),
            "ink_fraction_diagnostic_only": float(_ink(_gray(pixels)).mean()),
            "reason": "near_white_and_explicit_no_proposals" if confirmed else "insufficient_empty_evidence",
            "no_target_script_claim": True}