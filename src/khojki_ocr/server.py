"""Loopback-only upload service exposing reviewable, raw model predictions."""

import argparse
import copy
import io
import json
import math
import mimetypes
import os
import re
import threading
import time
import uuid
from collections import deque
from contextlib import closing
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

from khojki_ocr import pipeline
from khojki_ocr.core import invalid_codepoints, validate_text


WORKSPACE = Path(__file__).resolve().parents[2]
TERMINAL = frozenset({"review", "blank", "failed", "cancelled"})
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
INFERENCE_LOCK = threading.Lock()
PDF_LOCK = threading.Lock()
MAX_LINE_CHARACTERS = 16384
MAX_LINES = 2048
MAX_JOB_LINES = 4096
MAX_JOB_CHARACTERS = 131072
MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_PREVIEW_BYTES = 8 * 1024 * 1024


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class Cancelled(BaseException):
    pass


@dataclass(frozen=True)
class Limits:
    max_upload_bytes: int = 20 * 1024 * 1024
    max_pages: int = 20
    max_dimension: int = 10000
    max_pixels: int = 40_000_000
    max_jobs: int = 20
    max_storage_bytes: int = 1024 * 1024 * 1024
    preview_dimension: int = 1200
    dpi: float = 180

    def __post_init__(self):
        ceilings = {"max_upload_bytes": 20 * 1024 * 1024, "max_pages": 20,
                    "max_dimension": 10000, "max_pixels": 40_000_000,
                    "max_jobs": 20, "max_storage_bytes": 1024 * 1024 * 1024,
                    "preview_dimension": 1200}
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be an integer between 1 and {ceiling}")
        if not math.isfinite(self.dpi) or not 1 <= self.dpi <= 600:
            raise ValueError("DPI must be between 1 and 600")


def validate_filename(filename: str) -> str:
    if (not filename or len(filename) > 240 or filename in {".", ".."}
            or any(character in filename for character in "/\\%")
            or any(ord(character) < 32 or ord(character) == 127 for character in filename)):
        raise APIError(400, "invalid_filename", "X-Filename must be a plain filename")
    return filename


def validate_dimensions(width: float, height: float, limits: Limits) -> list[int]:
    if (not math.isfinite(width) or not math.isfinite(height)
            or width <= 0 or height <= 0):
        raise APIError(422, "invalid_dimensions", "Invalid page dimensions")
    size = [math.ceil(width), math.ceil(height)]
    if max(size) > limits.max_dimension or size[0] * size[1] > limits.max_pixels:
        raise APIError(413, "dimensions_exceeded", "Page dimensions exceed the configured limit")
    return size


def inspect_upload(payload: bytes, filename: str, limits: Limits) -> dict:
    from PIL import Image, UnidentifiedImageError

    validate_filename(filename)
    if not payload:
        raise APIError(400, "empty_upload", "The upload is empty")
    if len(payload) > limits.max_upload_bytes:
        raise APIError(413, "upload_too_large", "Upload exceeds the configured byte limit")
    if payload.startswith(b"%PDF-"):
        import pypdfium2 as pdfium

        try:
            with PDF_LOCK, pdfium.PdfDocument(payload) as document:
                count = len(document)
                if count < 1:
                    raise APIError(422, "empty_document", "The PDF has no pages")
                if count > limits.max_pages:
                    raise APIError(413, "pages_exceeded", "PDF exceeds the configured page limit")
                sizes = []
                for index in range(count):
                    page = document[index]
                    try:
                        width, height = page.get_size()
                        sizes.append(validate_dimensions(width * limits.dpi / 72,
                                                         height * limits.dpi / 72, limits))
                    finally:
                        page.close()
            return {"kind": "pdf", "extension": ".pdf", "page_count": count, "sizes": sizes}
        except APIError:
            raise
        except Exception:
            raise APIError(422, "invalid_pdf", "PDF could not be parsed; encrypted PDFs are unsupported") from None
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise APIError(415, "unsupported_file", "Only PDF, PNG and JPEG content is supported")
            if getattr(image, "n_frames", 1) != 1:
                raise APIError(422, "multiple_frames", "Upload a single-frame image")
            size = validate_dimensions(image.width, image.height, limits)
            extension = ".png" if image.format == "PNG" else ".jpg"
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
        return {"kind": "image", "extension": extension, "page_count": 1, "sizes": [size]}
    except APIError:
        raise
    except Image.DecompressionBombError:
        raise APIError(413, "dimensions_exceeded", "Image dimensions exceed the configured limit") from None
    except UnidentifiedImageError:
        raise APIError(415, "unsupported_file", "Only PDF, PNG and JPEG content is supported") from None
    except Exception:
        raise APIError(422, "invalid_image", "Image content is damaged or incomplete") from None


class Job:
    def __init__(self, identifier: str, directory: Path, filename: str, inspection: dict):
        self.id = identifier
        self.directory = directory
        self.input_path = directory / ("input" + inspection["extension"])
        self.inspection = inspection
        self.cancelled = threading.Event()
        self.condition = threading.Condition()
        self.preview_lock = threading.Lock()
        self.events = deque(maxlen=64)
        self.sequence = 0
        self.characters = 0
        self.created = time.monotonic()
        self.finished = None
        self.state = {
            "id": identifier, "job_id": identifier, "filename": filename,
            "status": "queued", "stage": "queued", "page": 0,
            "total": inspection["page_count"], "page_count": inspection["page_count"],
            "text": "", "review_required": True, "provenance": "raw_model_prediction",
            "certified": False, "references_enabled": False,
            "transcript_url": f"/api/jobs/{identifier}/transcript.txt",
            "pages": [{"page": number, "status": "pending", "review_required": True,
                       "lines": [], "raw_hypothesis": "", "size": size,
                       "preview_url": f"/api/jobs/{identifier}/pages/{number}.png"}
                      for number, size in enumerate(inspection["sizes"], 1)],
            "errors": [], "stage_seconds": {}, "elapsed_seconds": 0.0,
        }
        self.publish(stage="queued", status="queued")

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise Cancelled()

    def snapshot(self) -> dict:
        with self.condition:
            result = copy.deepcopy(self.state)
            measured_at = self.finished if self.finished is not None else time.monotonic()
            result["elapsed_seconds"] = round(measured_at - self.created, 3)
            result["event_id"] = self.sequence
            return result

    def publish(self, *, stage: str, status: str | None = None, **fields):
        with self.condition:
            self.state["stage"] = stage
            if status is not None:
                self.state["status"] = status
            if "page" in fields:
                self.state["page"] = fields["page"]
            measured_at = self.finished if self.finished is not None else time.monotonic()
            if self.state["status"] in TERMINAL:
                self.finished = measured_at
            self.state["elapsed_seconds"] = round(measured_at - self.created, 3)
            self.sequence += 1
            event = {"stage": stage, "status": self.state["status"],
                     "page": self.state["page"], "total": self.state["total"],
                     "review_required": self.state["review_required"],
                     "elapsed_seconds": self.state["elapsed_seconds"], **fields}
            self.events.append((self.sequence, event))
            self.condition.notify_all()

    def add_time(self, stage: str, started: float):
        with self.condition:
            times = self.state["stage_seconds"]
            times[stage] = round(times.get(stage, 0.0) + time.monotonic() - started, 3)

    def add_line(self, page_number: int, record):
        with self.condition:
            if self.characters + len(record.prediction) > MAX_JOB_CHARACTERS:
                raise ValueError("Accumulated prediction exceeds output limits")
        hypothesis = pipeline.line_hypothesis(record)
        hypothesis.pop("unserializable_confidences", None)
        hypothesis["line_id"] = str(record.id)
        if getattr(record, "recognition_error", None):
            hypothesis["recognition_error"] = record.recognition_error
            hypothesis["uncertainty_flags"].append("line_recognition_failed")
        with self.condition:
            self.characters += len(record.prediction)
            page = self.state["pages"][page_number - 1]
            page["lines"].append(hypothesis)
            page["raw_hypothesis"] = "\n".join(line["raw_hypothesis"] for line in page["lines"])
            self.state["text"] = "\n\n".join(page["raw_hypothesis"] for page in self.state["pages"]
                                               if page["raw_hypothesis"])
            line_number = len(page["lines"])
            total_lines = page.get("line_count", 0)
        self.publish(stage="recognition", page=page_number, text=record.prediction,
                     lines=[hypothesis], line=line_number, line_total=total_lines)


class JobService:
    def __init__(self, workspace: Path = WORKSPACE, *, model: Path | None = None,
                 limits: Limits | None = None, pdf_caption_texts=()):
        self.workspace = Path(workspace).resolve()
        self.storage = self.workspace / "runs" / "web-jobs"
        self.web_root = self.workspace / "web"
        self.model = Path(model) if model is not None else self.workspace / "models/khojki-printed.safetensors"
        self.limits = limits or Limits()
        pipeline._validate_caption_texts(pdf_caption_texts)
        self.pdf_caption_texts = tuple(pdf_caption_texts)
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}
        self.threads: list[threading.Thread] = []
        self.closing = threading.Event()
        self.models = None
        self.model_error = False

    def config(self) -> dict:
        return {"model_available": self.models is not None or (not self.model_error and self.model.is_file()),
                "model_loaded": self.models is not None, "model_load_failed": self.model_error,
                "model_name": self.model.name, "max_upload_bytes": self.limits.max_upload_bytes,
                "max_pages": self.limits.max_pages, "max_dimension": self.limits.max_dimension,
                "max_pixels": self.limits.max_pixels, "max_jobs": self.limits.max_jobs,
                "max_storage_bytes": self.limits.max_storage_bytes,
                "preview_dimension": self.limits.preview_dimension, "dpi": self.limits.dpi,
                "max_lines_per_page": MAX_LINES, "max_lines_per_job": MAX_JOB_LINES,
                "max_prediction_characters": MAX_JOB_CHARACTERS,
                "accepted_types": ["application/pdf", "image/png", "image/jpeg"],
                "upload_content_type": "application/octet-stream", "review_required": True,
                "references_enabled": False, "pdf_caption_texts": list(self.pdf_caption_texts),
                "caption_policy": "explicit_exact_text_only", "automatic_english_filter": False}

    def _usage(self) -> tuple[int, int]:
        for path in (self.workspace / "runs", self.storage):
            if path.is_symlink():
                raise APIError(507, "unsafe_storage", "Job storage cannot be a symbolic link")
        if not self.storage.exists():
            return 0, 0
        if not self.storage.is_dir():
            raise APIError(507, "unsafe_storage", "Job storage is unavailable")
        count = sum(1 for entry in self.storage.iterdir() if entry.is_dir())
        used = 0
        for root, directories, files in os.walk(self.storage, followlinks=False):
            for name in directories + files:
                entry = Path(root) / name
                if entry.is_symlink():
                    raise APIError(507, "unsafe_storage", "Job storage contains symbolic links")
            used += sum((Path(root) / name).stat().st_size for name in files)
        return count, used

    def preflight(self, length: int):
        if self.closing.is_set():
            raise APIError(503, "shutting_down", "The local server is stopping")
        if not self.config()["model_available"]:
            raise APIError(503, "model_unavailable", "The configured local model is unavailable")
        if not 0 < length <= self.limits.max_upload_bytes:
            raise APIError(413 if length else 400, "upload_too_large" if length else "empty_upload",
                           "Upload size is outside the configured limits")
        with self.lock:
            count, used = self._usage()
            if count >= self.limits.max_jobs:
                raise APIError(429, "job_limit", "Job retention limit reached; no existing files were deleted")
            if used + length + MAX_PREVIEW_BYTES + MAX_REPORT_BYTES > self.limits.max_storage_bytes:
                raise APIError(507, "storage_limit", "Private job storage is full; no existing files were deleted")

    def create_job(self, payload: bytes, filename: str, *, start: bool = True) -> Job:
        self.preflight(len(payload))
        started = time.monotonic()
        inspection = inspect_upload(payload, filename, self.limits)
        with self.lock:
            self.preflight(len(payload))
            identifier = uuid.uuid4().hex
            directory = self.storage / identifier
            directory.mkdir(mode=0o700, parents=True, exist_ok=False)
            job = Job(identifier, directory, filename, inspection)
            with job.input_path.open("xb") as stream:
                stream.write(payload)
            self.jobs[identifier] = job
            job.add_time("validation", started)
            if start:
                thread = threading.Thread(target=self.run_job, args=(job,), daemon=True,
                                          name=f"ocr-{identifier[:8]}")
                self.threads.append(thread)
                thread.start()
            return job

    def get_job(self, identifier: str) -> Job:
        if not JOB_ID.fullmatch(identifier):
            raise APIError(400, "invalid_job_id", "Job IDs must be 32 lowercase hexadecimal characters")
        with self.lock:
            job = self.jobs.get(identifier)
        if job is None:
            raise APIError(404, "job_not_found", "Job does not exist in this server session")
        return job

    def cancel(self, job: Job) -> dict:
        with job.condition:
            if job.state["status"] not in TERMINAL:
                job.cancelled.set()
                job.publish(stage="cancelling", status="cancelling")
        return job.snapshot()

    def _load_models(self):
        if self.models is None:
            if self.model_error:
                raise APIError(503, "model_load_failed", "The local model could not be loaded")
            try:
                os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
                import torch
                from kraken.tasks import SegmentationTaskModel

                torch.set_num_threads(2)
                self.models = (SegmentationTaskModel.load_model(),
                               pipeline.load_recognition_model(self.model, require_contract=True))
            except Exception:
                self.model_error = True
                raise APIError(503, "model_load_failed", "The local model could not be loaded") from None
        return self.models

    def _write(self, path: Path, payload: bytes):
        with self.lock:
            _, used = self._usage()
            if used + len(payload) > self.limits.max_storage_bytes:
                raise APIError(507, "storage_limit", "Private job storage is full; no existing files were deleted")
            with path.open("xb") as stream:
                stream.write(payload)

    def _segmenter(self, job: Job, number: int, model):
        def segment(image):
            job.check_cancelled()
            job.publish(stage="segmentation", status="running", page=number)
            started = time.monotonic()
            try:
                result = pipeline.segment_page(image, model)
                with job.condition:
                    detected = sum(page.get("line_count", 0) for page in job.state["pages"])
                    if len(result.lines) > MAX_LINES or detected + len(result.lines) > MAX_JOB_LINES:
                        raise ValueError("Detected line limit exceeded")
                    job.state["pages"][number - 1]["line_count"] = len(result.lines)
                job.check_cancelled()
                return result
            finally:
                job.add_time("segmentation", started)
        return segment

    def _recognizer(self, job: Job, number: int, model):
        def recognize(image, segmentation):
            for line in segmentation.lines:
                job.check_cancelled()
                started = time.monotonic()
                records = None
                try:
                    records = iter(pipeline.recognize_page(image, replace(segmentation, lines=[line]), model))
                    received = False
                    for record in records:
                        job.check_cancelled()
                        if received:
                            raise ValueError("Unexpected extra recognition record")
                        received = True
                        if not isinstance(record.prediction, str) or len(record.prediction) > MAX_LINE_CHARACTERS:
                            raise ValueError("Prediction exceeds output limits")
                        scores = [] if record.confidences is None else list(islice(record.confidences,
                                                                                  MAX_LINE_CHARACTERS + 1))
                        if len(scores) > MAX_LINE_CHARACTERS:
                            raise ValueError("Confidence output exceeds output limits")
                        prediction = SimpleNamespace(id=line.id, prediction=record.prediction,
                                                     confidences=scores)
                        job.add_line(number, prediction)
                        yield prediction
                    if not received:
                        raise ValueError("No recognition record returned")
                except Exception as error:
                    record = SimpleNamespace(id=line.id, prediction="", confidences=[],
                                             recognition_error=f"{type(error).__name__}: line recognition failed")
                    job.check_cancelled()
                    job.add_line(number, record)
                    yield record
                finally:
                    try:
                        if records is not None and hasattr(records, "close"):
                            records.close()
                    except Exception:
                        with job.condition:
                            job.state["errors"].append({"code": "line_cleanup_failed",
                                                       "message": "Line processing requires review"})
                    job.add_time("recognition", started)
        return recognize

    def run_job(self, job: Job):
        acquired = False
        iterator = None
        outcome = "failed"
        try:
            while not acquired:
                job.check_cancelled()
                acquired = INFERENCE_LOCK.acquire(timeout=0.1)
            job.check_cancelled()
            job.publish(stage="model_loading", status="running")
            started = time.monotonic()
            segmentation_model, recognition_model = self._load_models()
            job.add_time("model_loading", started)
            iterator = pipeline.iter_pages(job.input_path, dpi=self.limits.dpi,
                                           pdf_caption_texts=self.pdf_caption_texts)
            for expected in range(1, job.inspection["page_count"] + 1):
                job.check_cancelled()
                job.publish(stage="rendering", status="running", page=expected)
                started = time.monotonic()
                with PDF_LOCK:
                    number, image, metadata = next(iterator)
                job.add_time("rendering", started)
                validate_dimensions(image.width, image.height, self.limits)
                with job.condition:
                    job.state["pages"][number - 1]["status"] = "running"
                started = time.monotonic()
                result = pipeline.process_page(
                    image, number, segmenter=self._segmenter(job, number, segmentation_model),
                    recognizer=self._recognizer(job, number, recognition_model),
                    manifest=None, metadata=metadata, output_dir=None, line_crops=None)
                job.add_time("page_processing", started)
                job.check_cancelled()
                result.pop("transcript_path", None)
                result["errors"] = ["Page processing requires review" for _ in result["errors"]]
                for hypothesis in result["lines"]:
                    hypothesis.pop("unserializable_confidences", None)
                captions = result.get("source", {}).get("captions", {})
                if "error" in captions:
                    captions["error"] = "Caption inspection failed; pixels were preserved"
                with job.condition:
                    result["preview_url"] = job.state["pages"][number - 1]["preview_url"]
                    job.state["pages"][number - 1] = result
                    job.state["text"] = "\n\n".join(page.get("raw_hypothesis") or ""
                                                       for page in job.state["pages"]
                                                       if page.get("raw_hypothesis"))
                job.publish(stage="page_complete", page=number, page_status=result["status"],
                            review_required=result["review_required"])
            with job.condition:
                blank = all(page["status"] == "blank" for page in job.state["pages"])
                job.state["review_required"] = not blank
            outcome = "blank" if blank else "review"
        except Cancelled:
            outcome = "cancelled"
        except APIError as error:
            with job.condition:
                job.state["errors"].append({"code": error.code, "message": error.message})
        except Exception:
            with job.condition:
                job.state["errors"].append({"code": "processing_failed", "message": "Local page processing failed"})
        finally:
            try:
                if iterator is not None:
                    with PDF_LOCK:
                        iterator.close()
            except Exception:
                with job.condition:
                    job.state["errors"].append({"code": "page_cleanup_failed",
                                               "message": "Page resources could not be closed normally"})
                outcome = "failed"
            finally:
                if acquired:
                    INFERENCE_LOCK.release()
            try:
                if job.cancelled.is_set() and outcome in {"review", "blank"}:
                    outcome = "cancelled"
                stage = "complete" if outcome in {"review", "blank"} else outcome
                report = job.snapshot()
                report.update(status=outcome, stage=stage, event_id=report["event_id"] + 1)
                payload = json.dumps(report, ensure_ascii=True, allow_nan=False).encode("utf-8")
                if len(payload) > MAX_REPORT_BYTES:
                    raise APIError(507, "output_limit", "Job metadata exceeds the output limit")
                self._write(job.directory / "report.json", payload)
            except (APIError, OSError, ValueError):
                with job.condition:
                    job.state["errors"].append({"code": "report_unavailable", "message": "Private report could not be saved"})
                outcome = "failed"
            job.publish(stage="complete" if outcome in {"review", "blank"} else outcome, status=outcome)

    def preview(self, job: Job, number: int) -> bytes:
        if not 1 <= number <= job.inspection["page_count"]:
            raise APIError(404, "page_not_found", "Page does not exist")
        with job.preview_lock:
            path = job.directory / f"page-{number}.png"
            if path.exists():
                if path.is_symlink() or path.stat().st_size > MAX_PREVIEW_BYTES:
                    raise APIError(507, "unsafe_storage", "Preview storage is unavailable")
                return path.read_bytes()
            size = job.inspection["sizes"][number - 1]
            dpi = self.limits.dpi * min(1, self.limits.preview_dimension / max(size))
            try:
                with PDF_LOCK, closing(pipeline.iter_pages(job.input_path, [number], dpi=dpi)) as iterator:
                    _, image, _ = next(iterator)
                    with image.copy() as preview:
                        preview.thumbnail((self.limits.preview_dimension, self.limits.preview_dimension))
                        stream = io.BytesIO()
                        preview.save(stream, format="PNG")
                        payload = stream.getvalue()
                if len(payload) > MAX_PREVIEW_BYTES:
                    raise APIError(507, "preview_limit", "Preview exceeds the output limit")
                self._write(path, payload)
                return payload
            except APIError:
                raise
            except Exception:
                raise APIError(422, "preview_failed", "Source preview could not be rendered") from None

    def transcript(self, job: Job) -> bytes:
        state = job.snapshot()
        if state["status"] not in {"review", "blank"}:
            raise APIError(409, "transcript_not_ready", "Transcript requires a completed job")
        text = state["text"]
        if not text.strip():
            raise APIError(404, "empty_transcript", "No nonempty model transcript is available")
        if invalid_codepoints(text):
            raise APIError(422, "invalid_transcript", "Raw output contains non-target Unicode; no characters were removed")
        validate_text(text)
        return text.encode("utf-8")

    def close(self):
        self.closing.set()
        with self.lock:
            for job in self.jobs.values():
                self.cancel(job)


class LocalHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, address: tuple[str, int], service: JobService):
        if address[0] != "127.0.0.1":
            raise ValueError("The server must bind to 127.0.0.1")
        self.service = service
        super().__init__(address, RequestHandler)

    def server_close(self):
        self.service.close()
        super().server_close()

    def handle_error(self, request, client_address):
        pass


class RequestHandler(BaseHTTPRequestHandler):
    server: LocalHTTPServer
    protocol_version = "HTTP/1.1"
    server_version = "KhojkiLocal/1"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, format, *args):
        pass

    def _local_request(self):
        hosts = self.headers.get_all("Host", [])
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if len(hosts) != 1 or hosts[0].lower() not in allowed:
            raise APIError(403, "invalid_host", "Only the configured loopback host is permitted")
        origins = self.headers.get_all("Origin", [])
        if origins:
            if len(origins) != 1 or origins[0].lower() != "http://" + hosts[0].lower():
                raise APIError(403, "invalid_origin", "Cross-origin requests are not permitted")
        if self.headers.get("Sec-Fetch-Site", "") == "cross-site":
            raise APIError(403, "invalid_origin", "Cross-site requests are not permitted")

    def _respond(self, status: int, payload: bytes, content_type: str, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Connection", "close")
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.close_connection = True
        self.wfile.write(payload)

    def _json(self, status: int, value: dict):
        self._respond(status, json.dumps(value, ensure_ascii=True, allow_nan=False).encode("utf-8"),
                      "application/json; charset=utf-8")

    def _error(self, error: APIError):
        self._json(error.status, {"error": {"code": error.code, "message": error.message}})

    def _path(self) -> str:
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc or parsed.fragment:
            raise APIError(400, "invalid_path", "Invalid request path")
        path = unquote(parsed.path)
        if "\\" in path or "\x00" in path or ".." in path.split("/"):
            raise APIError(400, "invalid_path", "Path traversal is not permitted")
        return path

    def _events(self, job: Job):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        state = job.snapshot()
        cursor = state["event_id"]
        self._event(cursor, {**state, "snapshot": True})
        while not self.server.service.closing.is_set():
            with job.condition:
                pending = [(sequence, copy.deepcopy(event)) for sequence, event in job.events if sequence > cursor]
                terminal = job.state["status"] in TERMINAL
                if not pending and not terminal:
                    job.condition.wait(timeout=15)
                    pending = [(sequence, copy.deepcopy(event)) for sequence, event in job.events if sequence > cursor]
                    terminal = job.state["status"] in TERMINAL
                missed = bool(pending and pending[0][0] > cursor + 1)
            if missed:
                state = job.snapshot()
                cursor = state["event_id"]
                self._event(cursor, {**state, "snapshot": True})
            else:
                for sequence, event in pending:
                    self._event(sequence, event)
                    cursor = sequence
            if terminal:
                break
            if not pending:
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()

    def _event(self, sequence: int, event: dict):
        payload = json.dumps(event, ensure_ascii=True, allow_nan=False)
        self.wfile.write(f"id: {sequence}\nevent: progress\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _get(self):
        self._local_request()
        path = self._path()
        service = self.server.service
        if path == "/api/config":
            self._json(200, service.config())
            return
        if path.startswith("/api/jobs/"):
            parts = path.split("/")
            job = service.get_job(parts[3])
            if len(parts) == 4:
                self._json(200, job.snapshot())
            elif len(parts) == 5 and parts[4] == "events":
                self._events(job)
            elif len(parts) == 5 and parts[4] == "transcript.txt":
                payload = service.transcript(job)
                self._respond(200, payload, "text/plain; charset=utf-8",
                              {"Content-Disposition": f'attachment; filename="{job.id}-draft.txt"',
                               "X-Review-Required": "true", "X-Transcript-Provenance": "raw-model-prediction"})
            elif len(parts) == 6 and parts[4] == "pages" and re.fullmatch(r"[1-9][0-9]{0,2}\.png", parts[5]):
                self._respond(200, service.preview(job, int(parts[5][:-4])), "image/png")
            else:
                raise APIError(404, "route_not_found", "Endpoint does not exist")
            return
        if path.startswith("/api/"):
            raise APIError(404, "route_not_found", "Endpoint does not exist")
        root = service.web_root.resolve()
        if service.web_root.is_symlink():
            raise APIError(404, "asset_not_found", "Web assets are unavailable")
        asset = (root / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
        allowed = {".html", ".js", ".css", ".png", ".jpg", ".jpeg", ".ico", ".svg", ".woff", ".woff2", ".ttf"}
        if (root not in asset.parents or not asset.is_file() or asset.suffix.lower() not in allowed
                or asset.stat().st_size > 20 * 1024 * 1024):
            raise APIError(404, "asset_not_found", "Web asset does not exist")
        content_type = mimetypes.guess_type(str(asset))[0] or "application/octet-stream"
        if asset.suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        self._respond(200, asset.read_bytes(), content_type)

    def do_GET(self):
        try:
            self._get()
        except APIError as error:
            self._error(error)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception:
            self._error(APIError(500, "internal_error", "The local request could not be completed"))

    def do_POST(self):
        try:
            self._local_request()
            path = self._path()
            service = self.server.service
            if self.headers.get_all("Transfer-Encoding", []):
                raise APIError(400, "unsupported_transfer", "Chunked uploads are not supported")
            if path == "/api/jobs":
                types = self.headers.get_all("Content-Type", [])
                if len(types) != 1 or types[0].split(";", 1)[0].strip().lower() != "application/octet-stream":
                    raise APIError(415, "invalid_content_type", "Use application/octet-stream")
                lengths = self.headers.get_all("Content-Length", [])
                if not lengths:
                    raise APIError(411, "length_required", "Content-Length is required")
                if len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,10}", lengths[0]):
                    raise APIError(400, "invalid_length", "Content-Length must be a nonnegative integer")
                names = self.headers.get_all("X-Filename", [])
                if len(names) != 1:
                    raise APIError(400, "invalid_filename", "One X-Filename header is required")
                filename = validate_filename(names[0])
                length = int(lengths[0])
                service.preflight(length)
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise APIError(400, "truncated_upload", "Upload ended before Content-Length bytes arrived")
                job = service.create_job(payload, filename)
                self._json(202, {"id": job.id, "job_id": job.id, "status": job.snapshot()["status"],
                                 "page_count": job.inspection["page_count"],
                                 "events_url": f"/api/jobs/{job.id}/events",
                                 "state_url": f"/api/jobs/{job.id}"})
                return
            parts = path.split("/")
            if len(parts) == 5 and parts[1:3] == ["api", "jobs"] and parts[4] == "cancel":
                state = service.cancel(service.get_job(parts[3]))
                self._json(200 if state["status"] in TERMINAL else 202, state)
                return
            raise APIError(404, "route_not_found", "Endpoint does not exist")
        except APIError as error:
            self._error(error)
        except TimeoutError:
            self._error(APIError(408, "upload_timeout", "Upload timed out"))
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            self._error(APIError(500, "internal_error", "The local request could not be completed"))

    def do_OPTIONS(self):
        self._error(APIError(405, "method_not_allowed", "Only same-origin GET and POST are supported"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Loopback-only Khojki OCR draft review service")
    parser.add_argument("--model", type=Path, default=WORKSPACE / "models/khojki-printed.safetensors")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-upload-mb", type=int, default=20)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--max-dimension", type=int, default=10000)
    parser.add_argument("--max-pixels", type=int, default=40_000_000)
    parser.add_argument("--max-jobs", type=int, default=20)
    parser.add_argument("--max-storage-mb", type=int, default=1024)
    parser.add_argument("--dpi", type=float, default=180)
    parser.add_argument("--pdf-caption-text", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        limits = Limits(max_upload_bytes=args.max_upload_mb * 1024 * 1024,
                        max_pages=args.max_pages, max_dimension=args.max_dimension,
                        max_pixels=args.max_pixels, max_jobs=args.max_jobs,
                        max_storage_bytes=args.max_storage_mb * 1024 * 1024, dpi=args.dpi)
        service = JobService(model=args.model, limits=limits, pdf_caption_texts=args.pdf_caption_text)
        with LocalHTTPServer(("127.0.0.1", args.port), service) as server:
            print(f"Khojki draft review: http://127.0.0.1:{args.port}", flush=True)
            try:
                server.serve_forever(poll_interval=0.2)
            except KeyboardInterrupt:
                pass
        return 0
    except (OSError, ValueError):
        print("Local server could not start; check the port, model and configured limits", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())