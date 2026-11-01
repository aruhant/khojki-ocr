import io
import json
import tempfile
import threading
import unittest
from dataclasses import dataclass
from http.client import HTTPConnection
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from khojki_ocr.server import (
    APIError, INFERENCE_LOCK, JobService, Limits, LocalHTTPServer, inspect_upload,
    validate_filename,
)


KA = "\U00011208"


def png_bytes(size=(64, 64), color="white"):
    with Image.new("RGB", size, color) as image:
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()


@dataclass
class Bounds:
    type: str
    lines: list
    regions: dict


def bounds(names=("first", "second")):
    return Bounds("baselines", [SimpleNamespace(id=name, baseline=[(4, 20), (50, 20)],
                  boundary=[(4, 4), (50, 4), (50, 24), (4, 24)]) for name in names], {})


class ServiceCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.model = self.workspace / "mock.safetensors"
        self.model.write_bytes(b"mock model; never loaded")
        self.service = JobService(self.workspace, model=self.model)
        self.addCleanup(self.service.close)
        self.payload = png_bytes()

    def job(self):
        return self.service.create_job(self.payload, "source.png", start=False)


class ServiceTests(ServiceCase):
    def test_elapsed_time_stops_at_each_terminal_state(self):
        for status in ("review", "blank", "failed", "cancelled"):
            with self.subTest(status=status):
                job = self.job()
                with patch("khojki_ocr.server.time.monotonic") as clock:
                    clock.return_value = job.created + 2
                    self.assertEqual(job.snapshot()["elapsed_seconds"], 2)
                    clock.return_value = job.created + 7
                    job.publish(stage="complete", status=status)
                    self.assertEqual(job.events[-1][1]["elapsed_seconds"], 7)
                    clock.return_value = job.created + 100
                    self.assertEqual(job.snapshot()["elapsed_seconds"], 7)
                    job.publish(stage="complete")
                    self.assertEqual(job.events[-1][1]["elapsed_seconds"], 7)
                    self.assertEqual(job.state["elapsed_seconds"], 7)

    def test_default_model_is_the_packaged_pair_not_a_private_training_run(self):
        service = JobService(self.workspace)
        self.addCleanup(service.close)
        self.assertEqual(service.model, self.workspace.resolve() / "models/khojki-printed.safetensors")
        self.assertFalse(service.config()["model_available"])

    def test_content_not_filename_selects_format(self):
        result = inspect_upload(self.payload, "untrusted.pdf", Limits())
        self.assertEqual(result["extension"], ".png")
        self.assertEqual(result["page_count"], 1)
        job = self.service.create_job(self.payload, "untrusted.jpg", start=False)
        self.assertEqual(job.input_path.name, "input.png")

    def test_jpeg_content_is_supported(self):
        with Image.new("RGB", (20, 20)) as image:
            stream = io.BytesIO()
            image.save(stream, format="JPEG")
        self.assertEqual(inspect_upload(stream.getvalue(), "upload.png", Limits())["extension"], ".jpg")

    def test_empty_unsupported_damaged_and_oversized_images(self):
        cases = [(b"", Limits(), 400), (b"not an image", Limits(), 415),
                 (self.payload[:-20], Limits(), 422),
                 (self.payload, Limits(max_upload_bytes=1), 413),
                 (png_bytes((100, 20)), Limits(max_dimension=50), 413),
                 (png_bytes((100, 100)), Limits(max_pixels=100), 413)]
        for payload, limits, expected in cases:
            with self.subTest(expected=expected, length=len(payload)):
                with self.assertRaises(APIError) as caught:
                    inspect_upload(payload, "upload.png", limits)
                self.assertEqual(caught.exception.status, expected)

    def test_bad_pdf_is_rejected_without_job_files(self):
        with self.assertRaises(APIError) as caught:
            self.service.create_job(b"%PDF-1.7\nbroken", "file.pdf", start=False)
        self.assertEqual(caught.exception.code, "invalid_pdf")
        self.assertFalse(self.service.storage.exists())

    def test_pdf_page_count_and_dimensions_are_checked_before_rendering(self):
        document = Mock()
        document.__enter__ = Mock(return_value=document)
        document.__exit__ = Mock(return_value=False)
        document.__len__ = Mock(return_value=21)
        module = SimpleNamespace(PdfDocument=Mock(return_value=document))
        with patch.dict("sys.modules", {"pypdfium2": module}):
            with self.assertRaises(APIError) as caught:
                inspect_upload(b"%PDF-1.7\nmock", "source.pdf", Limits())
            self.assertEqual(caught.exception.code, "pages_exceeded")
            document.__len__.return_value = 1
            page = Mock()
            page.get_size.return_value = (100000, 100000)
            document.__getitem__ = Mock(return_value=page)
            with self.assertRaises(APIError) as caught:
                inspect_upload(b"%PDF-1.7\nmock", "source.pdf", Limits())
            self.assertEqual(caught.exception.code, "dimensions_exceeded")
            page.render.assert_not_called()
            page.close.assert_called_once()

    def test_filename_and_job_id_traversal_are_rejected(self):
        for filename in ("../x.png", "a\\x.png", "%2e%2e.png", "\x00.png", "..", ""):
            with self.subTest(filename=filename), self.assertRaises(APIError):
                validate_filename(filename)
        for identifier in ("../private", "A" * 32, "a" * 31, "a/" + "b" * 30):
            with self.subTest(identifier=identifier), self.assertRaises(APIError) as caught:
                self.service.get_job(identifier)
            self.assertEqual(caught.exception.code, "invalid_job_id")
        with self.assertRaises(APIError) as caught:
            self.service.get_job("a" * 32)
        self.assertEqual(caught.exception.status, 404)

    def test_retention_counts_existing_data_without_deleting_it(self):
        self.service.limits = Limits(max_jobs=1)
        existing = self.service.storage / "existing-user-data"
        existing.mkdir(parents=True)
        original = existing / "original.txt"
        original.write_text("keep", encoding="utf-8")
        with self.assertRaises(APIError) as caught:
            self.job()
        self.assertEqual(caught.exception.status, 429)
        self.assertEqual(original.read_text(encoding="utf-8"), "keep")

    def test_storage_limit_and_symlinks_fail_closed(self):
        self.service.limits = Limits(max_storage_bytes=1)
        with self.assertRaises(APIError) as caught:
            self.job()
        self.assertEqual(caught.exception.status, 507)
        self.service.limits = Limits()
        (self.workspace / "runs").mkdir()
        self.service.storage.symlink_to(self.workspace, target_is_directory=True)
        with self.assertRaises(APIError) as caught:
            self.job()
        self.assertEqual(caught.exception.code, "unsafe_storage")

    def test_line_failure_keeps_later_actual_record_and_review_status(self):
        job = self.job()

        def recognize(image, segmentation, model):
            if segmentation.lines[0].id == "first":
                raise ValueError("private filesystem details must not escape")
            yield SimpleNamespace(id="second", prediction=KA, confidences=[0.9])

        with patch.object(self.service, "_load_models", return_value=(object(), object())), patch(
                "khojki_ocr.server.pipeline.segment_page", return_value=bounds()), patch(
                "khojki_ocr.server.pipeline.recognize_page", side_effect=recognize):
            self.service.run_job(job)
        state = job.snapshot()
        self.assertEqual(state["status"], "review")
        self.assertTrue(state["review_required"])
        self.assertFalse(state["references_enabled"])
        self.assertEqual(state["pages"][0]["lines"][1]["raw_hypothesis"], KA)
        self.assertEqual(self.service.transcript(job), ("\n" + KA).encode("utf-8"))
        events = [event for _, event in job.events if event["stage"] == "recognition"]
        self.assertEqual([event["line"] for event in events], [1, 2])
        self.assertEqual(events[1]["text"], KA)
        self.assertNotIn("private filesystem", json.dumps(state))
        self.assertTrue((job.directory / "report.json").is_file())

    def test_read_only_record_id_and_model_cache(self):
        class Record:
            id = property(lambda self: "read-only")
            prediction = KA
            confidences = [0.9]

        job = self.job()
        with patch.object(self.service, "_load_models", return_value=(object(), object())), patch(
                "khojki_ocr.server.pipeline.segment_page", return_value=bounds(("first",))), patch(
                "khojki_ocr.server.pipeline.recognize_page", return_value=iter([Record()])):
            self.service.run_job(job)
        self.assertEqual(job.snapshot()["pages"][0]["lines"][0]["raw_hypothesis"], KA)
        loader = Mock(return_value=object())
        torch = SimpleNamespace(set_num_threads=Mock())
        tasks = SimpleNamespace(SegmentationTaskModel=SimpleNamespace(load_model=loader))
        with patch.dict("sys.modules", {"torch": torch, "kraken.tasks": tasks}), patch(
                "khojki_ocr.server.pipeline.load_recognition_model", return_value=object()) as recognition:
            first = self.service._load_models()
            self.assertIs(self.service._load_models(), first)
        loader.assert_called_once()
        recognition.assert_called_once_with(self.model, require_contract=True)

    def test_report_storage_failure_precedes_terminal_event(self):
        job = self.job()
        with patch.object(self.service, "_load_models", return_value=(object(), object())), patch(
                "khojki_ocr.server.pipeline.segment_page", return_value=bounds(())), patch.object(
                self.service, "_write", side_effect=APIError(507, "storage_limit", "Storage full")):
            self.service.run_job(job)
        terminal = [event for _, event in job.events if event["status"] in {"blank", "review", "failed"}]
        self.assertEqual([event["status"] for event in terminal], ["failed"])
        self.assertEqual(job.snapshot()["status"], "failed")

    def test_cancelled_before_execution_never_loads_model(self):
        job = self.job()
        self.service.cancel(job)
        with patch.object(self.service, "_load_models") as loader:
            self.service.run_job(job)
        loader.assert_not_called()
        self.assertEqual(job.snapshot()["status"], "cancelled")
        self.assertFalse(INFERENCE_LOCK.locked())

    def test_cancel_between_lines_retains_partial_record(self):
        job = self.job()
        calls = []

        def recognize(image, segmentation, model):
            calls.append(segmentation.lines[0].id)
            yield SimpleNamespace(id="first", prediction=KA, confidences=[0.9])
            job.cancelled.set()

        with patch.object(self.service, "_load_models", return_value=(object(), object())), patch(
                "khojki_ocr.server.pipeline.segment_page", return_value=bounds()), patch(
                "khojki_ocr.server.pipeline.recognize_page", side_effect=recognize):
            self.service.run_job(job)
        self.assertEqual(calls, ["first"])
        self.assertEqual(job.snapshot()["status"], "cancelled")
        self.assertEqual(job.snapshot()["pages"][0]["lines"][0]["raw_hypothesis"], KA)
        with self.assertRaises(APIError) as caught:
            self.service.transcript(job)
        self.assertEqual(caught.exception.status, 409)

    def test_no_lines_distinguishes_blank_from_content(self):
        for color, expected in (("white", "blank"), ("black", "review")):
            with self.subTest(color=color):
                job = self.service.create_job(png_bytes(color=color), "source.png", start=False)
                with patch.object(self.service, "_load_models", return_value=(object(), object())), patch(
                        "khojki_ocr.server.pipeline.segment_page", return_value=bounds(())):
                    self.service.run_job(job)
                self.assertEqual(job.snapshot()["status"], expected)
                self.assertEqual(job.snapshot()["pages"][0]["status"], expected)
                with self.assertRaises(APIError) as caught:
                    self.service.transcript(job)
                self.assertEqual(caught.exception.status, 404)

    def test_transcript_never_strips_invalid_or_normalizes_raw_output(self):
        job = self.job()
        for text, expected in ((" \n ", 404), (KA + " English", 422), (KA + "\r", 422)):
            with self.subTest(text=text):
                with job.condition:
                    job.state.update(status="review", text=text)
                with self.assertRaises(APIError) as caught:
                    self.service.transcript(job)
                self.assertEqual(caught.exception.status, expected)
        with job.condition:
            job.state.update(status="review", text=KA + "\n " + KA)
        self.assertEqual(self.service.transcript(job), (KA + "\n " + KA).encode("utf-8"))

    def test_preview_is_actual_upload_bounded_and_cached(self):
        job = self.service.create_job(png_bytes((1600, 800), "red"), "source.png", start=False)
        payload = self.service.preview(job, 1)
        with Image.open(io.BytesIO(payload)) as image:
            self.assertEqual(image.size, (1200, 600))
            self.assertEqual(image.getpixel((600, 300)), (255, 0, 0))
        with patch("khojki_ocr.server.pipeline.iter_pages", side_effect=AssertionError("cache missed")):
            self.assertEqual(self.service.preview(job, 1), payload)
        with self.assertRaises(APIError) as caught:
            self.service.preview(job, 2)
        self.assertEqual(caught.exception.status, 404)

    def test_missing_model_prevents_upload(self):
        self.model.unlink()
        self.assertFalse(self.service.config()["model_available"])
        with self.assertRaises(APIError) as caught:
            self.job()
        self.assertEqual(caught.exception.status, 503)


class HTTPTests(ServiceCase):
    def setUp(self):
        super().setUp()
        self.server = LocalHTTPServer(("127.0.0.1", 0), self.service)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.host = f"127.0.0.1:{self.server.server_port}"

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, body, headers={"Host": self.host, **(headers or {})})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_config_and_same_origin_upload_contract(self):
        status, _, payload = self.request("GET", "/api/config")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["max_pages"], 20)
        with patch.object(self.service, "run_job"):
            status, _, payload = self.request("POST", "/api/jobs", self.payload,
                                              {"Content-Type": "application/octet-stream", "X-Filename": "source.png",
                                               "Origin": "http://" + self.host})
        self.assertEqual(status, 202)
        identifier = json.loads(payload)["job_id"]
        status, _, payload = self.request("GET", f"/api/jobs/{identifier}")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["status"], "queued")
        status, _, payload = self.request("POST", f"/api/jobs/{identifier}/cancel")
        self.assertEqual(status, 202)
        self.assertEqual(json.loads(payload)["status"], "cancelling")

    def test_host_origin_and_traversal_are_rejected(self):
        cases = [({"Host": "evil.example"}, "/api/config", 403),
                 ({"Origin": "https://evil.example"}, "/api/config", 403),
                 ({"Origin": "null"}, "/api/config", 403),
                 ({"Sec-Fetch-Site": "cross-site"}, "/api/config", 403),
                 ({}, "/%2e%2e/pyproject.toml", 400),
                 ({}, "/api/jobs/../../private", 400)]
        for headers, path, expected in cases:
            with self.subTest(path=path, headers=headers):
                status, _, _ = self.request("GET", path, headers=headers)
                self.assertEqual(status, expected)

    def test_wrong_upload_type_size_and_filename(self):
        for headers, expected in (({"Content-Type": "image/png", "X-Filename": "source.png"}, 415),
                                  ({"Content-Type": "application/octet-stream", "X-Filename": "../source.png"}, 400),
                                  ({"Content-Type": "application/octet-stream", "X-Filename": "source.png",
                                    "Content-Length": str(21 * 1024 * 1024)}, 413)):
            with self.subTest(headers=headers):
                status, _, _ = self.request("POST", "/api/jobs", self.payload, headers)
                self.assertEqual(status, expected)

    def test_terminal_sse_and_draft_download_headers(self):
        job = self.job()
        with job.condition:
            job.state.update(text=KA, review_required=True)
        job.publish(stage="complete", status="review")
        status, headers, payload = self.request("GET", f"/api/jobs/{job.id}/events")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/event-stream"))
        self.assertIn(b"event: progress\n", payload)
        self.assertIn(b'"status": "review"', payload)
        status, headers, payload = self.request("GET", f"/api/jobs/{job.id}/transcript.txt")
        self.assertEqual((status, payload), (200, KA.encode("utf-8")))
        self.assertEqual(headers["X-Review-Required"], "true")

    def test_static_assets_are_confined_to_web(self):
        self.service.web_root.mkdir()
        (self.service.web_root / "index.html").write_text("<main>local</main>", encoding="utf-8")
        (self.service.web_root / "private.js").symlink_to(self.model)
        status, _, payload = self.request("GET", "/")
        self.assertEqual((status, payload), (200, b"<main>local</main>"))
        self.assertEqual(self.request("GET", "/private.js")[0], 404)
        self.assertEqual(self.request("GET", "/runs/real-adaptation-001/best.safetensors")[0], 404)


if __name__ == "__main__":
    unittest.main()