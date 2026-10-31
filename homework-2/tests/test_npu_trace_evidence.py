"""Reject host-only/malformed profiler exports; accept documented Chrome layouts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from profile_engine_npu import trace_events


class TraceEvidenceTests(unittest.TestCase):
    def check_document(self, value):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps(value))
            result = trace_events(path)
            self.assertEqual(result["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(result["bytes"], path.stat().st_size)
            return result

    def test_cann_array_has_device_task(self):
        events = [{"ph": "X", "name": "aclnnMm", "dur": 4.3, "args": {"Task Type": "AI_CORE"}}]
        self.assertEqual(self.check_document(events)["device_compute_events"], 1)

    def test_chrome_object_has_kernel(self):
        events = [{"ph": "X", "cat": "kernel", "dur": 2.0}]
        self.assertEqual(self.check_document({"traceEvents": events})["events"], 1)

    def test_host_only_or_empty_not_device_capture(self):
        for document in ({}, [], {"traceEvents": []}, [{"ph": "X", "name": "aten::mm", "dur": 100}],
                         [{"ph": "X", "cat": "kernel", "dur": 0}]):
            with self.subTest(document=document), self.assertRaises(ValueError):
                self.check_document(document)


if __name__ == "__main__":
    unittest.main()
