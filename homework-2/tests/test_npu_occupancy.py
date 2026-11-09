"""Regression tests using the actual captured npu-smi column layout."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("npu_occupancy", ROOT / "scripts/npu_occupancy.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
HEADER = "| NPU Chip | Process id | Process name | Process memory(MB) |\n+----+\n"


class OccupancyTests(unittest.TestCase):
    def test_actual_pre_run_inventories_only_chip_one_is_free(self):
        paths = sorted((ROOT / "results/steven-streaming-20261109").glob("*-inventory.txt"))
        self.assertEqual(len(paths), 5)
        expected = {(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)}
        for path in paths:
            with self.subTest(path=path.name):
                self.assertEqual(module.occupied_chips(path.read_text()), expected)

    def test_pipe_between_location_and_pid_is_required(self):
        self.assertEqual(module.occupied_chips(HEADER + "| 0 1 | 123 | python | 112 |\n"), {(0, 1)})
        with self.assertRaises(ValueError):
            module.occupied_chips(HEADER + "| 0 1 123 python 112 |\n")

    def test_missing_empty_or_duplicated_header_is_not_free(self):
        for text in ("", "npu-smi failed", HEADER, HEADER + HEADER):
            with self.subTest(text=text), self.assertRaises(ValueError):
                module.occupied_chips(text)

    def test_partial_malformed_table_does_not_return_partial_availability(self):
        for row in ("| 0 1 | unknown | python | 112 |", "permission denied",
                    "| 0 1 | 123 | python | unavailable |"):
            with self.subTest(row=row), self.assertRaises(ValueError):
                module.occupied_chips(HEADER + "| 0 0 | 456 | python | 112 |\n" + row)

    def test_explicit_empty_table_cannot_conflict_with_process_rows(self):
        empty = HEADER + "| No running processes found |\n"
        self.assertEqual(module.occupied_chips(empty), set())
        with self.assertRaises(ValueError):
            module.occupied_chips(empty + "| 0 0 | 456 | python | 112 |\n")


if __name__ == "__main__":
    unittest.main()
