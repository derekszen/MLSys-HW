"""Standard-library-only tests; these never execute or stand in for HCCL."""

import argparse
import builtins
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hccl_smoke.py"
SPEC = importlib.util.spec_from_file_location("hccl_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
_original_import = builtins.__import__


def no_torch_import(name, *args, **kwargs):
    if name.split(".")[0] in ("torch", "torch_npu"):
        raise ImportError("Torch intentionally unavailable in helper tests")
    return _original_import(name, *args, **kwargs)


# Fail at collection time if the helper module acquires a Torch dependency.
with mock.patch("builtins.__import__", side_effect=no_torch_import):
    SPEC.loader.exec_module(smoke)


DEVICES = ("3", "7")
RUN_ID = "this-test-only"


def good_rank(rank):
    return {
        "run_id": RUN_ID, "rank": rank, "world_size": 2,
        "logical_device": rank, "physical_device_requested": DEVICES[rank],
        "visible_devices": ",".join(DEVICES), "backend": "hccl", "passed": True,
        "pid": 1000 + rank,
        "memory": {"allocated_bytes": 0, "reserved_bytes": 512,
                   "peak_allocated_bytes": 256, "peak_reserved_bytes": 512},
        "checks": [
            {"dtype": dtype, "collective": collective, "verified": True,
             "actual": smoke.expected_output(rank, collective),
             "expected": smoke.expected_output(rank, collective)}
            for dtype in smoke.DTYPES for collective in smoke.COLLECTIVES
        ],
    }


class ConfigurationTests(unittest.TestCase):
    def test_physical_ids_preserve_order_and_normalize(self):
        self.assertEqual(smoke.parse_physical_devices("07,3"), ("7", "3"))
        # Nonadjacent IDs are accepted; no topology inference is made here.
        self.assertEqual(smoke.parse_physical_devices("3,7"), DEVICES)

    def test_invalid_devices(self):
        for value in ("", "0", "0,1,2", "1,1", "01,1", "-1,0", "0, 1",
                      "0,", "０,1", "a,1", "0;1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                smoke.parse_physical_devices(value)

    def test_ports(self):
        for value, expected in (("auto", "auto"), ("1", 1), ("65535", 65535)):
            self.assertEqual(smoke.parse_port(value), expected)
        for value in ("0", "65536", "-1", "1.5", "AUTO", "１２", " 123"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                smoke.parse_port(value)

    def test_port_probe_is_loopback_and_keeps_required_port(self):
        with mock.patch.object(smoke.socket, "socket") as factory:
            sock = factory.return_value.__enter__.return_value
            sock.getsockname.return_value = ("127.0.0.1", 23456)
            self.assertEqual(smoke.choose_port(23456), 23456)
            sock.bind.assert_called_once_with(("127.0.0.1", 23456))
            factory.return_value.__exit__.assert_called_once()
        with mock.patch.object(smoke.socket, "socket") as factory:
            sock = factory.return_value.__enter__.return_value
            sock.getsockname.return_value = ("127.0.0.1", 34567)
            self.assertEqual(smoke.choose_port("auto"), 34567)
            sock.bind.assert_called_once_with(("127.0.0.1", 0))
        with mock.patch.object(smoke.socket, "socket") as factory:
            factory.return_value.__enter__.return_value.bind.side_effect = OSError("busy")
            with self.assertRaises(OSError):
                smoke.choose_port(23456)

    def test_cli_required_arguments_and_timeout_bounds(self):
        base = ["--physical-devices", "3,7", "--port", "auto", "--output-dir", "new"]
        args = smoke.parse_args(base)
        self.assertEqual((args.dist_timeout, args.run_timeout), (30, 120))
        invalid = [[], base[:2], base + ["--dist-timeout", "0"],
                   base + ["--run-timeout", "1801"],
                   base + ["--dist-timeout", "30", "--run-timeout", "39"],
                   base + ["--dist-timeout", "nan"]]
        for argv in invalid:
            with self.subTest(argv=argv), mock.patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    smoke.parse_args(argv)
                self.assertNotEqual(raised.exception.code, 0)

    def test_expected_outputs_independent_constants(self):
        self.assertEqual(smoke.expected_output(0, "all_reduce"), [4, 6, 8, 10])
        self.assertEqual(smoke.expected_output(1, "all_reduce"), [4, 6, 8, 10])
        for rank in (0, 1):
            self.assertEqual(smoke.expected_output(rank, "all_gather"),
                             [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(smoke.expected_output(0, "reduce_scatter"), [8, 10, 12, 14])
        self.assertEqual(smoke.expected_output(1, "reduce_scatter"), [16, 18, 20, 22])
        for rank in (-1, 2, True):
            with self.assertRaises(ValueError):
                smoke.expected_output(rank, "all_reduce")
        with self.assertRaises(ValueError):
            smoke.expected_output(0, "broadcast")


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.records = [good_rank(0), good_rank(1)]

    def write_records(self):
        for rank, record in enumerate(self.records):
            smoke.write_json(self.directory / f"rank-{rank}.json", record)

    def aggregate(self):
        return smoke.aggregate_results(self.directory, DEVICES, RUN_ID)

    def test_both_rank_files_required_and_valid(self):
        self.write_records()
        self.assertEqual(self.aggregate(), self.records)
        (self.directory / "rank-1.json").unlink()
        with self.assertRaisesRegex(ValueError, "rank-1.json"):
            self.aggregate()

    def test_corrupt_json_is_failure(self):
        self.write_records()
        (self.directory / "rank-1.json").write_text("{broken", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unreadable"):
            self.aggregate()

    def test_wrong_identity_backend_world_size_or_status_is_failure(self):
        changes = {"rank": 0, "world_size": 1, "logical_device": 0,
                   "physical_device_requested": "3", "visible_devices": "7,3",
                   "backend": "gloo", "passed": False, "pid": 0, "run_id": "stale"}
        for key, value in changes.items():
            with self.subTest(key=key):
                record = good_rank(1)
                record[key] = value
                with self.assertRaises(ValueError):
                    smoke.validate_rank_result(record, 1, DEVICES, RUN_ID)
        record = good_rank(1)
        record["world_size"] = 2.0
        with self.assertRaises(ValueError):
            smoke.validate_rank_result(record, 1, DEVICES, RUN_ID)

    def test_duplicate_pid_is_failure(self):
        self.records[1]["pid"] = self.records[0]["pid"]
        self.write_records()
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.aggregate()

    def test_every_collective_in_both_dtypes_is_required(self):
        for index in range(6):
            for mutation in ("missing", "duplicate", "unverified", "wrong_actual", "wrong_expected"):
                with self.subTest(index=index, mutation=mutation):
                    record = good_rank(1)
                    if mutation == "missing":
                        record["checks"].pop(index)
                    elif mutation == "duplicate":
                        record["checks"][index] = copy.deepcopy(record["checks"][(index + 1) % 6])
                    elif mutation == "unverified":
                        record["checks"][index]["verified"] = False
                    else:
                        key = "actual" if mutation == "wrong_actual" else "expected"
                        record["checks"][index][key] = [999]
                    with self.assertRaises(ValueError):
                        smoke.validate_rank_result(record, 1, DEVICES, RUN_ID)

    def test_exact_numeric_checks_do_not_accept_tolerance_nan_or_booleans(self):
        for actual in ([4, 6, 8, 10.00001], [4, 6, 8, float("nan")],
                       [4, 6, 8], [[4, 6, 8, 10]], [4, 6, 8, "10"]):
            self.assertFalse(smoke.exact_numbers(actual, [4, 6, 8, 10]))
        self.assertFalse(smoke.exact_numbers([False, True], [0, 1]))
        self.assertTrue(smoke.exact_numbers([4.0, 6.0], [4, 6]))

    def test_invalid_memory_is_failure(self):
        for memory in ({}, None, {key: -1 for key in smoke.MEMORY_KEYS},
                       dict(good_rank(0)["memory"], reserved_bytes=0, allocated_bytes=1),
                       dict(good_rank(0)["memory"], allocated_bytes=True)):
            record = good_rank(0)
            record["memory"] = memory
            with self.assertRaises(ValueError):
                smoke.validate_rank_result(record, 0, DEVICES, RUN_ID)


class LifecycleTests(unittest.TestCase):
    def test_existing_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "rank-0.json"
            sentinel.write_text("do not overwrite", encoding="utf-8")
            with mock.patch("sys.stderr", new=io.StringIO()):
                code = smoke.main(["--physical-devices", "3,7", "--port", "auto",
                                   "--output-dir", directory])
            self.assertEqual(code, 1)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not overwrite")
            self.assertFalse((Path(directory) / "result.json").exists())

    def test_missing_torch_is_failed_result_not_skip_or_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with mock.patch("builtins.__import__", side_effect=no_torch_import), \
                    mock.patch.object(smoke, "choose_port", return_value=23456), \
                    mock.patch.dict(smoke.os.environ), mock.patch("sys.stdout", new=io.StringIO()):
                code = smoke.main(["--physical-devices", "3,7", "--port", "auto",
                                   "--output-dir", str(output)])
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertIs(result["passed"], False)
            self.assertIs(result["same_card_verified"], False)
            self.assertIn("Torch intentionally unavailable", result["error"])

    def test_missing_npu_or_wrong_visible_count_is_failed_rank(self):
        for available, count in ((False, 0), (True, 1), (True, 3)):
            with self.subTest(available=available, count=count), tempfile.TemporaryDirectory() as directory:
                torch = ModuleType("torch")
                torch.__path__ = []
                torch.npu = mock.Mock()
                torch.npu.is_available.return_value = available
                torch.npu.device_count.return_value = count
                torch.distributed = ModuleType("torch.distributed")
                torch.distributed.destroy_process_group = mock.Mock()
                modules = {"torch": torch, "torch_npu": ModuleType("torch_npu"),
                           "torch.distributed": torch.distributed}
                config = {"devices": DEVICES, "run_id": RUN_ID, "output_dir": directory}
                with mock.patch.dict(sys.modules, modules), mock.patch.dict(smoke.os.environ):
                    with self.assertRaisesRegex(RuntimeError, "rank 0 failed"):
                        smoke.worker(0, config)
                record = json.loads((Path(directory) / "rank-0.json").read_text(encoding="utf-8"))
                self.assertIs(record["passed"], False)
                self.assertIn("exactly two visible Ascend NPUs", record["error"])
                torch.npu.set_device.assert_not_called()
                torch.distributed.destroy_process_group.assert_not_called()

    def test_deadline_and_failed_child_fail_without_context_join(self):
        for exitcodes, deadline, error in (([None, None], 1, TimeoutError),
                                           ([1, None], 100, RuntimeError)):
            context = mock.Mock()
            context.processes = [mock.Mock(exitcode=code) for code in exitcodes]
            with mock.patch.object(smoke.time, "monotonic", return_value=2):
                with self.assertRaises(error):
                    smoke.wait_for_children(context, deadline)
            context.join.assert_not_called()

    def test_successful_workers_are_joined(self):
        context = mock.Mock()
        context.processes = [mock.Mock(exitcode=0), mock.Mock(exitcode=0)]
        with mock.patch.object(smoke.time, "monotonic", return_value=2):
            smoke.wait_for_children(context, 100)
        context.join.assert_called_once_with(timeout=0)

    def test_cleanup_targets_only_owned_process_objects(self):
        running, exited = mock.Mock(), mock.Mock()
        running.is_alive.side_effect = [True, True, False]
        exited.is_alive.return_value = False
        context = mock.Mock(processes=[running, exited])
        smoke.stop_owned_children(context)
        running.terminate.assert_called_once_with()
        running.kill.assert_called_once_with()
        self.assertEqual(running.join.call_count, 2)
        exited.terminate.assert_not_called()
        exited.kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
