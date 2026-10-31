"""CPU-only pure-helper checks, not msprof/NPU capture or lifecycle evidence."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("profile_workers", ROOT / "scripts/profile_workers.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def child(pid, ppid=100, name="tokenspeed::sch", created=1234.5):
    return {"pid": pid, "ppid": ppid, "name": name, "create_time": created}


def captured(*records):
    return {"parent_pid": 100, "children": list(records)}


def outputs(concurrency=2):
    return [{"output_ids": [index + 7] * 16, "meta_info": {
        "prompt_tokens": 32, "completion_tokens": 16, "cached_tokens": 0}}
        for index in range(concurrency)]


class WorkerProfileTests(unittest.TestCase):
    def test_only_captured_connected_children_are_selected(self):
        first = child(102, ppid=101)
        second = child(103)
        snapshot = captured(second, child(104, ppid=999), first,
                            child(101, name="python"), child(105, name="resource_tracker"))
        before = copy.deepcopy(snapshot)
        self.assertEqual(profile.select_workers(snapshot), [first, second])
        self.assertEqual(snapshot, before)
        # Returned records are copies, not mutable aliases into the snapshot.
        profile.select_workers(snapshot)[0]["name"] = "changed"
        self.assertEqual(snapshot, before)

    def test_existing_full_names_are_supported_without_inventing_device_mapping(self):
        workers = [child(102, name="tokenspeed::scheduler_tp0"),
                   child(103, name="tokenspeed::scheduler_tp1")]
        self.assertEqual(profile.select_workers(captured(*workers)), workers)
        self.assertFalse(any("device" in key or "rank" in key for key in workers[0]))
        self.assertEqual(len(profile.select_workers(captured(
            child(102, name="tokenspeed::scheduler"), child(103)))), 2)

    def test_names_must_not_match_unrelated_substrings(self):
        for name in ("scheduler", "other::scheduler_tp0", "tokenspeed::scheduler_helper",
                     "tokenspeed::scheduler_tp0 extra", "tokenspeed::sch-unrelated", None):
            with self.subTest(name=name):
                self.assertFalse(profile.scheduler_name(name))

    def test_zero_one_or_three_owned_schedulers_are_rejected(self):
        for count in (0, 1, 3):
            with self.subTest(count=count), self.assertRaises(ValueError):
                profile.select_workers(captured(*(child(102 + i) for i in range(count))))

    def test_disconnected_and_cyclic_records_cannot_complete_the_pair(self):
        for records in ((child(103, ppid=999),),
                        (child(103, ppid=104), child(104, ppid=103, name="python"))):
            with self.subTest(records=records), self.assertRaises(ValueError):
                profile.select_workers(captured(child(102), *records))

    def test_ambiguous_pid_and_parent_reinsertion_are_rejected(self):
        for snapshot in (captured(child(102), child(102)),
                         captured(child(102), child(103), child(100, ppid=99))):
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                profile.select_workers(snapshot)

    def test_incomplete_identity_or_failed_snapshot_is_rejected(self):
        for value in (None, 0, -1, float("nan"), float("inf"), True):
            with self.subTest(created=value), self.assertRaises(ValueError):
                profile.select_workers(captured(child(102, created=value), child(103)))
        snapshot = captured(child(102), child(103))
        snapshot["error"] = "AccessDenied"
        with self.assertRaises(ValueError):
            profile.select_workers(snapshot)
        snapshot.pop("error")
        snapshot["children"][0]["error"] = "NoSuchProcess"
        with self.assertRaises(ValueError):
            profile.select_workers(snapshot)

    def test_an_unreadable_owned_name_cannot_hide_a_third_scheduler(self):
        for extra in (child(104, name=None), child(104, name=""),
                      dict(child(104, name="python"), error="AccessDenied")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                profile.select_workers(captured(child(102), child(103), extra))

    def test_duplicate_or_out_of_tp2_rank_names_are_rejected(self):
        for names in (("tokenspeed::scheduler_tp0", "tokenspeed::scheduler_tp0"),
                      ("tokenspeed::scheduler_tp1", "tokenspeed::scheduler_tp2")):
            with self.subTest(names=names), self.assertRaises(ValueError):
                profile.select_workers(captured(child(102, name=names[0]), child(103, name=names[1])))

    def test_collector_command_is_exactly_pid_scoped_dynamic_collection(self):
        for duration in (8, 20):
            with self.subTest(duration=duration):
                command = profile.collector_command("/vendor/msprof", 102, duration, "/own new/raw")
                self.assertEqual(command, ["/vendor/msprof", "--dynamic=on", "--pid=102",
                    f"--duration={duration}", "--output=/own new/raw", "--task-time=on",
                    "--hccl=on", "--runtime-api=off", "--ai-core=off",
                    "--storage-limit=200MB", "--export=off"])
                self.assertFalse(any(option.startswith(("--application", "--sys", "--devices"))
                                     for option in command))

    def test_collector_command_rejects_unbounded_or_missing_pid(self):
        for pid, duration in ((0, 8), (-1, 8), (True, 8), ("102", 8), (102, 7),
                              (102, 21), (102, 8.0), (102, True)):
            with self.subTest(pid=pid, duration=duration), self.assertRaises(ValueError):
                profile.collector_command("msprof", pid, duration, "/own/raw")

    def test_repeat_requires_exact_ids_for_each_request(self):
        warmup = outputs()
        profile.validate_repeat(copy.deepcopy(warmup), warmup, 2)
        changed = copy.deepcopy(warmup)
        changed[1]["output_ids"][0] += 1
        with self.assertRaises(ValueError):
            profile.validate_repeat(changed, warmup, 2)
        with self.assertRaises(ValueError):
            profile.validate_repeat(list(reversed(warmup)), warmup, 2)

    def test_repeat_requires_all_counts_and_no_cache_reuse(self):
        for field, value in (("prompt_tokens", 31), ("completion_tokens", 15), ("cached_tokens", 1)):
            changed = outputs()
            changed[0]["meta_info"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                profile.validate_repeat(changed, outputs(), 2)
        for changed in (outputs(1), [], outputs() + outputs(1)):
            with self.subTest(count=len(changed)), self.assertRaises(ValueError):
                profile.validate_repeat(changed, outputs(), 2)
        changed = outputs()
        changed[0]["output_ids"].pop()
        with self.assertRaises(ValueError):
            profile.validate_repeat(changed, outputs(), 2)

    def test_metadata_only_inventory_does_not_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            data = root / "PROF_test/device_0/data/trace.bin"
            data.parent.mkdir(parents=True)
            data.write_bytes(b"not evidence of device events")
            outside = Path(tmp) / "outside"
            outside.mkdir()
            (outside / "private.bin").write_bytes(b"excluded")
            (root / "linked-directory").symlink_to(outside, target_is_directory=True)
            (data.parent / "linked-file").symlink_to(outside / "private.bin")
            rows = profile.artifact_metadata(root)
            self.assertEqual(rows, [{"path": "PROF_test/device_0/data/trace.bin",
                                     "bytes": data.stat().st_size,
                                     "mtime_unix_ns": data.stat().st_mtime_ns}])
            self.assertTrue(profile.has_device_data(rows))

    def test_exit_or_host_logs_alone_cannot_satisfy_device_data_gate(self):
        for rows in ([], [{"path": "collector.log", "bytes": 100}],
                     [{"path": "PROF_x/host/data/trace", "bytes": 100}],
                     [{"path": "PROF_x/device_0/data/trace", "bytes": 0}],
                     [{"path": "PROF_x/device_0/info.json", "bytes": 100}],
                     [{"path": "PROF_x/device_other/data/trace", "bytes": 100}]):
            with self.subTest(rows=rows):
                self.assertFalse(profile.has_device_data(rows))


if __name__ == "__main__":
    unittest.main()
