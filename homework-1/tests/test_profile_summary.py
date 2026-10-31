"""Host trace summarization regressions; runnable without PyTorch."""
import importlib.util
import json
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_profile.py"
spec = importlib.util.spec_from_file_location("summarize_profile", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProfileSummaryTests(unittest.TestCase):
    def test_valid_name_unchanged(self):
        data = {"traceEvents": [{"name": 'quote " and slash \\', "pid": 1}]}
        parsed, repairs = module.repair_names(json.dumps(data))
        self.assertEqual(parsed, data)
        self.assertEqual(repairs, 0)

    def test_nested_json_name_is_repaired(self):
        raw = '{"traceEvents": [\n{"name": "tokenspeed.forward.v1:{"mode":"EXTEND"}", "pid": 1}\n]}'
        parsed, repairs = module.repair_names(raw)
        self.assertEqual(parsed["traceEvents"][0]["name"], 'tokenspeed.forward.v1:{"mode":"EXTEND"}')
        self.assertEqual(repairs, 1)

    def test_exclusive_time_subtracts_only_direct_children(self):
        def event(name, ts, dur, tid=1):
            return dict(name=name, ts=ts, dur=dur, tid=tid, pid=1, ph="X", cat="cpu_op")
        events = [event("parent", 0, 10), event("child", 1, 5), event("nested", 2, 2),
                  event("child", 7, 2), event("parallel", 0, 10, tid=2)]
        values = {x["name"]: x for x in module.summarize(events)["cpu_ops"]}
        self.assertEqual(values["parent"]["self_us"], 3)
        self.assertEqual(values["child"]["self_us"], 5)
        self.assertEqual(values["child"]["calls"], 2)
        self.assertEqual(values["parallel"]["self_us"], 10)

    def test_non_nested_events_fail(self):
        events = [dict(name="x", ts=0, dur=2, tid=1, pid=1, ph="X", cat="cpu_op"),
                  dict(name="y", ts=1, dur=2, tid=1, pid=1, ph="X", cat="cpu_op")]
        with self.assertRaises(ValueError):
            module.summarize(events)


if __name__ == "__main__":
    unittest.main()
