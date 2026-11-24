import json
import tempfile
import unittest
from pathlib import Path

from experiment import summarize_resolution, summarize_usage


class ExperimentSmokeTest(unittest.TestCase):
    def test_resolution_output_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = summarize_resolution(
                Path("baseline") / "R" / "RQ1" / "at_resolution.csv",
                output,
            )
            self.assertGreater(result["n_total"], 0)
            self.assertIn("mann_whitney_p", result)
            self.assertTrue((output / "rq1_resolution.json").exists())
            self.assertTrue((output / "rq1_resolution.png").exists())

    def test_usage_output_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = summarize_usage(
                Path("baseline") / "scripts" / "data" / "events_gpt.csv",
                output,
            )
            self.assertGreater(result["n_events"], 0)
            self.assertIn("event_counts", result)
            payload = json.loads((output / "rq3_usage.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["n_events"], result["n_events"])


if __name__ == "__main__":
    unittest.main()
