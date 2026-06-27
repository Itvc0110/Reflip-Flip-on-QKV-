import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PlotQkvSummaryTest(unittest.TestCase):
    def test_report_labels_match_standalone_qk_scope(self):
        from tools.plot_qkv_summary import report_method_label, report_title_prefix

        summary = {"xspot": {"layer_id": 8, "group_id": 3}}

        self.assertEqual("RTN", report_method_label("Nearest"))
        self.assertEqual("Flip + ReFlip", report_method_label("ReFlip"))
        self.assertEqual(
            "Standalone Q--K Surrogate: Layer 8, GQA Group 3",
            report_title_prefix(summary),
        )

    def test_writes_all_summary_plots(self):
        from tools.plot_qkv_summary import plot_summary

        summary = {
            "xspot": {"layer_id": 8, "group_id": 3},
            "flip_stats": {
                "total_flip_weights": 120,
                "wq_total_flips": 96,
                "wk_total_flips": 24,
            },
            "reflip_stats": {"total_integer_flips": 8},
            "metrics": {
                "nearest_mean_abs_attention_error": 0.20,
                "flip_mean_abs_attention_error": 0.10,
                "reflip_mean_abs_attention_error": 0.05,
                "nearest_mean_abs_relative_error_pct": 2.0,
                "flip_mean_abs_relative_error_pct": 1.0,
                "reflip_mean_abs_relative_error_pct": 0.5,
                "nearest_max_abs_attention_error": 0.30,
                "flip_max_abs_attention_error": 0.15,
                "reflip_max_abs_attention_error": 0.08,
                "nearest_to_flip_mean_improvement_pct": 50.0,
                "nearest_to_reflip_mean_improvement_pct": 75.0,
                "flip_to_reflip_mean_improvement_pct": 50.0,
            },
            "heads": [
                {
                    "head": 0,
                    "original_score": -10.0,
                    "nearest_score": -9.7,
                    "flip_score": -9.9,
                    "reflip_score": -9.95,
                    "nearest_abs_error": 0.30,
                    "flip_abs_error": 0.10,
                    "reflip_abs_error": 0.05,
                },
                {
                    "head": 1,
                    "original_score": -5.0,
                    "nearest_score": -5.2,
                    "flip_score": -5.1,
                    "reflip_score": -5.03,
                    "nearest_abs_error": 0.20,
                    "flip_abs_error": 0.10,
                    "reflip_abs_error": 0.03,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_path = tmp_path / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            written = plot_summary(summary_path, tmp_path)

            expected = {
                "score_comparison_by_head.png",
                "error_waterfall_by_head.png",
                "method_summary_errors.png",
                "flip_cost_vs_error_reduction.png",
                "manual_attention_error_by_head.png",
            }
            self.assertEqual(expected, {path.name for path in written})
            for path in written:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
