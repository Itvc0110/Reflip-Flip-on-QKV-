"""Unit test for tools/compare_lm_eval_samples.py using synthetic lm-eval fixtures.

No model, GPU, or network needed: builds two samples_arc_easy.jsonl files shaped
like real `lm_eval --log_samples` output (doc_id / doc / filtered_resps), then
checks the join, correctness, fixed/regressed selection, and plotting.

Usage:
    python -m pytest tests/test_compare_lm_eval_samples.py -v
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import compare_lm_eval_samples as cmp  # noqa: E402


def make_doc(doc_id, answer_key, lls):
    """One lm-eval-shaped multiple-choice sample record."""
    return {
        "doc_id": doc_id,
        "doc": {
            "question": f"Question number {doc_id}?",
            "choices": {"text": [f"choice A{doc_id}", f"choice B{doc_id}",
                                 f"choice C{doc_id}", f"choice D{doc_id}"],
                        "label": ["A", "B", "C", "D"]},
            "answerKey": answer_key,
        },
        "filtered_resps": [[ll, False] for ll in lls],
    }


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def build_fixtures(root: Path):
    # gold answers: doc0->B(1), doc1->A(0), doc2->C(2)
    baseline = [
        make_doc(0, "B", [-1.0, -3.0, -4.0, -5.0]),   # predicts A -> WRONG
        make_doc(1, "A", [-0.5, -2.0, -3.0, -4.0]),   # predicts A -> correct
        make_doc(2, "C", [-4.0, -1.0, -2.0, -3.0]),   # predicts B -> WRONG
    ]
    variant = [
        make_doc(0, "B", [-2.5, -1.0, -4.0, -5.0]),   # predicts B -> FIXED
        make_doc(1, "A", [-2.0, -0.5, -3.0, -4.0]),   # predicts B -> REGRESSED
        make_doc(2, "C", [-4.0, -3.0, -1.0, -2.0]),   # predicts C -> FIXED
    ]
    write_jsonl(root / "base" / "samples_arc_easy_2026.jsonl", baseline)
    write_jsonl(root / "var" / "samples_arc_easy_2026.jsonl", variant)


def test_helpers_and_selection():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_fixtures(root)

        base = cmp.load_samples(cmp.find_samples_file(root / "base", "arc_easy"))
        var = cmp.load_samples(cmp.find_samples_file(root / "var", "arc_easy"))
        assert set(base) == set(var) == {0, 1, 2}

        assert cmp.predicted_index(base[0]) == 0 and cmp.gold_index(base[0]) == 1
        assert not cmp.is_correct(base[0])
        assert cmp.is_correct(var[0])          # fixed
        assert cmp.is_correct(base[1]) and not cmp.is_correct(var[1])  # regressed
        assert not cmp.is_correct(base[2]) and cmp.is_correct(var[2])  # fixed

        fixed = [d for d in sorted(base) if not cmp.is_correct(base[d]) and cmp.is_correct(var[d])]
        regressed = [d for d in sorted(base) if cmp.is_correct(base[d]) and not cmp.is_correct(var[d])]
        assert fixed == [0, 2]
        assert regressed == [1]

        # plotting produces a file
        out = root / "plots" / "example.png"
        cmp.plot_example(0, base[0], var[0], out, "RTN", "RTN + ReFlip")
        assert out.exists() and out.stat().st_size > 0


def test_cli_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_fixtures(root)
        script = Path(cmp.__file__)
        proc = subprocess.run(
            [sys.executable, str(script),
             "--baseline-dir", str(root / "base"),
             "--variant-dir", str(root / "var"),
             "--task", "arc_easy", "--top-n", "2",
             "--plot-dir", str(root / "plots")],
            capture_output=True, text=True, encoding="utf-8")
        assert proc.returncode == 0, proc.stderr
        assert "accuracy: 1/3" in proc.stdout   # baseline
        assert "accuracy: 2/3" in proc.stdout   # variant
        assert "fixed by RTN + ReFlip: 2" in proc.stdout
        assert "regressed: 1" in proc.stdout
        pngs = list((root / "plots").glob("fixed_*.png"))
        assert len(pngs) == 2


if __name__ == "__main__":
    test_helpers_and_selection()
    test_cli_end_to_end()
    print("OK: compare_lm_eval_samples tests passed")
