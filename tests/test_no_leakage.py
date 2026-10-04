"""Leakage guards. These are meant to break the build, not to be read in review.

`reports/03_evaluation_strategy.md` §5 lists five leakage vectors. Three of them are mechanically
checkable and are checked here.

Run:  python -m pytest tests/ -q      (or: python tests/test_no_leakage.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "poc"))

POC_DATA = ROOT / "poc/data"


def test_full_persona_is_never_imported() -> None:
    """Vector 1: full_persona embeds the wave-4 ground truth (100.0%, reports/exploration/02_leakage_audit.md).

    The audit script is the one legitimate reader; everything else must not touch it.
    """
    allowed = {"02_leakage_audit.py", "04_persona_budget.py", "common.py"}
    offenders = []
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "poc").rglob("*.py")):
        if p.name in allowed:
            continue
        if "full_persona" in p.read_text():
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, f"full_persona referenced outside the audit: {offenders}"


def test_prompts_contain_no_wave4_answer() -> None:
    """Vectors 2 and 3: the wave-4 answer must never reach the prompt.

    Checked structurally rather than by string match: in the held_out arm the target
    item must not appear in the persona at all, and in both arms any prior answer shown
    for the target item must be the wave-1-3 value, never the wave-4 one.
    """
    from common import load_responses

    w13, w4 = load_responses()
    for arm in ("held_out", "same_item"):
        for split in ("train", "test"):
            path = POC_DATA / f"{arm}_{split}.jsonl"
            if not path.exists():
                continue
            n = 0
            with path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    n += 1
                    assert r["copyfwd"] == int(w13.loc[r["pid"], r["col"]]), (
                        "the copy-forward field must hold the wave-1-3 answer")
                    # The label the model is trained toward must be the wave-4 value.
                    assert r["answer"] == int(w4.loc[r["pid"], r["col"]])
                    if arm == "held_out":
                        # Compare full question text, not a prefix: the pricing block contains
                        # items that share 60 characters ("Soft Drinks - Carbonated" vs
                        # "- Non Carbonated") while being genuinely different questions.
                        target_q = r["prompt"].split("Question:")[-1].split("\nOptions:")[0].strip()
                        persona = r["prompt"].split("Question:")[0]
                        shown = {ln.lstrip("- ").rsplit(" -> ", 1)[0].strip()
                                 for ln in persona.splitlines() if ln.startswith("- ")}
                        assert target_q not in shown, (
                            f"held_out arm leaked the target item into the persona for "
                            f"pid={r['pid']} col={r['col']}")
            assert n, f"{path} is empty"


def test_splits_are_by_participant() -> None:
    """Vector 4: splitting on (person, question) pairs puts the same person in both sides."""
    for arm in ("held_out", "same_item"):
        tr, te = POC_DATA / f"{arm}_train.jsonl", POC_DATA / f"{arm}_test.jsonl"
        if not (tr.exists() and te.exists()):
            continue
        a = {json.loads(l)["pid"] for l in tr.read_text().splitlines() if l}
        b = {json.loads(l)["pid"] for l in te.read_text().splitlines() if l}
        assert not (a & b), f"{arm}: {len(a & b)} participants appear in train and test"


def test_population_mode_excludes_test_participants() -> None:
    """A baseline fitted on the test set is not a baseline."""
    meta = json.loads((POC_DATA / "meta.json").read_text())
    assert "pop_mode" in meta and meta["pop_mode"], "population-mode baseline missing"
    assert meta["n_train_pids"] + meta["n_test_pids"] == 2058


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} leakage guards passed")
