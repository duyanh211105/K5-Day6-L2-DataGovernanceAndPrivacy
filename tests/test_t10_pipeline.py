"""T10 Domain Shift Radar — our tests (success, failure, edge cases, review loop)."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src import main as runner
from src import t10_features as F

FX = "fixtures/t10_domain_shift"


def _run(out, inp=FX, pipeline="src.pipeline"):
    return runner.main(["--track", "t10", "--input", str(inp), "--output", str(out), "--pipeline", pipeline,
                        "--offline", "--seed", "42", "--run-name", "test"])


@pytest.fixture(scope="module")
def poc_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("poc")
    assert _run(out) == 0
    return out


# ---------------------------------------------------------------- success
def test_poc_writes_contract_and_artifacts(poc_out):
    s = pd.read_csv(poc_out / "shift_scores.csv")
    assert list(s.columns[:3]) == ["batch_id", "shift_score", "pred_high_risk"]
    assert len(s) == 6 and s.batch_id.is_unique
    assert s.shift_score.between(0, 1).all() and set(s.pred_high_risk) <= {0, 1}
    assert s.set_index("batch_id").loc["b0_source", "shift_score"] == 0
    for f in ["review_queue.csv", "contact_sheet.html", "dashboard.html", "decision.json", "decision_log.csv",
              "slice_scan.csv", "ablation_image_only.csv", "ablation_model_only.csv"]:
        assert (poc_out / f).exists(), f
    d = json.loads((poc_out / "decision.json").read_text(encoding="utf-8"))
    assert d["owner"] and d["reason_code"] and d["canonical"] in {"GO", "HOLD", "REWORK"}
    q = pd.read_csv(poc_out / "review_queue.csv")
    assert len(q) == 20 and q.item_id.str.startswith("batch:").sum() == 6


def test_same_seed_is_reproducible(poc_out, tmp_path):
    assert _run(tmp_path) == 0
    assert (tmp_path / "shift_scores.csv").read_text() == (poc_out / "shift_scores.csv").read_text()


def test_baseline_runs_and_flags_night(tmp_path):
    assert _run(tmp_path, pipeline="src.baseline_t10") == 0
    b = pd.read_csv(tmp_path / "shift_scores.csv").set_index("batch_id")
    assert b.loc["b1_night", "pred_high_risk"] == 1 and b.loc["b0_source", "shift_score"] == 0


# ---------------------------------------------------------------- failure / edge
def _copy_fixture(tmp_path):
    dst = tmp_path / "t10_copy"
    shutil.copytree(FX, dst, ignore=shutil.ignore_patterns("gt"))
    return dst


def test_tampered_image_fails_provenance(tmp_path):
    fx = _copy_fixture(tmp_path)
    img = fx / "batches/b1_night/b1_night_00.jpg"
    img.write_bytes(img.read_bytes() + b"tamper")
    assert _run(tmp_path / "out", inp=fx) == 2          # sha256 mismatch -> input error, not a silent run


def test_missing_predictions_fail_clearly(tmp_path):
    fx = _copy_fixture(tmp_path)
    p = pd.read_csv(fx / "model_predictions.csv")
    p.iloc[:-1].to_csv(fx / "model_predictions.csv", index=False)   # one image without prediction
    assert _run(tmp_path / "out", inp=fx) == 2


def test_null_calibration_identical_samples_not_significant():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    r = F.permutation_z(F.ks_stat, x[:, 0], x[:, 0].copy(), rng, 200)
    assert r["stat"] == 0 and r["p"] > 0.5            # same distribution -> no alarm


def test_permutation_z_detects_real_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=40), rng.normal(loc=1.5, size=40)
    assert F.permutation_z(F.ks_stat, a, b, rng, 200)["z"] > 3


# ---------------------------------------------------------------- human in the loop
def _review(out, rows):
    q = pd.read_csv(out / "review_queue.csv", dtype=str).fillna("")
    for item, dec, val in rows:
        m = q.item_id == item
        q.loc[m, ["reviewer", "reviewer_decision", "override_value", "review_note"]] = ["tester", dec, val, "unit test"]
    q.to_csv(out / "review_queue_reviewed.csv", index=False)


def test_review_override_changes_risk_and_is_audited(tmp_path):
    assert _run(tmp_path) == 0
    q = pd.read_csv(tmp_path / "review_queue.csv")
    img_item = q[q.item_id.str.startswith("img:")].item_id.iloc[0]
    _review(tmp_path, [("batch:b4_city", "edit", "1"), (img_item, "escalate", "")])
    assert _run(tmp_path) == 0
    s = pd.read_csv(tmp_path / "shift_scores.csv").set_index("batch_id")
    assert s.loc["b4_city", "pred_high_risk"] == 1
    img_batch = img_item.split(":", 1)[1].rsplit("_", 1)[0]
    assert s.loc[img_batch, "pred_high_risk"] == 1
    events = [json.loads(l) for l in (tmp_path / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["action"] == "review:edit" and e["item_id"] == "batch:b4_city" and e["actor"] == "tester" for e in events)
    assert json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))["reason_code"] == "HIGH_RISK_BATCHES"


def test_review_bad_override_value_fails(tmp_path):
    assert _run(tmp_path) == 0
    _review(tmp_path, [("batch:b2_rain", "edit", "maybe")])
    assert _run(tmp_path) == 2
