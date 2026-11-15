"""T10 baseline (handout §4): compare mean brightness only.

    python -m src.main --track t10 --pipeline src.baseline_t10 --output out/baseline --run-name baseline --offline --seed 42

shift_score    = |mean_brightness(batch) - mean_brightness(ref)| / sd_brightness(ref)
pred_high_risk = shift_score >= 2.0   (frozen before evaluation, same as PoC's spirit: ~2 sd)
Same pool (6 batches), same seed, same reference batch as the PoC.
"""
from __future__ import annotations

from .common.context import RunContext
from .common.decision import Decision, write_decision
from .metrics.tracks import get_track
from .pipeline import load

BASELINE_SD_THRESHOLD = 2.0


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)  # same provenance check + inputs as the PoC
    df, ref = data["images"], data["reference"]
    r = df[df.batch_id == ref].brightness
    mu, sd = r.mean(), r.std(ddof=1) or 1.0
    bt = df.groupby("batch_id", sort=True).brightness.mean().rename("mean_brightness").reset_index()
    bt["shift_score"] = ((bt.mean_brightness - mu).abs() / sd).round(4)
    bt["pred_high_risk"] = (bt.shift_score >= BASELINE_SD_THRESHOLD).astype(int)
    ctx.stats["n_items"] = len(bt)
    bt[["batch_id", "shift_score", "pred_high_risk", "mean_brightness"]].to_csv(ctx.out(spec.pred_file), index=False)
    flagged = bt[bt.pred_high_risk == 1].batch_id.tolist()
    write_decision(Decision("HOLD", "BASELINE_ONLY", f"Brightness-only baseline flags {flagged}; never release on this",
                            owner=ctx.config.get("owner", "baseline"), scope=ctx.track,
                            evidence=[str(ctx.out(spec.pred_file))]),
                   ctx.out("decision.json"), ctx.out("decision_log.csv"))
