"""T10 · Domain Shift Radar — PoC pipeline (owner: duyanh211105).

Goal: a ShiftScore per batch that tracks the *model's performance drop*, not only "the images look different".

    01 load      manifest sha256 check + images.csv + model_predictions.csv + image descriptors
    02 process   per batch vs reference (b0_source):
                   - image drift  : MMD^2 (RBF) on 8 standardized condition descriptors, permutation z
                   - model drift  : KS on pred_score distribution, permutation z
                   - shift_score  = w_pred * s_pred + w_img * s_img   (s_pred = z/z_cap, s_img = MMD^2/2, in [0,1])
                   - pred_high_risk = model-drift z >= z_risk          (image drift alone never triggers HOLD)
                   - slice scan   : brightness / sharpness tertiles (reference quantiles) for silent risk
    03 rank      review queue = every batch (B_batch = 6) + top image outliers (B_img)
    04 review    reviewed CSV -> overrides on pred_high_risk + audit log
    05 decide    per batch KEEP / REVIEW / HOLD + release decision + dashboard.html

All weights and thresholds come from configs/tracks/t10.yaml (frozen before any evaluation).
No ground truth is read; no network is used.
"""
from __future__ import annotations

import html
import json

import cv2
import numpy as np
import pandas as pd

from . import t10_features as F
from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.manifest import verify_manifest
from .common.review import load_review_decisions, make_review_queue, write_review_queue
from .metrics.tracks import get_track

BLUE, ORANGE = "#2a78d6", "#eb6834"   # dataviz reference palette, slots 1-2 (fixed order)


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict:
    if ctx.track != "t10":
        raise NotImplementedError(f"this pipeline implements t10 only (got {ctx.track})")
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    for name in ("images.csv", "model_predictions.csv"):
        if not (ctx.input_dir / name).exists():
            raise FileNotFoundError(f"missing input table {ctx.input_dir / name}")
    images = pd.read_csv(ctx.input_dir / "images.csv")
    preds = pd.read_csv(ctx.input_dir / "model_predictions.csv")
    if images.image_id.duplicated().any() or preds.image_id.duplicated().any():
        raise ValueError("duplicate image_id in images.csv or model_predictions.csv")
    df = images.merge(preds[["image_id", "pred_score", "pred_label"]], on="image_id", how="left", validate="1:1")
    if df.pred_score.isna().any():
        raise ValueError(f"{int(df.pred_score.isna().sum())} images have no model prediction")
    if not df.pred_score.between(0, 1).all():
        raise ValueError("pred_score must be in [0, 1]")
    ref = ctx.param("reference_batch", "b0_source")
    if ref not in set(df.batch_id):
        raise ValueError(f"reference batch {ref!r} not found in images.csv")
    feats = pd.DataFrame([F.image_descriptors(ctx.input_dir / p) for p in df.path])
    df = pd.concat([df.reset_index(drop=True), feats], axis=1)
    df["uncertainty"] = F.uncertainty(df.pred_score)
    ctx.stats["n_images"] = len(df)
    ctx.stats["n_batches"] = int(df.batch_id.nunique())
    return {"images": df, "reference": ref}


# ---------------------------------------------------------------- 02 process
def _slice_scan(ref_df: pd.DataFrame, df: pd.DataFrame, ctx: RunContext) -> pd.DataFrame:
    """Per batch × slice stats. Slice edges = reference tertiles, so slices mean the same thing in every batch."""
    min_n = int(ctx.param("slice_min_n", 5))
    unc_gap = float(ctx.param("slice_unc_gap", 0.15))
    rows = []
    ref_unc = ref_df.uncertainty.mean()
    for desc in ("brightness", "log_sharpness"):
        q1, q2 = ref_df[desc].quantile([1 / 3, 2 / 3])
        names = {"brightness": ("dark", "mid", "bright"), "log_sharpness": ("blurry", "mid", "sharp")}[desc]
        lab = np.where(df[desc] < q1, names[0], np.where(df[desc] < q2, names[1], names[2]))
        for (b, s), g in df.assign(_s=lab).groupby(["batch_id", "_s"]):
            n_batch = int((df.batch_id == b).sum())
            gap = g.uncertainty.mean() - ref_unc
            rows.append({
                "batch_id": b, "slice": f"{desc}={s}", "n": len(g), "share": round(len(g) / n_batch, 3),
                "mean_uncertainty": round(g.uncertainty.mean(), 3), "uncertainty_gap_vs_ref": round(gap, 3),
                "pos_rate": round(g.pred_label.mean(), 3),
                "silent_risk": bool(len(g) >= min_n and len(g) / n_batch <= 0.5 and gap >= unc_gap),
            })
    return pd.DataFrame(rows)


def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    df, ref = data["images"], data["reference"]
    n_perm = int(ctx.param("n_perm", 300))
    w_pred, w_img = float(ctx.param("w_pred", 0.6)), float(ctx.param("w_img", 0.4))
    z_risk = float(ctx.param("z_risk", 1.645))
    ref_df = df[df.batch_id == ref]
    ref_z = F.standardize(ref_df, ref_df)
    gamma = F.median_gamma(ref_z)
    mmd = lambda a, b: F._rbf_mmd2(a, b, gamma)  # noqa: E731
    rows = []
    for b, g in df.groupby("batch_id", sort=True):
        if b == ref:
            rows.append({"batch_id": b, "n": len(g), "img_mmd2": 0.0, "img_z": 0.0, "img_p": 1.0,
                         "pred_ks": 0.0, "pred_z": 0.0, "pred_p": 1.0, "top_descriptors": "reference"})
            continue
        gz = F.standardize(ref_df, g)
        im = F.permutation_z(mmd, ref_z, gz, ctx.rng, n_perm)
        pr = F.permutation_z(F.ks_stat, ref_df.pred_score.to_numpy(), g.pred_score.to_numpy(), ctx.rng, n_perm)
        sh = F.per_descriptor_shift(ref_df, g)
        top = sorted(sh.items(), key=lambda kv: -abs(kv[1]))[:3]
        rows.append({"batch_id": b, "n": len(g), "img_mmd2": round(im["stat"], 4), "img_z": round(im["z"], 2),
                     "img_p": round(im["p"], 4), "pred_ks": round(pr["stat"], 3), "pred_z": round(pr["z"], 2),
                     "pred_p": round(pr["p"], 4), "top_descriptors": "; ".join(f"{k} {v:+.1f}sd" for k, v in top)})
    bt = pd.DataFrame(rows)
    bt["mean_uncertainty"] = bt.batch_id.map(df.groupby("batch_id").uncertainty.mean()).round(3)
    bt["pos_rate"] = bt.batch_id.map(df.groupby("batch_id").pred_label.mean()).round(3)

    # Fixed scales (not "divide by max"): v1 divided by the max over batches and turned pure noise
    # (all model-drift z < 1) into a full-scale signal. Now: model z of z_cap (default 3) = full scale;
    # image MMD^2 (RBF kernel, bounded by 2) / 2. Both in [0, 1], comparable across runs.
    z_cap = float(ctx.param("z_cap", 3.0))
    bt["s_pred"] = (bt.pred_z.clip(lower=0) / z_cap).clip(upper=1).round(4)
    bt["s_img"] = (bt.img_mmd2.clip(lower=0) / 2.0).clip(upper=1).round(4)
    bt["shift_score"] = (w_pred * bt.s_pred + w_img * bt.s_img).round(4)
    bt["pred_high_risk"] = (bt.pred_z >= z_risk).astype(int)

    slices = _slice_scan(ref_df, df, ctx)
    silent = slices[slices.silent_risk].groupby("batch_id").slice.apply(lambda s: ",".join(s))
    bt["silent_slices"] = [silent.get(b, "") for b in bt.batch_id]
    big_img = float(ctx.param("z_img_review", 3.0))
    bt["status"] = np.where(bt.pred_high_risk == 1, "HOLD",
                            np.where((bt.img_z >= big_img) | (bt.silent_slices != ""), "REVIEW", "KEEP"))
    bt["reason"] = bt.apply(_batch_reason, axis=1, z_risk=z_risk, big_img=big_img)
    bt["score"] = bt.shift_score

    # image-level outliers for the review queue: atypical conditions × model unsure
    ref_loo = F.knn_atypicality(ref_z, ref_z, k=6)  # includes self (distance 0) -> ~LOO with k=5
    scale = float(np.median(ref_loo)) or 1.0
    atyp = F.knn_atypicality(ref_z, F.standardize(ref_df, df)) / scale
    df["atypicality"] = np.round(atyp, 3)
    df["item_score"] = np.round(atyp * (0.5 + df.uncertainty), 4)

    data.update({"batches": bt, "slices": slices, "images": df})
    return bt


def _batch_reason(r, z_risk: float, big_img: float) -> str:
    if r.top_descriptors == "reference":
        return "reference batch"
    parts = [f"model-drift z={r.pred_z:+.2f} (KS={r.pred_ks:.3f}, p={r.pred_p:.3f})",
             f"image-drift z={r.img_z:+.1f} [{r.top_descriptors}]"]
    if r.pred_high_risk:
        parts.append(f"model z >= {z_risk} -> HIGH RISK")
    elif r.img_z >= big_img:
        parts.append("large visual shift but no model evidence -> possible false alarm, human check")
    if r.silent_slices:
        parts.append(f"silent slice risk: {r.silent_slices}")
    return " | ".join(parts)


# ---------------------------------------------------------------- 03 rank
def _montage(ctx: RunContext, df: pd.DataFrame, batch: str, n: int = 8) -> str:
    g = df[df.batch_id == batch].sort_values("item_score", ascending=False).head(n)
    tiles = [cv2.resize(cv2.imread(str(ctx.input_dir / p)), (192, 108)) for p in g.path]
    while len(tiles) < n:
        tiles.append(np.zeros((108, 192, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, n, 4)])
    p = ctx.out(f"montages/{batch}.jpg")
    cv2.imwrite(str(p), grid)
    return str(p.resolve())


def rank(ctx: RunContext, data: dict) -> pd.DataFrame:
    bt, df = data["batches"], data["images"]
    b_img = int(ctx.param("budget_images", 14))
    bq = bt.assign(item_id="batch:" + bt.batch_id, slice_=bt.silent_slices,
                   action=bt.status, evidence=[_montage(ctx, df, b) for b in bt.batch_id])
    q1 = make_review_queue(bq, "item_id", "shift_score", len(bq), reason_col="reason",
                           action_col="action", slice_col="slice_", evidence_col="evidence")
    cand = df[df.batch_id != data["reference"]]
    iq = cand.assign(item_id="img:" + cand.image_id,
                     reason=cand.apply(lambda r: f"atypicality={r.atypicality:.2f}x ref, pred_score={r.pred_score:.2f} "
                                                 f"(uncertainty {r.uncertainty:.2f})", axis=1),
                     action="check: is the pedestrian visible / prediction right?",
                     evidence=cand.path)
    q2 = make_review_queue(iq, "item_id", "item_score", b_img, reason_col="reason",
                           action_col="action", slice_col="batch_id", evidence_col="evidence")
    q = pd.concat([q1, q2], ignore_index=True)
    q["rank"] = range(1, len(q) + 1)
    write_review_queue(q, ctx.out("review_queue.csv"))
    render_contact_sheet(q, ctx.out("contact_sheet.html"), image_root=ctx.input_dir,
                         title="T10 Domain Shift Radar: batch montages + image outliers", thumb_width=360)
    return q


# ---------------------------------------------------------------- 04 review
def review(ctx: RunContext, data: dict, queue: pd.DataFrame) -> pd.DataFrame:
    """Reviewer semantics (written into out/review_queue_reviewed.csv):

    batch:<id>  approve  -> keep model's pred_high_risk
                edit     -> override_value 0/1 replaces pred_high_risk
                reject   -> pred_high_risk = 0 (reviewer says: false alarm)
                escalate -> pred_high_risk = 1
    img:<id>    escalate -> the image's batch gets pred_high_risk = 1 (silent slice risk confirmed)
                other    -> logged only
    """
    bt = data["batches"].copy()
    path = ctx.output_dir / "review_queue_reviewed.csv"
    if not path.exists():
        ctx.stats["n_reviewed"] = 0
        return bt
    rv = load_review_decisions(path)
    ctx.stats["n_reviewed"] = len(rv)
    img2batch = dict(zip("img:" + data["images"].image_id, data["images"].batch_id))
    for r in rv.itertuples(index=False):
        if r.item_id.startswith("batch:"):
            b = r.item_id.split(":", 1)[1]
        else:
            b = img2batch.get(r.item_id)
        if b is None or b not in set(bt.batch_id):
            raise ValueError(f"review row refers to unknown item {r.item_id!r}")
        mask = bt.batch_id == b
        before = int(bt.loc[mask, "pred_high_risk"].iloc[0])
        after = before
        if r.item_id.startswith("batch:"):
            if r.reviewer_decision == "edit":
                if str(r.override_value).strip() not in {"0", "1"}:
                    raise ValueError(f"{r.item_id}: edit needs override_value 0 or 1, got {r.override_value!r}")
                after = int(r.override_value)
            elif r.reviewer_decision == "reject":
                after = 0
            elif r.reviewer_decision == "escalate":
                after = 1
        elif r.reviewer_decision == "escalate":
            after = 1
        ctx.audit.log(actor=r.reviewer, action=f"review:{r.reviewer_decision}", item_id=r.item_id,
                      before=before, after=after, reason=r.review_note, batch_id=b)
        if after != before:
            bt.loc[mask, "pred_high_risk"] = after
            bt.loc[mask, "status"] = "HOLD" if after else "KEEP"
            bt.loc[mask, "reason"] += f" | human override by {r.reviewer}: {before}->{after}"
    return bt


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, bt: pd.DataFrame) -> Decision:
    owner = ctx.config.get("owner", "TODO-owner")
    for r in bt.itertuples(index=False):
        st = {"HOLD": "HOLD", "KEEP": "KEEP", "REVIEW": "ESCALATE"}[r.status]
        code = {"HOLD": "MODEL_DRIFT_HIGH_RISK", "KEEP": "NO_SIGNIFICANT_DRIFT", "REVIEW": "VISUAL_SHIFT_NEEDS_REVIEW"}[r.status]
        write_decision(Decision(status=st, reason_code=code, reason=r.reason, owner=owner, scope=r.batch_id,
                                evidence=[str(ctx.out("shift_scores.csv")), str(ctx.out("dashboard.html"))],
                                metrics={"shift_score": r.shift_score, "pred_z": r.pred_z, "img_z": r.img_z}),
                       ctx.out(f"decisions/{r.batch_id}.json"), ctx.out("decision_log.csv"))
    hold = bt[bt.status == "HOLD"].batch_id.tolist()
    review_ = bt[bt.status == "REVIEW"].batch_id.tolist()
    if hold:
        return Decision("HOLD", "HIGH_RISK_BATCHES", f"Do not ship model on batches {hold}: significant model-output drift; "
                        f"retrain/relabel before release. Needs human check: {review_}", owner, scope="t10-release",
                        evidence=[str(ctx.out("shift_scores.csv")), str(ctx.out("dashboard.html"))],
                        metrics={"n_hold": len(hold), "n_review": len(review_)})
    if review_:
        return Decision("HOLD", "PENDING_HUMAN_REVIEW", f"No model-drift alarm, but batches {review_} show large visual "
                        "shift or silent slices; release after reviewer sign-off", owner, scope="t10-release",
                        evidence=[str(ctx.out("review_queue.csv"))], metrics={"n_review": len(review_)})
    return Decision("GO", "NO_SIGNIFICANT_DRIFT", "All batches within reference behaviour", owner, scope="t10-release")


# ---------------------------------------------------------------- dashboard
def _bar_svg(bt: pd.DataFrame) -> str:
    """Grouped horizontal bars of the two normalized components (one shared 0..1 axis)."""
    w, lh, left, right = 640, 34, 150, 60
    h = 30 + lh * len(bt) + 20
    sx = lambda v: left + v * (w - left - right)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" role="img" aria-label="Image vs model drift per batch">']
    for t in (0, 0.5, 1):
        out.append(f'<line x1="{sx(t)}" x2="{sx(t)}" y1="20" y2="{h-20}" stroke="#e4e3df"/>'
                   f'<text x="{sx(t)}" y="{h-5}" font-size="11" fill="#52514e" text-anchor="middle">{t}</text>')
    for i, r in enumerate(bt.itertuples(index=False)):
        y = 26 + i * lh
        out.append(f'<text x="{left-8}" y="{y+14}" font-size="12" fill="#0b0b0b" text-anchor="end">{html.escape(r.batch_id)}</text>')
        for j, (v, c, name, z) in enumerate(((r.s_img, BLUE, "image drift", r.img_z), (r.s_pred, ORANGE, "model drift", r.pred_z))):
            yy = y + j * 13
            wd = max(sx(v) - left, 0)
            out.append(f'<rect x="{left}" y="{yy}" width="{wd}" height="11" rx="2" fill="{c}">'
                       f'<title>{html.escape(r.batch_id)} · {name}: normalized {v:.2f} (z={z:+.2f})</title></rect>')
            out.append(f'<text x="{left+wd+4}" y="{yy+9}" font-size="10" fill="#52514e">z={z:+.1f}</text>')
    out.append("</svg>")
    return "".join(out)


def write_dashboard(ctx: RunContext, bt: pd.DataFrame, slices: pd.DataFrame, decision: Decision) -> None:
    icon = {"HOLD": "■ HOLD", "REVIEW": "▲ REVIEW", "KEEP": "● KEEP"}
    cols = ["batch_id", "shift_score", "pred_high_risk", "status", "pred_z", "pred_ks", "img_z", "mean_uncertainty",
            "pos_rate", "top_descriptors", "silent_slices"]
    t = bt[cols].assign(status=bt.status.map(icon)).to_html(index=False, escape=True, border=0)
    sl = slices.sort_values(["silent_risk", "uncertainty_gap_vs_ref"], ascending=False).head(15).to_html(index=False, border=0)
    page = f"""<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8"><title>T10 Domain Shift Radar</title>
<style>body{{font-family:system-ui,sans-serif;background:#fcfcfb;color:#0b0b0b;max-width:980px;margin:24px auto;padding:0 16px}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0 24px}}td,th{{padding:4px 8px;border-bottom:1px solid #e4e3df;text-align:left}}
.legend span{{display:inline-block;width:12px;height:12px;border-radius:2px;margin:0 4px 0 12px;vertical-align:middle}}
.dec{{padding:12px;border:1px solid #e4e3df;border-radius:6px}} small{{color:#52514e}}</style></head><body>
<h1>T10 · Domain Shift Radar</h1>
<div class="dec"><b>Release decision: {html.escape(decision.status)}</b> ({html.escape(decision.reason_code)})<br>
{html.escape(decision.reason)}<br><small>owner: {html.escape(decision.owner)} · run: {html.escape(ctx.run_name)} · seed {ctx.seed}</small></div>
<h2>Image drift vs model drift (normalized, 0–1)</h2>
<div class="legend"><span style="background:{BLUE}"></span>image drift (MMD² on descriptors)<span style="background:{ORANGE}"></span>model drift (KS on pred_score)</div>
{_bar_svg(bt)}
<p><small>shift_score = {ctx.param('w_pred', 0.6)}·model + {ctx.param('w_img', 0.4)}·image. HOLD needs model-drift z ≥ {ctx.param('z_risk', 1.645)};
large image drift alone → REVIEW (possible false alarm). z from {ctx.param('n_perm', 300)} permutations vs {html.escape(str(ctx.param('reference_batch')))}.</small></p>
<h2>Per-batch table</h2>{t}
<h2>Slice scan (top 15, silent-risk first)</h2>{sl}
<p><small>Slice edges = reference tertiles. silent_risk: n ≥ {ctx.param('slice_min_n', 5)}, share ≤ 50% of batch,
mean uncertainty ≥ reference + {ctx.param('slice_unc_gap', 0.15)}.</small></p>
<p>Review images: <a href="contact_sheet.html">contact_sheet.html</a> · queue: review_queue.csv</p>
</body></html>"""
    ctx.out("dashboard.html").write_text(page, encoding="utf-8")


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)
    bt = process(ctx, data)
    ctx.stats["n_items"] = len(bt)
    queue = rank(ctx, data)
    final = review(ctx, data, queue)
    final[spec.pred_columns + ["pred_high_risk"] + [c for c in final.columns if c not in spec.pred_columns + ["pred_high_risk"]]] \
        .to_csv(ctx.out(spec.pred_file), index=False)
    # ablations (same run, same seed): which signal carries the information?
    final[["batch_id"]].assign(shift_score=final.s_img, pred_high_risk=final.pred_high_risk).to_csv(ctx.out("ablation_image_only.csv"), index=False)
    final[["batch_id"]].assign(shift_score=final.s_pred, pred_high_risk=final.pred_high_risk).to_csv(ctx.out("ablation_model_only.csv"), index=False)
    data["slices"].to_csv(ctx.out("slice_scan.csv"), index=False)
    dec = decide(ctx, final)
    write_decision(dec, ctx.out("decision.json"), ctx.out("decision_log.csv"))
    write_dashboard(ctx, final, data["slices"], dec)
    ctx.out("batch_summary.json").write_text(json.dumps(final.drop(columns=["score"]).to_dict("records"), indent=2,
                                                        ensure_ascii=False), encoding="utf-8")
