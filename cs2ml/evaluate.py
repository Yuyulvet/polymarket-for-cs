"""Evaluation: metrics, calibration, baselines comparison.

Usage:
    python -m cs2ml.evaluate
Writes reports/eval_report.md and prints a summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from . import config


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def _calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        rows.append({
            "bin": f"[{edges[b]:.1f},{edges[b + 1]:.1f})",
            "n": int(sel.sum()),
            "avg_pred": float(p[sel].mean()),
            "actual": float(y[sel].mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    preds = pd.read_parquet(config.DATA_DIR / "predictions.parquet")
    y = preds["y"].to_numpy()

    models = {
        "constant 0.5": np.full_like(y, 0.5, dtype=float),
        "glicko-only": preds["p_glicko"].to_numpy(),
        "logistic": preds["p_lr"].to_numpy(),
        "lgbm+isotonic": preds["p_lgbm"].to_numpy(),
    }

    lines: list[str] = ["# CS2 胜率模型评估报告（walk-forward，无泄漏）", ""]
    lines.append(f"- 样本：{len(preds)} 场比赛（{preds['start_date'].min()[:10]} ~ {preds['start_date'].max()[:10]}）")
    lines.append(f"- team1 胜率基准：{y.mean():.4f}")
    lines.append("")

    lines.append("## 总体指标")
    lines.append("")
    lines.append("| 模型 | Accuracy | LogLoss | Brier |")
    lines.append("|---|---|---|---|")
    for name, p in models.items():
        m = _metrics(y, p)
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['logloss']:.4f} | {m['brier']:.4f} |")

    # bo3.gg AI baseline (winner pick only -> accuracy on available subset)
    ai_mask = preds["ai_available"] == 1
    if ai_mask.any():
        y_ai = y[ai_mask.to_numpy()]
        acc_ai = float(((preds.loc[ai_mask, "p_ai"] >= 0.5).astype(int) == y_ai).mean())
        acc_ours = float(((preds.loc[ai_mask, "p_lgbm"] >= 0.5).astype(int) == y_ai).mean())
        lines.append("")
        lines.append(f"- bo3.gg AI 参考线（{int(ai_mask.sum())} 场有预测）：其 accuracy **{acc_ai:.4f}**，"
                     f"同期我们的 lgbm accuracy **{acc_ours:.4f}**")

    lines.append("")
    lines.append("## 校准表（lgbm+isotonic）")
    lines.append("")
    cal = _calibration_table(y, models["lgbm+isotonic"])
    lines.append("| 概率区间 | 场次 | 平均预测 | 实际胜率 |")
    lines.append("|---|---|---|---|")
    for _, r in cal.iterrows():
        lines.append(f"| {r['bin']} | {r['n']} | {r['avg_pred']:.3f} | {r['actual']:.3f} |")

    lines.append("")
    lines.append("## 分赛事级别")
    lines.append("")
    lines.append("| tier | 场次 | lgbm acc | lgbm logloss | glicko acc |")
    lines.append("|---|---|---|---|---|")
    for tier, g in preds.groupby("tier"):
        yt = g["y"].to_numpy()
        m_ours = _metrics(yt, g["p_lgbm"].to_numpy())
        m_gl = _metrics(yt, g["p_glicko"].to_numpy())
        lines.append(f"| {tier} | {len(g)} | {m_ours['accuracy']:.4f} | {m_ours['logloss']:.4f} | {m_gl['accuracy']:.4f} |")

    lines.append("")
    lines.append("## 按版本时代")
    lines.append("")
    if "version_era" in preds.columns:
        lines.append("| 版本 | 场次 | lgbm acc | lgbm logloss |")
        lines.append("|---|---|---|---|")
        for era, g in preds.groupby("version_era"):
            yt = g["y"].to_numpy()
            m_ours = _metrics(yt, g["p_lgbm"].to_numpy())
            lines.append(f"| {era} | {len(g)} | {m_ours['accuracy']:.4f} | {m_ours['logloss']:.4f} |")
    else:
        lines.append("- (无 version_era 列 — 需重新运行 `python -m cs2ml.train`)")

    lines.append("")
    lines.append("## 按季度（时间稳定性）")
    lines.append("")
    q = pd.to_datetime(preds["start_date"], utc=True, format="ISO8601").dt.to_period("Q").astype(str)
    lines.append("| 季度 | 场次 | lgbm acc | lgbm logloss |")
    lines.append("|---|---|---|---|")
    for period, g in preds.groupby(q):
        yt = g["y"].to_numpy()
        m_ours = _metrics(yt, g["p_lgbm"].to_numpy())
        lines.append(f"| {period} | {len(g)} | {m_ours['accuracy']:.4f} | {m_ours['logloss']:.4f} |")

    report = "\n".join(lines)
    out = config.REPORTS_DIR / "eval_report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nreport saved -> {out}")


if __name__ == "__main__":
    main()
