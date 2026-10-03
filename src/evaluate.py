"""Многокритериальная оценка всех файлов results/preds_*.jsonl.

Метрики: accuracy, macro-F1 (+95% bootstrap CI), F1 по классам, MCC, balanced accuracy,
доля невалидных ответов (невалидный ответ = ошибка при расчёте метрик качества),
ECE по заявленной уверенности (10 бинов), латентность p50/p95, токены и стоимость API на 1000 клипов.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CLASSES = ["male", "female", "group", "instrumental"]
INVALID = "invalid"
# Цены USD за 1M токенов (input, output) у API-провайдеров на 2026-10-03, см. отчёт:
# gemma-3-4b-it — OpenRouter API; Qwen2.5-VL-7B/32B — pricepertoken.com (верхняя оценка, Fireworks для 32B);
# GigaChat-2-Max — пакет 1950 руб / 3M токенов = 650 руб/1M, пересчёт по 1 USD = 81 руб.
PRICES = {"gemma3_4b": (0.05, 0.10), "qwen25vl7b": (0.20, 0.20), "qwen25vl32b": (0.90, 0.90),
          "gigachat2max": (650 / 81, 650 / 81), "gemini25flash": (0.30, 2.50)}


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return e


def bootstrap_f1(y, p, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = [f1_score(y[i], p[i], labels=CLASSES, average="macro", zero_division=0)
            for i in (rng.randint(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


def evaluate(path):
    d = pd.read_json(path, lines=True)
    model, mode, subset = re.fullmatch(r"preds_(.+)_(zeroshot|fewshot|supervised)_(cmp|full_test)", path.stem).groups()
    y = d.y_true.values
    p = d.y_pred.fillna(INVALID).values
    valid = p != INVALID
    f1s = f1_score(y, p, labels=CLASSES, average=None, zero_division=0)
    lo, hi = bootstrap_f1(y, p)
    conf = d.confidence.astype(float)
    has_conf = valid & conf.notna().values
    row = {
        "model": model, "mode": mode, "subset": subset, "n": len(d),
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, labels=CLASSES, average="macro", zero_division=0),
        "macro_f1_ci_lo": lo, "macro_f1_ci_hi": hi,
        **{f"f1_{c}": v for c, v in zip(CLASSES, f1s)},
        "mcc": matthews_corrcoef(y, p),
        "balanced_acc": balanced_accuracy_score(y, p),
        "invalid_rate": 1 - valid.mean(),
        "ece": ece(conf[has_conf], (y == p)[has_conf]) if has_conf.any() else np.nan,
        "pred_dist": json.dumps(pd.Series(p).value_counts().to_dict()),
        "latency_p50_s": d.latency_s.median(),
        "latency_p95_s": d.latency_s.quantile(0.95),
        "tokens_in_mean": d.tokens_in.mean() if "tokens_in" in d else np.nan,
        "tokens_out_mean": d.tokens_out.mean() if "tokens_out" in d else np.nan,
    }
    pin, pout = PRICES.get(model, (0.0, 0.0))
    row["api_cost_usd_per_1000"] = 1000 * (np.nan_to_num(row["tokens_in_mean"]) * pin +
                                           np.nan_to_num(row["tokens_out_mean"]) * pout) / 1e6
    cm = confusion_matrix(y, p, labels=CLASSES + [INVALID])[:len(CLASSES)]
    return row, cm


def main():
    rows = []
    figs = []
    for path in sorted(RES.glob("preds_*.jsonl")):
        row, cm = evaluate(path)
        rows.append(row)
        figs.append((f"{row['model']} | {row['mode']} | {row['subset']}", cm))
    t = pd.DataFrame(rows).sort_values(["subset", "macro_f1"], ascending=[True, False])
    t.to_csv(RES / "metrics.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        print(t.drop(columns=["pred_dist"]).to_string(index=False))

    n = len(figs)
    fig, axes = plt.subplots((n + 3) // 4, 4, figsize=(16, 3.6 * ((n + 3) // 4)), squeeze=False)
    for ax, (title, cm) in zip(axes.flat, figs):
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
        ax.set_xticks(range(5), CLASSES + [INVALID], rotation=30, fontsize=7)
        ax.set_yticks(range(4), CLASSES, fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("predicted", fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RES / "confusion_matrices.png", dpi=120)


if __name__ == "__main__":
    main()
