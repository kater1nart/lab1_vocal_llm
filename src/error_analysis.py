"""Анализ ошибок на full_test: лучшая LLM vs baseline.

- распределение предсказаний (коллапс в один класс?);
- точность в жанровых подгруппах (rock / opera / classical / electronic) — проверка сцепления класса с жанром;
- согласие LLM и baseline;
- 10 характерных ошибок LLM (с метаданными клипа).
Вывод: results/error_analysis.md
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LLM = sys.argv[1] if len(sys.argv) > 1 else "gemma3_4b_fewshot"


def main():
    meta = pd.read_csv(ROOT / "data" / "processed" / "dataset_final.csv")
    ann = pd.read_csv(ROOT / "data" / "raw" / "annotations_final.csv", sep="\t",
                      usecols=["clip_id", "rock", "opera", "classical", "electronic", "pop"])
    llm = pd.read_json(RES / f"preds_{LLM}_full_test.jsonl", lines=True)
    base = pd.read_json(RES / "preds_logreg_mfcc_supervised_full_test.jsonl", lines=True)
    d = (llm[["clip_id", "y_true", "y_pred", "confidence", "raw"]]
         .merge(base[["clip_id", "y_pred"]].rename(columns={"y_pred": "y_base"}), on="clip_id")
         .merge(meta[["clip_id", "artist", "title"]], on="clip_id").merge(ann, on="clip_id"))
    d["y_pred"] = d.y_pred.fillna("invalid")
    d["ok_llm"] = d.y_true == d.y_pred
    d["ok_base"] = d.y_true == d.y_base

    out = [f"# Анализ ошибок: {LLM} vs LogReg (full_test, n={len(d)})\n"]
    out.append("## Распределение предсказаний LLM по истинным классам\n")
    out.append(pd.crosstab(d.y_true, d.y_pred, margins=True).to_markdown() + "\n")
    out.append("## Уверенность LLM\n")
    out.append(d.groupby("ok_llm").confidence.describe()[["count", "mean", "min", "max"]].round(3).to_markdown() + "\n")
    out.append("## Accuracy в жанровых подгруппах\n")
    rows = []
    for g in ["rock", "opera", "classical", "electronic", "pop"]:
        for flag in [1, 0]:
            s = d[d[g] == flag]
            rows.append({"подгруппа": f"{g}={flag}", "n": len(s), "acc_llm": s.ok_llm.mean(), "acc_baseline": s.ok_base.mean()})
    out.append(pd.DataFrame(rows).round(3).to_markdown(index=False) + "\n")
    out.append("## Согласие LLM и baseline\n")
    out.append(pd.crosstab(d.ok_llm.map({True: "LLM верно", False: "LLM ошибка"}),
                           d.ok_base.map({True: "baseline верно", False: "baseline ошибка"})).to_markdown() + "\n")
    out.append("## 10 характерных ошибок LLM\n")
    err = d[~d.ok_llm]
    sample = err.groupby(["y_true", "y_pred"]).head(2).sort_values(["y_true", "y_pred"]).head(10)
    out.append(sample[["clip_id", "artist", "title", "y_true", "y_pred", "confidence", "y_base"]].to_markdown(index=False) + "\n")
    (RES / "error_analysis.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
