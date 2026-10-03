"""Классический baseline без LLM: логистическая регрессия на статистиках лог-мел/MFCC (те же 10 с).

Обучение на train, подбор C на val, предсказания на cmp_test и full_test в том же формате, что у LLM.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
RES = ROOT / "results"


def main():
    df = pd.read_csv(P / "dataset_final.csv")
    z = np.load(P / "features.npz")
    feats = pd.DataFrame(z["X"], index=z["clip_id"])
    X = lambda d: feats.loc[d.clip_id].values
    tr, va, te = (df[df.split == s] for s in ["train", "val", "test"])

    best = None
    for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        m.fit(X(tr), tr.label)
        f = f1_score(va.label, m.predict(X(va)), average="macro")
        print(f"C={C} val macroF1={f:.3f}")
        if best is None or f > best[0]:
            best = (f, C, m)
    f, C, m = best
    RES.mkdir(exist_ok=True)
    for subset, d in [("cmp", te[te.in_cmp_test]), ("full_test", te)]:
        t0 = time.perf_counter()
        proba = m.predict_proba(X(d))
        dt = (time.perf_counter() - t0) / len(d)
        pred = m.classes_[proba.argmax(1)]
        with (RES / f"preds_logreg_mfcc_supervised_{subset}.jsonl").open("w") as fh:
            for cid, yt, yp, c in zip(d.clip_id, d.label, pred, proba.max(1)):
                fh.write(json.dumps({"clip_id": int(cid), "y_true": yt, "y_pred": yp, "confidence": float(c),
                                     "raw": "", "latency_s": dt, "tokens_in": None, "tokens_out": None}) + "\n")
    print("best C", C, "val macroF1", round(f, 3))


if __name__ == "__main__":
    main()
