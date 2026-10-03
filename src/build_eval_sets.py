"""Проверка почти-дубликатов по аудио-признакам и фиксация наборов для протокола сравнения.

- near-duplicates: косинусное сходство стандартизованных признаков > 0.995 между клипами разных split;
  такие пары удаляются из val/test (train не трогаем).
- cmp_test: стратифицированная подвыборка test (N_PER_CLASS на класс) — единый тест для сравнения LLM.
- full_test: весь test — «новый срез большего объема» для лучшей модели.
- few-shot: по одному фиксированному примеру каждого класса из train.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
SEED, N_PER_CLASS, THR = 42, 50, 0.995
CLASSES = ["male", "female", "group", "instrumental"]


def main():
    df = pd.read_csv(P / "labels_split.csv")
    z = np.load(P / "features.npz")
    feats = pd.DataFrame(z["X"], index=z["clip_id"])
    df = df[df.clip_id.isin(feats.index)].reset_index(drop=True)
    X = feats.loc[df.clip_id].values
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    split = df.split.values
    drop, pairs = set(), []
    for s in ["val", "test"]:
        q = np.where(split == s)[0]
        other = np.where(split != s)[0]
        sim = X[q] @ X[other].T
        for qi, oi in zip(*np.where(sim > THR)):
            pairs.append((int(df.clip_id[q[qi]]), int(df.clip_id[other[oi]]), float(sim[qi, oi])))
            drop.add(int(df.clip_id[q[qi]]))
    df = df[~df.clip_id.isin(drop)].reset_index(drop=True)

    rng = np.random.RandomState(SEED)
    test = df[df.split == "test"]
    cmp_ids = np.concatenate([rng.choice(test[test.label == c].clip_id, N_PER_CLASS, replace=False) for c in CLASSES])
    train = df[df.split == "train"]
    shots = {c: int(rng.choice(train[train.label == c].clip_id)) for c in CLASSES}

    df["in_cmp_test"] = df.clip_id.isin(cmp_ids)
    df.to_csv(P / "dataset_final.csv", index=False)
    meta = {
        "near_dup_threshold": THR,
        "near_dup_pairs_cross_split": len(pairs),
        "dropped_from_val_test": len(drop),
        "examples": pairs[:10],
        "final_counts": df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "cmp_test_size": int(len(cmp_ids)),
        "full_test_size": int((df.split == "test").sum()),
        "few_shot_clip_ids": shots,
    }
    (P / "eval_sets.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
