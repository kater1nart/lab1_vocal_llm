"""Формирование меток типа фронт-вокала из тегов MagnaTagATune и разбиение train/val/test.

Классы: male, female, group (хор/дуэт/смешанный), instrumental.
Клипы без вокальных тегов исключаются (в MTAT отсутствие тега != отрицательная метка).
Клипы с противоречивыми тегами (вокал + "no vocals") исключаются.
Разбиение группируется по артисту, чтобы клипы одного исполнителя/трека не попадали в разные части.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SEED = 42

MALE = ["male", "male vocal", "male vocals", "male voice", "man", "man singing", "male singer", "male opera", "men"]
FEMALE = ["female", "female vocal", "female vocals", "female voice", "woman", "woman singing",
          "female singer", "female opera", "female singing", "women"]
GROUP = ["choir", "choral", "duet", "voices"]
INSTR = ["no vocal", "no vocals", "no voice", "no voices", "no singing", "no singer", "instrumental"]
CLASSES = ["male", "female", "group", "instrumental"]


def any_tag(df, tags):
    return df[tags].max(axis=1).astype(bool)


def main():
    ann = pd.read_csv(RAW / "annotations_final.csv", sep="\t")
    info = pd.read_csv(RAW / "clip_info_final.csv", sep="\t")
    df = ann[["clip_id", "mp3_path"]].merge(info[["clip_id", "title", "artist", "album", "track_number"]], on="clip_id")

    m, f, g, i = (any_tag(ann, t).values for t in (MALE, FEMALE, GROUP, INSTR))
    vocal = m | f | g
    label = np.full(len(ann), None, dtype=object)
    label[m & ~f & ~g & ~i] = "male"
    label[f & ~m & ~g & ~i] = "female"
    label[((m & f) | g) & ~i] = "group"
    label[i & ~vocal] = "instrumental"
    df["label"] = label

    stats = {
        "clips_total": int(len(df)),
        "no_vocal_tags": int((~vocal & ~i).sum()),
        "conflict_vocal_and_instrumental": int((vocal & i).sum()),
        "empty_mp3_path": int(df["mp3_path"].isna().sum()),
    }
    df = df[df["label"].notna() & df["mp3_path"].notna()].copy()

    # Дубликаты: одинаковый mp3_path или одинаковые (artist, title, segment) — известная проблема MTAT.
    stats["dup_mp3_path"] = int(df["mp3_path"].duplicated().sum())
    df = df.drop_duplicates("mp3_path")
    df["track_key"] = (df["artist"].str.lower() + "|" + df["album"].str.lower() + "|" + df["title"].str.lower())
    stats["labeled_clips"] = int(len(df))
    stats["unique_tracks"] = int(df["track_key"].nunique())
    stats["unique_artists"] = int(df["artist"].nunique())
    stats["class_counts"] = df["label"].value_counts().to_dict()
    stats["clips_per_track_mean"] = float(df.groupby("track_key").size().mean())

    # Разбиение по артистам: ~70/15/15. Сначала отделяем test (1/7 ~ 15%), затем val из остатка.
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    rest_idx, test_idx = next(sgkf.split(df, df["label"], groups=df["artist"]))
    df["split"] = "train"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"
    rest = df.iloc[rest_idx]
    sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
    tr, va = next(sgkf2.split(rest, rest["label"], groups=rest["artist"]))
    df.loc[rest.index[va], "split"] = "val"

    # Проверка утечки по группам.
    arts = {s: set(df.loc[df.split == s, "artist"]) for s in ["train", "val", "test"]}
    stats["artist_overlap"] = {
        "train_val": len(arts["train"] & arts["val"]),
        "train_test": len(arts["train"] & arts["test"]),
        "val_test": len(arts["val"] & arts["test"]),
    }
    stats["split_class_counts"] = df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(orient="index")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "labels_split.csv", index=False)
    (OUT / "data_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
