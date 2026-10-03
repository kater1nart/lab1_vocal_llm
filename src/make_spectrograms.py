"""Мел-спектрограммы клипов (PNG для VLM) + аудио-признаки для baseline и проверки почти-дубликатов.

Единый формат изображения для всех моделей: лог-мел (128 полос, 22.05 кГц, 0-8 кГц),
первые 10 секунд клипа, 448x224 px, colormap magma, ось частот подписана (Гц).
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MP3 = ROOT / "data" / "raw" / "mp3"
SPEC = ROOT / "data" / "processed" / "spec"
FEAT = ROOT / "data" / "processed" / "features.npz"
SR, DUR, N_MELS, FMAX = 22050, 10.0, 128, 8000


def render(path_png, S_db):
    fig = plt.figure(figsize=(4.48, 2.24), dpi=100)
    ax = fig.add_axes([0.11, 0.1, 0.88, 0.88])
    librosa.display.specshow(S_db, sr=SR, x_axis=None, y_axis="mel", fmax=FMAX, cmap="magma", ax=ax)
    ax.set_ylabel("")
    ax.tick_params(labelsize=6)
    fig.savefig(path_png)
    plt.close(fig)


def process(row):
    import librosa.display  # noqa: F401  (нужно в дочернем процессе)
    clip_id, mp3_path = row
    try:
        y, _ = librosa.load(MP3 / mp3_path, sr=SR, mono=True, duration=DUR)
        if len(y) < SR:  # меньше секунды — битый файл
            return clip_id, None
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, fmax=FMAX)
        S_db = librosa.power_to_db(S, ref=np.max)
        png = SPEC / f"{clip_id}.png"
        if not png.exists():
            render(png, S_db)
        mfcc = librosa.feature.mfcc(S=S_db, n_mfcc=20)
        feat = np.concatenate([S_db.mean(1), S_db.std(1), mfcc.mean(1), mfcc.std(1)]).astype(np.float32)
        return clip_id, feat
    except Exception as e:  # битые mp3 в MTAT встречаются
        print("ERR", clip_id, e, file=sys.stderr)
        return clip_id, None


def main():
    SPEC.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "processed" / "labels_split.csv")
    rows = list(zip(df.clip_id, df.mp3_path))
    ids, feats, bad = [], [], []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for k, (cid, ft) in enumerate(ex.map(process, rows, chunksize=16)):
            if ft is None:
                bad.append(int(cid))
            else:
                ids.append(cid)
                feats.append(ft)
            if k % 500 == 0:
                print(k, flush=True)
    np.savez(FEAT, clip_id=np.array(ids), X=np.stack(feats))
    (ROOT / "data" / "processed" / "bad_clips.json").write_text(json.dumps(bad))
    print("ok", len(ids), "bad", len(bad))


if __name__ == "__main__":
    main()
