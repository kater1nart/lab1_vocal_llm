"""Выборочная загрузка только размеченных клипов MTAT из zip-зеркала на Hugging Face (HTTP Range).

Зеркало: https://huggingface.co/datasets/confit/magnatagatune (annotations совпадают с оригиналом побайтно).
Порядок: test -> val -> train, чтобы эксперимент с LLM можно было начать раньше.
"""
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
MP3 = ROOT / "data" / "raw" / "mp3"
URL = "https://huggingface.co/datasets/confit/magnatagatune/resolve/main/mp3.zip"
local = threading.local()


def zipfile():
    if not hasattr(local, "z"):
        local.z = RemoteZip(requests.head(URL, allow_redirects=True).url)
    return local.z


def fetch(name):
    dst = MP3 / name
    if dst.exists() and dst.stat().st_size > 0:
        return True
    for _ in range(3):
        try:
            data = zipfile().read(name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            return True
        except Exception as e:  # истёкшая подпись редиректа / сетевой сбой — переоткрываем
            print("retry", name, type(e).__name__, file=sys.stderr)
            if hasattr(local, "z"):
                del local.z
    return False


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "labels_split.csv")
    order = {"test": 0, "val": 1, "train": 2}
    names = df.assign(o=df.split.map(order)).sort_values("o").mp3_path.tolist()
    ok = 0
    with ThreadPoolExecutor(8) as ex:
        for k, r in enumerate(ex.map(fetch, names)):
            ok += r
            if k % 250 == 0:
                print(k, ok, flush=True)
    print("done", ok, "of", len(names))


if __name__ == "__main__":
    main()
