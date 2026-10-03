"""Единый протокол запуска LLM для классификации типа фронт-вокала.

Пример:
  python src/run_llm.py --model qwen25vl7b --mode fewshot --subset cmp
Модели: qwen25vl7b, qwen25vl32b, gemma3_4b (локальные VLM, MLX, вход — PNG мел-спектрограммы),
        qwen2audio7b (локальная аудио-LLM, вход — тот же 10-секундный фрагмент аудио),
        gigachat2max (API Сбера, вход — PNG мел-спектрограммы; нужен GIGACHAT_CREDENTIALS в .env),
        gemini25flash (API, недоступен из РФ — не использовался).
Результат: results/preds_<model>_<mode>_<subset>.jsonl (по строке на клип, с raw-ответом и латентностью).
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
RES = ROOT / "results"
load_dotenv(ROOT / ".env")
CLASSES = ["male", "female", "group", "instrumental"]

TASK = (
    "You are a music-catalog annotator. Determine the FRONT (lead) vocal type of a music clip.\n"
    "Classes:\n"
    "- male: a single lead male voice\n"
    "- female: a single lead female voice\n"
    "- group: choir, duet or several singers / mixed male and female voices\n"
    "- instrumental: no vocals\n"
)
INPUT_DESC = {
    "image": "The clip is given as a log-mel spectrogram image (x: time, 10 s; y: frequency 0-8 kHz, mel scale; "
             "brighter = louder). Look for harmonic vocal formants, vibrato and pitch range "
             "(male fundamental ~85-180 Hz, female ~165-255 Hz).\n",
    "audio": "The clip is given as audio (first 10 seconds).\n",
}
ANSWER = ('Answer ONLY with JSON: {"label": "<male|female|group|instrumental>", "confidence": <0..1>}')


def build_prompt(kind, mode, shots):
    noun = "spectrogram" if kind == "image" else "audio clip"
    txt = TASK + INPUT_DESC[kind]
    if mode == "fewshot":
        txt += f"\nYou are given {len(shots) + 1} {noun}s. The first {len(shots)} are labeled reference examples:\n"
        for k, (lab, _) in enumerate(shots, 1):
            txt += f"{noun} {k}: {lab}\n"
        txt += f"Classify {noun} {len(shots) + 1}.\n"
    else:
        txt += f"Classify the {noun}.\n"
    return txt + ANSWER


def parse(text):
    """Возвращает (label|None, confidence|None). None => невалидный ответ."""
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None
    lab = str(obj.get("label", "")).strip().lower()
    if lab not in CLASSES:
        return None, None
    try:
        conf = float(obj.get("confidence"))
        conf = min(max(conf, 0.0), 1.0)
    except (TypeError, ValueError):
        conf = None
    return lab, conf


# ---------------- backends ----------------
class MLXVLM:
    kind = "image"

    def __init__(self, repo):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        self.model, self.processor = load(repo)
        self.config = load_config(repo)

    def __call__(self, prompt, media):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        p = apply_chat_template(self.processor, self.config, prompt, num_images=len(media))
        r = generate(self.model, self.processor, p, image=[str(m) for m in media], max_tokens=48,
                     temperature=0.0, verbose=False)
        return r.text, getattr(r, "prompt_tokens", None), getattr(r, "generation_tokens", None)


class Qwen2Audio:
    kind = "audio"

    def __init__(self, repo):
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(repo)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.float16).to("mps")
        self.model.eval()

    def __call__(self, prompt, media):
        import librosa
        sr = self.proc.feature_extractor.sampling_rate
        audios = [librosa.load(m, sr=sr, duration=10.0)[0] for m in media]
        content = [{"type": "audio", "audio_url": str(m)} for m in media] + [{"type": "text", "text": prompt}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                             tokenize=False)
        inp = self.proc(text=text, audio=audios, sampling_rate=sr, return_tensors="pt", padding=True).to("mps")
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=48, do_sample=False)
        gen = out[:, inp["input_ids"].shape[1]:]
        return (self.proc.batch_decode(gen, skip_special_tokens=True)[0], int(inp["input_ids"].shape[1]),
                int(gen.shape[1]))


class Gemini:
    kind = "image"

    def __init__(self, model_name):
        from google import genai
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.name = model_name

    def __call__(self, prompt, media):
        from google.genai import types
        parts = [types.Part.from_bytes(data=Path(m).read_bytes(), mime_type="image/png") for m in media]
        for attempt in range(6):
            try:
                r = self.client.models.generate_content(
                    model=self.name, contents=parts + [prompt],
                    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=64,
                                                       thinking_config=types.ThinkingConfig(thinking_budget=0)))
                u = r.usage_metadata
                return r.text or "", u.prompt_token_count, u.candidates_token_count
            except Exception as e:  # 429 / 503 на бесплатном tier
                wait = 2 ** attempt * 5
                print("retry", attempt, type(e).__name__, wait, flush=True)
                time.sleep(wait)
        return "", None, None


class GigaChatVLM:
    """GigaChat API: одно изображение на сообщение, поэтому few-shot подаётся диалогом
    (эталон -> ответ ассистента с меткой), а финальное сообщение содержит ту же инструкцию и целевую спектрограмму."""
    kind = "image"
    shot_labels = []

    def __init__(self, model_name):
        from gigachat import GigaChat
        self.client = GigaChat(credentials=os.environ["GIGACHAT_CREDENTIALS"], scope="GIGACHAT_API_PERS",
                               model=model_name, ca_bundle_file=str(ROOT / "certs" / "russian_trusted_root_ca.pem"),
                               timeout=120)
        self.cache = {}

    def upload(self, path):
        with open(path, "rb") as fh:
            return self.client.upload_file(fh, purpose="general").id_

    def __call__(self, prompt, media):
        from gigachat.models import Chat, Messages, MessagesRole
        *shots, target = media
        msgs = []
        for k, (path, lab) in enumerate(zip(shots, self.shot_labels), 1):
            if path not in self.cache:
                self.cache[path] = self.upload(path)
            msgs.append(Messages(role=MessagesRole.USER, content=f"Reference spectrogram {k}.",
                                 attachments=[self.cache[path]]))
            msgs.append(Messages(role=MessagesRole.ASSISTANT, content=json.dumps({"label": lab, "confidence": 1.0})))
        for attempt in range(6):
            fid = None
            try:
                fid = self.upload(target)
                msgs_q = msgs + [Messages(role=MessagesRole.USER, content=prompt, attachments=[fid])]
                r = self.client.chat(Chat(messages=msgs_q, temperature=0.0, max_tokens=48))
                u = r.usage
                return r.choices[0].message.content, u.prompt_tokens, u.completion_tokens
            except Exception as e:  # 429 (один поток у физлиц) / сетевые сбои
                wait = 2 ** attempt * 3
                print("retry", attempt, type(e).__name__, str(e)[:120], wait, flush=True)
                time.sleep(wait)
            finally:
                if fid:
                    try:
                        self.client.delete_file(fid)
                    except Exception:
                        pass
        return "", None, None


MODELS = {
    "gigachat2max": lambda: GigaChatVLM("GigaChat-2-Max"),
    "qwen25vl7b": lambda: MLXVLM("mlx-community/Qwen2.5-VL-7B-Instruct-4bit"),
    "qwen25vl32b": lambda: MLXVLM("mlx-community/Qwen2.5-VL-32B-Instruct-4bit"),
    "gemma3_4b": lambda: MLXVLM("mlx-community/gemma-3-4b-it-4bit"),
    "qwen2audio7b": lambda: Qwen2Audio("Qwen/Qwen2-Audio-7B-Instruct"),
    "gemini25flash": lambda: Gemini("gemini-2.5-flash"),
}


def media_path(kind, clip_id, mp3_path):
    if kind == "image":
        return P / "spec" / f"{clip_id}.png"
    return ROOT / "data" / "raw" / "mp3" / mp3_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--mode", default="fewshot", choices=["zeroshot", "fewshot"])
    ap.add_argument("--subset", default="cmp", choices=["cmp", "full_test"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(P / "dataset_final.csv")
    meta = json.loads((P / "eval_sets.json").read_text())
    data = df[df.in_cmp_test] if args.subset == "cmp" else df[df.split == "test"]
    data = data.sample(frac=1.0, random_state=0)  # фиксированный порядок, классы перемешаны
    if args.limit:
        data = data.head(args.limit)

    backend = MODELS[args.model]()
    by_id = df.set_index("clip_id")
    shots = []
    if args.mode == "fewshot":
        for lab, cid in meta["few_shot_clip_ids"].items():
            shots.append((lab, media_path(backend.kind, cid, by_id.loc[cid, "mp3_path"])))
    backend.shot_labels = [s[0] for s in shots]
    prompt = build_prompt(backend.kind, args.mode, shots)

    RES.mkdir(exist_ok=True)
    out = RES / f"preds_{args.model}_{args.mode}_{args.subset}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["clip_id"] for l in out.read_text().splitlines() if l.strip()}
    with out.open("a") as fh:
        for k, row in enumerate(data.itertuples()):
            if row.clip_id in done:
                continue
            media = [s[1] for s in shots] + [media_path(backend.kind, row.clip_id, row.mp3_path)]
            t0 = time.perf_counter()
            text, tin, tout = backend(prompt, media)
            dt = time.perf_counter() - t0
            lab, conf = parse(text)
            fh.write(json.dumps({"clip_id": int(row.clip_id), "y_true": row.label, "y_pred": lab, "confidence": conf,
                                 "raw": text, "latency_s": dt, "tokens_in": tin, "tokens_out": tout}) + "\n")
            fh.flush()
            if k % 20 == 0:
                print(k, row.label, lab, f"{dt:.1f}s", repr(text[:60]), flush=True)
    print("saved", out)


if __name__ == "__main__":
    main()
