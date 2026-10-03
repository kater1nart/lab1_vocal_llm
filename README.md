# Лабораторная работа 1. LLM для разметки музыкального каталога по типу фронт-вокала

Дисциплина «Прикладные задачи машинного обучения». Тема ВКР: *Разработка системы автоматической разметки
музыкального каталога по типу фронт-вокала для задач поиска и рекомендаций.*

Задача: по лог-мел спектрограмме 10-секундного фрагмента определить тип ведущего вокала —
`male`, `female`, `group` (хор / дуэт / смешанный), `instrumental`.
Сравниваются мультимодальные LLM (VLM) в режимах zero-shot и few-shot по единому протоколу,
плюс обучаемый baseline без LLM.

## Данные

MagnaTagATune (Law et al., ISMIR 2009), лицензия CC BY-NC-SA 3.0.
Оригинал: https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset;
аудио взято из зеркала https://huggingface.co/datasets/confit/magnatagatune (файлы аннотаций побайтно совпадают с оригиналом).

## Воспроизведение

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv -r requirements.txt
# аннотации
curl -O https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv   # -> data/raw/
curl -O https://mirg.city.ac.uk/datasets/magnatagatune/clip_info_final.csv     # -> data/raw/
.venv/bin/python src/prepare_labels.py      # метки + split по артистам (70/15/15)
.venv/bin/python src/fetch_audio.py         # только размеченные клипы, ~0.8 ГБ
.venv/bin/python src/make_spectrograms.py   # PNG спектрограммы + акустические признаки
.venv/bin/python src/build_eval_sets.py     # почти-дубликаты, тестовые наборы, few-shot примеры
.venv/bin/python src/baseline.py            # LogReg на мел/MFCC
# LLM (локально через MLX на Apple Silicon; Gemini — нужен GEMINI_API_KEY в .env)
.venv/bin/python src/run_llm.py --model qwen25vl7b --mode fewshot --subset cmp
.venv/bin/python src/evaluate.py            # results/metrics.csv, results/confusion_matrices.png
```

Все случайные операции фиксированы `seed=42`, генерация LLM — `temperature=0`.

## Структура

- `src/` — код пайплайна;
- `data/processed/` — метки, split, статистика данных (`data_stats.json`, `eval_sets.json`);
- `results/` — сырые ответы моделей (`preds_*.jsonl`), метрики, графики;
- `report/` — отчёт.
