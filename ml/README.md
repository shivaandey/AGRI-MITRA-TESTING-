# ML (Intent Training)

This project already supports:
- Speech-to-text via `POST /api/stt` (faster-whisper)
- Text-to-speech via `POST /api/tts` (Piper)

To make it a *voice assistant*, you typically also need **NLU** (intent detection) so the app can decide which action to run (weather/irrigation advice, fertilizer prices, etc.).

## 1) Add training data

Edit `ml/intent_data.jsonl` and add lines like:

```json
{"text":"Will it rain tomorrow?","intent":"analyze_weather"}
{"text":"Urea price","intent":"fertilizer_prices"}
```

Recommended intents used by the API:
- `analyze_weather`
- `fertilizer_prices`
- `greeting`
- `help`

## 2) Train

From repo root:

```bash
python ml/train_intent_model.py
```

This writes the model to `api/models/intent.joblib` (ignored by git).

## 3) Use

Run the server and call:
- `POST /api/intent` with `{ "text": "..." }`
- `POST /api/assistant` with either `{ "text": "..." }` or an `audio` upload

