# Agri Mitra

A Flask-based agricultural advisory application that provides weather-based irrigation recommendations and fertilizer pricing information.

## Features

- Real-time weather data integration
- Crop-specific irrigation advice
- Soil photo scan (camera/gallery) for soil-type-based water guidance
- Government fertilizer pricing
- Multi-language support (English/Telugu)
- Mobile-responsive web interface

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   WEATHER_API_KEY=your_openweathermap_api_key
   GOVT_API_KEY=your_govt_api_key
   OFFLINE_MODE=1  # optional: disables external APIs for localhost testing
   ```

4. Run locally:
   ```bash
   python app.py
   ```

## Deployment to Vercel

1. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. Deploy:
   ```bash
   vercel
   ```

3. Set environment variables in Vercel dashboard:
   - `WEATHER_API_KEY`
   - `GOVT_API_KEY`

If you see a Vercel error page with an ID like `bom1::...`, it usually means the serverless function returned a 500. Check Vercel "Functions" logs and confirm `WEATHER_API_KEY` is set correctly.

## API Endpoints

- `POST /api/analyze`: Analyze field conditions and provide recommendations
  - Parameters: `lat`, `lon`, `crop`, `lang`
  - Response includes: `rain_expected_3d` and `fertilizer_advice`
- `POST /api/soil-scan`: Upload a soil photo and get soil type + water recommendation
  - Form-data: `image` (file), `crop` (optional), `lang` (optional), `temp` (optional)
- `POST /api/intent`: Intent detection (trainable) for voice assistant
  - JSON: `{ "text": "...", "lang": "auto|en|te|hi|..." }`
- `POST /api/assistant`: Voice assistant orchestration (intent + action)
  - JSON: `{ "text": "...", "lat": ..., "lon": ..., "crop": "...", "lang": "..." }`
  - or Form-data: `audio` (file), `lat` (optional), `lon` (optional), `crop` (optional), `lang` (optional)
- `POST /api/stt` (local/OCI only): Speech-to-text via faster-whisper
  - Form-data: `audio` (file), `lang` (optional)
- `POST /api/tts` (local/OCI only): Text-to-speech via Piper (returns WAV)
  - JSON: `{ "text": "...", "lang": "auto|en|te|hi|..." }`

## Training (Intent Model)

To train the intent model used by `/api/intent` and `/api/assistant`, see `ml/README.md`.

## Technologies Used

- Flask
- OpenWeatherMap API
- Google Translate API
- Tailwind CSS
