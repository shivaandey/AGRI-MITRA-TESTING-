import os
import time
from datetime import datetime, timedelta, timezone
import subprocess
import tempfile

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

try:
    import joblib  # type: ignore
except Exception:
    joblib = None

try:
    from PIL import Image, ImageStat  # type: ignore
except Exception:
    Image = None
    ImageStat = None

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None

# Load local env vars for development. On Vercel, configure env vars in
# Project Settings -> Environment Variables.
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

# Serve `public/` for local testing (`python app.py`), so you can open http://127.0.0.1:5000/
# On Vercel, static files are served by routing rules in `vercel.json`.
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
app = Flask(__name__, static_folder=os.path.join(_BASE_DIR, "public"), static_url_path="")
CORS(app)


def _json_error(message, status_code=500, **extra):
    payload = {"error": message}
    payload.update(extra)
    return jsonify(payload), status_code


_WHISPER_MODEL = None
_WHISPER_MODEL_ID = None


def _get_whisper_model():
    global _WHISPER_MODEL, _WHISPER_MODEL_ID
    if WhisperModel is None:
        return None, "faster-whisper not installed"

    model_id = os.getenv("WHISPER_MODEL", "small")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    key = f"{model_id}|{device}|{compute_type}"
    if _WHISPER_MODEL is not None and _WHISPER_MODEL_ID == key:
        return _WHISPER_MODEL, None

    try:
        _WHISPER_MODEL = WhisperModel(model_id, device=device, compute_type=compute_type)
        _WHISPER_MODEL_ID = key
        return _WHISPER_MODEL, None
    except Exception as exc:
        _WHISPER_MODEL = None
        _WHISPER_MODEL_ID = None
        return None, str(exc)


_INTENT_PIPELINE = None
_INTENT_PIPELINE_ID = None


def _get_intent_pipeline():
    global _INTENT_PIPELINE, _INTENT_PIPELINE_ID
    if joblib is None:
        return None, "joblib not installed"

    default_path = os.path.join(os.path.dirname(__file__), "models", "intent.joblib")
    model_path = os.getenv("INTENT_MODEL_PATH") or default_path
    try:
        st = os.stat(model_path)
    except FileNotFoundError:
        return None, f"intent model not found at {model_path}"
    except Exception as exc:
        return None, str(exc)

    key = f"{model_path}|{st.st_mtime_ns}|{st.st_size}"
    if _INTENT_PIPELINE is not None and _INTENT_PIPELINE_ID == key:
        return _INTENT_PIPELINE, None

    try:
        loaded = joblib.load(model_path)
        pipe = loaded.get("pipeline") if isinstance(loaded, dict) else loaded
        if pipe is None or not hasattr(pipe, "predict"):
            return None, "intent model is invalid (missing pipeline.predict)"
        _INTENT_PIPELINE = pipe
        _INTENT_PIPELINE_ID = key
        return _INTENT_PIPELINE, None
    except Exception as exc:
        _INTENT_PIPELINE = None
        _INTENT_PIPELINE_ID = None
        return None, str(exc)


def _keyword_intent(text):
    t = (text or "").strip().lower()
    if not t:
        return "unknown"

    if any(x in t for x in ("hello", "hi", "hey", "namaste", "नमस्ते", "నమస్తే", "ನಮಸ್ಕಾರ")):
        return "greeting"
    if any(x in t for x in ("help", "what can you do", "how to use", "guide")):
        return "help"

    if any(
        x in t
        for x in (
            "fertilizer",
            "fertiliser",
            "urea",
            "dap",
            "npk",
            "price",
            "prices",
            "rate",
            "rates",
            "ఎరువు",
            "ఎరువుల",
        )
    ):
        return "fertilizer_prices"

    if any(
        x in t
        for x in (
            "weather",
            "rain",
            "irrigat",
            "water",
            "watering",
            "నీరు",
            "వర్ష",
        )
    ):
        return "analyze_weather"

    return "unknown"


def _predict_intent(text):
    raw = (text or "").strip()
    if not raw:
        return {"intent": "unknown", "confidence": 0.0, "method": "empty"}

    pipe, err = _get_intent_pipeline()
    if pipe is not None and not err:
        try:
            intent = pipe.predict([raw])[0]
            confidence = None
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba([raw])[0]
                try:
                    confidence = float(max(proba))
                except Exception:
                    confidence = None
            return {"intent": str(intent), "confidence": confidence, "method": "ml"}
        except Exception:
            pass

    intent = _keyword_intent(raw)
    return {
        "intent": intent,
        "confidence": 0.35 if intent != "unknown" else 0.0,
        "method": "rules",
        "model_error": err,
    }


def _get_api_keys():
    return os.getenv("GOVT_API_KEY"), os.getenv("WEATHER_API_KEY")


def _is_offline_mode():
    return os.getenv("OFFLINE_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def _pick_lang(requested_lang):
    # Prefer explicit `lang` from the client, else use browser locale from Accept-Language.
    if requested_lang and str(requested_lang).strip().lower() not in ("auto", "default"):
        raw = str(requested_lang).strip()
        lower = raw.lower()
        # Preserve special BCP-47 tags used by some Indian languages in Google Translate (e.g. Manipuri).
        if lower in ("mni-mtei",):
            return "mni-Mtei"
        # Otherwise keep base language.
        return lower.split("-")[0]

    header = request.headers.get("Accept-Language", "")
    if not header:
        return "en"

    # Example header: "en-IN,en;q=0.9,te;q=0.8"
    first = header.split(",")[0].split(";")[0].strip().lower()
    if not first:
        return "en"

    if first in ("mni-mtei",):
        return "mni-Mtei"
    return first.split("-")[0]


_IN_STATE_TO_LANG = {
    "Andhra Pradesh": "te",
    "Telangana": "te",
    "Tamil Nadu": "ta",
    "Karnataka": "kn",
    "Kerala": "ml",
    "Maharashtra": "mr",
    "Gujarat": "gu",
    "Punjab": "pa",
    "Odisha": "or",
    "West Bengal": "bn",
    "Assam": "bn",
    "Bihar": "hi",
    "Chhattisgarh": "hi",
    "Delhi": "hi",
    "Goa": "hi",
    "Haryana": "hi",
    "Himachal Pradesh": "hi",
    "Jharkhand": "hi",
    "Madhya Pradesh": "hi",
    "Rajasthan": "hi",
    "Uttar Pradesh": "hi",
    "Uttarakhand": "hi",
    "Andaman and Nicobar Islands": "hi",
    "Arunachal Pradesh": "en",
    "Chandigarh": "hi",
    "Dadra and Nagar Haveli and Daman and Diu": "gu",
    "Jammu and Kashmir": "ur",
    "Ladakh": "hi",
    "Lakshadweep": "ml",
    "Puducherry": "ta",
    "Sikkim": "ne",
    "Tripura": "bn",
    "Meghalaya": "en",
    "Manipur": "mni-Mtei",
    "Mizoram": "en",
    "Nagaland": "en",
}


def _infer_lang_from_place(lat, lon, weather_key):
    # Uses OpenWeather reverse geocoding to infer state/country.
    # If this fails for any reason, return None and fall back to browser language.
    try:
        url = (
            "https://api.openweathermap.org/geo/1.0/reverse"
            f"?lat={lat}&lon={lon}&limit=1&appid={weather_key}"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        items = resp.json()
        if not isinstance(items, list) or not items:
            return None
        item = items[0] or {}
        country = (item.get("country") or "").upper()
        state = item.get("state")
        if country == "IN" and state:
            return _IN_STATE_TO_LANG.get(state)
        return None
    except Exception:
        return None


def _client_ip():
    # Vercel typically provides `x-forwarded-for` with a comma-separated list: client, proxy1, proxy2...
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return (
        request.headers.get("x-real-ip")
        or request.headers.get("cf-connecting-ip")
        or request.remote_addr
    )


def _infer_place_from_ip():
    # Best-effort GeoIP fallback when browser geolocation is unavailable.
    # Note: This is approximate and depends on the user's IP.
    ip = _client_ip()
    if not ip:
        return None

    try:
        # ipapi supports both `/json/` (caller IP) and `/<ip>/json/`.
        resp = requests.get(f"https://ipapi.co/{ip}/json/", timeout=5)
        if resp.status_code != 200:
            return None
        data = resp.json() or {}

        lat = data.get("latitude")
        lon = data.get("longitude")
        if lat is None or lon is None:
            return None

        return {
            "lat": float(lat),
            "lon": float(lon),
            "city": data.get("city"),
            "region": data.get("region"),
            "country": data.get("country_code") or data.get("country"),
            "source": "ip",
        }
    except Exception:
        return None


def _translate(text, lang):
    if not lang or lang == "en":
        return text
    # Normalize language codes for broader Indian language support.
    lang = str(lang).strip()
    lang_lower = lang.lower()
    # Common aliases and Google Translate specific codes.
    normalize = {
        "od": "or",  # Odia
        "kok": "gom",  # Konkani (Google uses 'gom')
        "konkani": "gom",
        "mni": "mni-Mtei",  # Manipuri
        "mni-mtei": "mni-Mtei",
        "panjabi": "pa",
        "punjabi": "pa",
        "urdu": "ur",
        "kashmiri": "ks",
        "santali": "sat",
        "bodo": "brx",
        "dogri": "doi",
    }
    lang = normalize.get(lang_lower, lang_lower)
    try:
        from deep_translator import GoogleTranslator  # type: ignore

        return GoogleTranslator(source="auto", target=lang).translate(text)
    except Exception:
        return text


_ADVICE_TEMPLATES = {
    "en": {
        "hot": "It is hot ({temp}°C). Apply 15-20 Liters of water today.",
        "fine": "Temperature is fine ({temp}°C). Apply 10 Liters of water.",
    },
    "te": {
        "hot": "వాతావరణం వేడిగా ఉంది ({temp}°C). ఈ రోజు 15-20 లీటర్ల నీరు ఇవ్వండి.",
        "fine": "ఉష్ణోగ్రత బాగుంది ({temp}°C). 10 లీటర్ల నీరు ఇవ్వండి.",
    },
    "hi": {
        "hot": "मौसम गर्म है ({temp}°C)। आज 15-20 लीटर पानी दें।",
        "fine": "तापमान ठीक है ({temp}°C)। 10 लीटर पानी दें।",
    },
    "ta": {
        "hot": "வானிலை சூடாக உள்ளது ({temp}°C). இன்று 15-20 லிட்டர் தண்ணீர் கொடுக்கவும்.",
        "fine": "வெப்பநிலை சரியாக உள்ளது ({temp}°C). 10 லிட்டர் தண்ணீர் கொடுக்கவும்.",
    },
    "kn": {
        "hot": "ಹವಾಮಾನ ಬಿಸಿ ಇದೆ ({temp}°C). ಇಂದು 15-20 ಲೀಟರ್ ನೀರು ಹಾಕಿ.",
        "fine": "ತಾಪಮಾನ ಚೆನ್ನಾಗಿದೆ ({temp}°C). 10 ಲೀಟರ್ ನೀರು ಹಾಕಿ.",
    },
    "ml": {
        "hot": "കാലാവസ്ഥ ചൂടാണ് ({temp}°C). ഇന്ന് 15-20 ലിറ്റർ വെള്ളം നൽകുക.",
        "fine": "താപനില നല്ലതാണ് ({temp}°C). 10 ലിറ്റർ വെള്ളം നൽകുക.",
    },
    "bn": {
        "hot": "আবহাওয়া গরম ({temp}°C)। আজ ১৫-২০ লিটার পানি দিন।",
        "fine": "তাপমাত্রা ঠিক আছে ({temp}°C)। ১০ লিটার পানি দিন।",
    },
    "mr": {
        "hot": "हवामान गरम आहे ({temp}°C). आज 15-20 लिटर पाणी द्या.",
        "fine": "तापमान ठीक आहे ({temp}°C). 10 लिटर पाणी द्या.",
    },
    "gu": {
        "hot": "હવામાન ગરમ છે ({temp}°C). આજે 15-20 લિટર પાણી આપો.",
        "fine": "તાપમાન ઠીક છે ({temp}°C). 10 લિટર પાણી આપો.",
    },
    "pa": {
        "hot": "ਮੌਸਮ ਗਰਮ ਹੈ ({temp}°C)। ਅੱਜ 15-20 ਲੀਟਰ ਪਾਣੀ ਦਿਓ।",
        "fine": "ਤਾਪਮਾਨ ਠੀਕ ਹੈ ({temp}°C)। 10 ਲੀਟਰ ਪਾਣੀ ਦਿਓ।",
    },
    "or": {
        "hot": "ପାଗ ଗରମ ଅଛି ({temp}°C)। ଆଜି 15-20 ଲିଟର ପାଣି ଦିଅନ୍ତୁ।",
        "fine": "ତାପମାତ୍ରା ଠିକ ଅଛି ({temp}°C)। 10 ଲିଟର ପାଣି ଦିଅନ୍ତୁ।",
    },
    "ur": {
        "hot": "موسم گرم ہے ({temp}°C)۔ آج 15-20 لیٹر پانی دیں۔",
        "fine": "درجہ حرارت ٹھیک ہے ({temp}°C)۔ 10 لیٹر پانی دیں۔",
    },
}


def _advice_text(temp, lang):
    try:
        temp = float(temp)
    except (TypeError, ValueError):
        temp = 28.0

    lang = (lang or "en").strip().lower()
    temp_str = f"{temp:.1f}".rstrip("0").rstrip(".")

    if temp >= 32:
        advice_en = f"It is hot ({temp_str}°C). Apply maximum water."
    elif temp <= 24:
        advice_en = f"It is cool ({temp_str}°C). Apply minimum water."
    else:
        advice_en = f"Temperature is fine ({temp_str}°C). Apply moderate water."

    return _translate(advice_en, lang)


def _classify_soil_from_photo(pil_img):
    # Lightweight heuristic classification (no ML model).
    img = pil_img.convert("RGB")
    img.thumbnail((128, 128))

    stat = ImageStat.Stat(img)
    r, g, b = stat.mean[:3]
    r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0

    brightness = (r_n + g_n + b_n) / 3.0
    max_c = max(r_n, g_n, b_n)
    min_c = min(r_n, g_n, b_n)
    saturation = 0.0 if max_c == 0 else (max_c - min_c) / max_c
    redness = r_n - (g_n + b_n) / 2.0

    if brightness < 0.25:
        return {"soil_type": "Black soil", "confidence": 0.8}
    if redness > 0.18 and r_n > g_n and r_n > b_n:
        return {"soil_type": "Red soil", "confidence": float(min(0.9, 0.6 + redness))}
    if brightness > 0.7 and saturation < 0.25:
        return {"soil_type": "Sandy soil", "confidence": 0.75}
    if saturation > 0.45 and 0.25 <= brightness <= 0.7:
        return {"soil_type": "Clay soil", "confidence": 0.65}
    return {"soil_type": "Loamy soil", "confidence": 0.6}


def _soil_water_liters_range(soil_type, crop, temp=None):
    base = {
        "Sandy soil": (12, 18),
        "Loamy soil": (10, 15),
        "Clay soil": (8, 12),
        "Black soil": (6, 10),
        "Red soil": (10, 15),
    }
    low, high = base.get(soil_type, (10, 15))

    crop_mul = {
        "Paddy": 1.4,
        "Wheat": 1.1,
        "Maize": 1.2,
        "Sugarcane": 2.2,
        "Sorghum (Jowar)": 1.1,
        "Pearl Millet (Bajra)": 1.0,
        "Finger Millet (Ragi)": 1.0,
        "Groundnut": 1.1,
        "Soybean": 1.1,
        "Mustard": 1.0,
        "Sunflower": 1.1,
        "Chickpea (Gram)": 1.0,
        "Pigeon Pea (Tur)": 1.0,
        "Lentil": 1.0,
        "Green Gram (Moong)": 1.0,
        "Black Gram (Urad)": 1.0,
        "Tea": 1.2,
        "Coffee": 1.2,
        "Rubber": 1.3,
        "Banana": 1.7,
        "Mango": 1.5,
        "Grapes": 1.6,
        "Apple": 1.4,
        "Orange": 1.4,
        "Onion": 1.2,
        "Potato": 1.2,
        "Tomato": 1.2,
        "Chilli": 1.2,
        "Turmeric": 1.1,
        "Ginger": 1.1,
        "Jute": 1.0,
        "Coconut": 1.8,
        "Cotton": 1.2,
    }.get(crop, 1.0)

    try:
        if temp is not None:
            temp = float(temp)
    except (TypeError, ValueError):
        temp = None

    if temp is not None and temp > 32:
        crop_mul *= 1.2
    if temp is not None and temp < 20:
        crop_mul *= 0.9

    low = int(round(low * crop_mul))
    high = int(round(high * crop_mul))
    if high < low:
        high = low
    return low, high


def _soil_irrigation_text(soil_category, crop, lang, temp=None, soil_display=None):
    low, high = _soil_water_liters_range(soil_category, crop, temp=temp)
    shown = soil_display or soil_category
    text_en = f"Detected {shown}. Recommended irrigation today: {low}-{high} liters."
    if soil_category == "Sandy soil":
        text_en += " Sandy soil drains fast; split watering into 2 times."
    if soil_category in ("Clay soil", "Black soil"):
        text_en += " Clay-rich soils hold water; avoid over-watering."
    if lang and lang != "en":
        return _translate(text_en, lang), low, high
    return text_en, low, high


def _soilgrids_texture(lat, lon):
    # SoilGrids gives sand/silt/clay estimates by coordinates (topsoil).
    # We use it to derive a texture class and a simple irrigation category.
    url = (
        "https://rest.isric.org/soilgrids/v2.0/properties/query"
        f"?lat={lat}&lon={lon}"
        "&property=sand&property=silt&property=clay"
        "&depth=0-5cm&value=mean"
    )
    try:
        resp = requests.get(url, timeout=10)
    except requests.RequestException as exc:
        return None, str(exc)

    try:
        data = resp.json()
    except ValueError:
        data = {}

    if resp.status_code != 200:
        return None, (data.get("message") if isinstance(data, dict) else None) or "soilgrids_error"

    def _mean(prop_name):
        try:
            prop = (data.get("properties") or {}).get(prop_name) or {}
            depths = prop.get("depths") or []
            if not depths:
                return None
            values = (depths[0] or {}).get("values") or {}
            val = values.get("mean")
            return float(val) if val is not None else None
        except Exception:
            return None

    sand = _mean("sand")
    silt = _mean("silt")
    clay = _mean("clay")
    if sand is None or silt is None or clay is None:
        return None, "soilgrids_missing_values"

    # SoilGrids returns g/kg typically. Convert to percent if needed.
    if sand > 100 or silt > 100 or clay > 100:
        sand = sand / 10.0
        silt = silt / 10.0
        clay = clay / 10.0

    # Normalize (best-effort).
    total = sand + silt + clay
    if total > 0:
        sand_n = sand * 100.0 / total
        silt_n = silt * 100.0 / total
        clay_n = clay * 100.0 / total
    else:
        sand_n, silt_n, clay_n = sand, silt, clay

    # Simple texture label (not full USDA triangle).
    if clay_n >= 35:
        texture = "Clay"
        category = "Clay soil"
    elif sand_n >= 70:
        texture = "Sandy"
        category = "Sandy soil"
    elif silt_n >= 50:
        texture = "Silt loam"
        category = "Loamy soil"
    else:
        texture = "Loam"
        category = "Loamy soil"

    return (
        {
            "texture_class": texture,
            "category": category,
            "sand_pct": round(sand_n, 1),
            "silt_pct": round(silt_n, 1),
            "clay_pct": round(clay_n, 1),
            "source": "soilgrids",
        },
        None,
    )


def _rain_forecast_next_event(lat, lon, weather_key, days=3):
    # Uses OpenWeather 5-day / 3-hour forecast to detect rain in the next `days`
    # and (if any) returns the first expected rain time + probability/intensity signals.
    url = (
        "https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={lat}&lon={lon}&appid={weather_key}&units=metric"
    )
    try:
        resp = requests.get(url, timeout=10)
    except requests.RequestException as exc:
        return None, str(exc)

    try:
        data = resp.json()
    except ValueError:
        data = {}

    if resp.status_code != 200:
        return None, (data.get("message") or data.get("cod") or "forecast_error")

    tz_offset = 0
    try:
        tz_offset = int(((data.get("city") or {}).get("timezone")) or 0)
    except Exception:
        tz_offset = 0

    now_utc = time.time()
    end_utc = now_utc + float(days) * 86400.0
    now_local_date = datetime.fromtimestamp(now_utc + tz_offset, tz=timezone.utc).date()
    items = data.get("list") or []

    def _pop_pct(raw):
        try:
            if raw is None:
                return None
            pop = float(raw)
            if pop <= 1.0:
                pop *= 100.0
            return int(round(max(0.0, min(100.0, pop))))
        except (TypeError, ValueError):
            return None

    def _rain_intensity_class(mm_per_hr):
        if mm_per_hr is None:
            return None
        try:
            r = float(mm_per_hr)
        except (TypeError, ValueError):
            return None
        # Common hourly-rate classification.
        if r < 2.5:
            return "light"
        if 2.6 <= r <= 7.5:
            return "moderate"
        if r > 7.5:
            return "heavy"
        return "light"

    pop_max_pct = 0
    for item in items:
        dt = item.get("dt")
        if not isinstance(dt, (int, float)):
            continue
        if dt > end_utc:
            break
        if dt < now_utc - 3600:
            continue

        pop_pct = _pop_pct(item.get("pop"))
        if pop_pct is not None and pop_pct > pop_max_pct:
            pop_max_pct = pop_pct

        rain = item.get("rain") or {}
        if isinstance(rain, dict):
            amount_3h = rain.get("3h")
            amount_1h = rain.get("1h")
            amount = amount_3h if amount_3h is not None else (amount_1h if amount_1h is not None else 0)
            try:
                if float(amount) > 0:
                    window_h = 3.0 if amount_3h is not None else 1.0
                    mm_per_hr = float(amount) / window_h if window_h else None
                    local_dt = datetime.fromtimestamp(float(dt) + tz_offset, tz=timezone.utc)
                    day_offset = (local_dt.date() - now_local_date).days
                    return (
                        {
                            "expected": True,
                            "first_rain_dt_utc": int(dt),
                            "first_rain_dt_local_iso": local_dt.isoformat(),
                            "first_rain_day_offset": int(day_offset),
                            # OpenWeather returns rain volume in millimeters for the previous 1h/3h window.
                            "first_rain_mm": float(amount),
                            "first_rain_mm_window": "3h" if amount_3h is not None else "1h",
                            "first_rain_mm_per_hr": mm_per_hr,
                            "first_rain_intensity": _rain_intensity_class(mm_per_hr),
                            "first_rain_pop_pct": pop_pct,
                            "pop_max_pct": pop_max_pct,
                            "tz_offset_sec": tz_offset,
                        },
                        None,
                    )
            except (TypeError, ValueError):
                pass

        weather = item.get("weather") or []
        for w in weather if isinstance(weather, list) else []:
            main = str((w or {}).get("main") or "").lower()
            desc = str((w or {}).get("description") or "").lower()
            if "rain" in main or "rain" in desc or "drizzle" in main or "drizzle" in desc:
                local_dt = datetime.fromtimestamp(float(dt) + tz_offset, tz=timezone.utc)
                day_offset = (local_dt.date() - now_local_date).days
                return (
                    {
                        "expected": True,
                        "first_rain_dt_utc": int(dt),
                        "first_rain_dt_local_iso": local_dt.isoformat(),
                        "first_rain_day_offset": int(day_offset),
                        "first_rain_mm": None,
                        "first_rain_mm_window": None,
                        "first_rain_mm_per_hr": None,
                        "first_rain_intensity": None,
                        "first_rain_pop_pct": pop_pct,
                        "pop_max_pct": pop_max_pct,
                        "tz_offset_sec": tz_offset,
                    },
                    None,
                )

    return {"expected": False, "tz_offset_sec": tz_offset, "pop_max_pct": pop_max_pct}, None


def _pop_bucket_label_en(pop_pct):
    if pop_pct is None:
        return None
    try:
        p = int(pop_pct)
    except (TypeError, ValueError):
        return None
    if p < 10:
        return "Very low chance"
    if 10 <= p <= 20:
        return "Isolated/slight chance"
    if 30 <= p <= 50:
        return "Scattered showers"
    if 60 <= p <= 70:
        return "Numerous showers"
    if 80 <= p <= 100:
        return "Near certainty"
    if 21 <= p <= 29:
        return "Low-to-moderate chance"
    if 51 <= p <= 59:
        return "Moderate-to-high chance"
    if 71 <= p <= 79:
        return "Very high chance"
    return "Chance of rain"


def _rain_when_label_en(day_offset):
    if day_offset == 0:
        return "today"
    if day_offset == 1:
        return "tomorrow"
    if day_offset == 2:
        return "day after tomorrow"
    if day_offset is None:
        return "soon"
    if day_offset < 0:
        return "today"
    return f"in {day_offset} days"


def _day_part_en(hour):
    try:
        hour = int(hour)
    except Exception:
        return None

    if 5 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 16:
        return "afternoon"
    if 17 <= hour <= 20:
        return "evening"
    return "night"


def _rain_summary_text(next_rain, lang):
    if not next_rain:
        text_en = "Rain forecast is unavailable."
        return _translate(text_en, lang) if lang and lang != "en" else text_en

    if not next_rain.get("expected"):
        pop = next_rain.get("pop_max_pct")
        bucket = _pop_bucket_label_en(pop)
        if pop:
            bucket_text = f" ({bucket}, {int(pop)}%)" if bucket else f" ({int(pop)}%)"
        else:
            bucket_text = ""
        # Avoid claiming "no rain expected" when the API still reports a non-trivial precipitation probability.
        text_en = f"No rain shown in the forecast for the next 3 days.{bucket_text}"
        return _translate(text_en, lang) if lang and lang != "en" else text_en

    when = _rain_when_label_en(next_rain.get("first_rain_day_offset"))
    local_iso = next_rain.get("first_rain_dt_local_iso")
    time_part = ""
    day_part = None
    try:
        if local_iso:
            local_dt = datetime.fromisoformat(local_iso)
            time_part = local_dt.strftime("%H:%M")
            day_part = _day_part_en(local_dt.hour)
    except Exception:
        time_part = ""
        day_part = None

    pop = next_rain.get("first_rain_pop_pct") or next_rain.get("pop_max_pct")
    bucket = _pop_bucket_label_en(pop)
    chance = ""
    if pop is not None:
        chance = f" ({bucket}, {int(pop)}%)" if bucket else f" ({int(pop)}%)"

    intensity = next_rain.get("first_rain_intensity")
    mm = next_rain.get("first_rain_mm")
    intensity_text = ""
    if intensity:
        intensity_text = f", {intensity} rain"
        if mm is not None:
            window = next_rain.get("first_rain_mm_window")
            if window:
                intensity_text += f" ({mm:.1f} mm/{window})"

    if day_part and time_part:
        text_en = f"Rain expected {when}{chance}{intensity_text} in the {day_part} around {time_part}."
    elif time_part:
        text_en = f"Rain expected {when}{chance}{intensity_text} around {time_part}."
    else:
        text_en = f"Rain expected {when}{chance}{intensity_text}."

    return _translate(text_en, lang) if lang and lang != "en" else text_en


def _fertilizer_rain_advice_text(next_rain, lang):
    if not next_rain:
        text_en = "Rain forecast is unavailable. Please follow local weather updates before fertilizing."
    elif next_rain.get("expected") is True:
        pop = next_rain.get("first_rain_pop_pct") or next_rain.get("pop_max_pct")
        intensity = next_rain.get("first_rain_intensity")

        when = _rain_when_label_en(next_rain.get("first_rain_day_offset"))
        local_iso = next_rain.get("first_rain_dt_local_iso")
        day_part = None
        time_part = None
        try:
            if local_iso:
                local_dt = datetime.fromisoformat(local_iso)
                day_part = _day_part_en(local_dt.hour)
                time_part = local_dt.strftime("%H:%M")
        except Exception:
            day_part = None
            time_part = None
        when_phrase = f"{when} in the {day_part}" if day_part else when
        if time_part:
            when_phrase = f"{when_phrase} around {time_part}"

        p = None
        try:
            if pop is not None:
                p = int(pop)
        except (TypeError, ValueError):
            p = None

        if p is not None and p >= 80:
            text_en = (
                f"Near certainty of rain {when_phrase} ({p}%). Do not fertilize now—apply after rain stops "
                "and the soil is workable."
            )
        elif p is not None and 60 <= p <= 70:
            text_en = (
                f"High likelihood of rain {when_phrase} ({p}%). Avoid fertilizing now—apply after rain stops "
                "to prevent nutrient wash-off."
            )
        elif p is not None and 30 <= p <= 50:
            text_en = (
                f"Moderate chance of scattered showers {when_phrase} ({p}%). If you must fertilize, use a smaller dose "
                "and avoid application right before rain."
            )
        elif p is not None and 10 <= p <= 20:
            text_en = (
                f"Slight chance of isolated showers {when_phrase} ({p}%). Fertilizing is generally okay; avoid if clouds build."
            )
        else:
            text_en = (
                f"Rain may occur {when_phrase}. To reduce nutrient loss, avoid fertilizing right before rainfall and prefer a dry window."
            )

        if intensity in ("heavy",):
            text_en += " Heavy rain risk: ensure drainage and avoid field operations during downpours."
    else:
        pop = next_rain.get("pop_max_pct")
        p = None
        try:
            if pop is not None:
                p = int(pop)
        except (TypeError, ValueError):
            p = None

        if p is not None and p >= 80:
            text_en = (
                f"Near certainty of precipitation in the forecast window ({p}%). Avoid fertilizing now; apply after a dry window."
            )
        elif p is not None and 60 <= p <= 70:
            text_en = (
                f"High likelihood of precipitation in the forecast window ({p}%). Prefer delaying fertilizing to reduce nutrient loss."
            )
        elif p is not None and 30 <= p <= 50:
            text_en = (
                f"Moderate chance of scattered showers in the forecast window ({p}%). Fertilize only if needed; avoid right before rain."
            )
        elif p is not None and 10 <= p <= 20:
            text_en = (
                f"Slight chance of isolated showers in the forecast window ({p}%). Fertilizing is generally okay; monitor conditions."
            )
        else:
            text_en = "No rain shown in the forecast for the next 3 days. Fertilizing is okay (as per crop stage)."

    if lang and lang != "en":
        return _translate(text_en, lang)
    return text_en


def _fertilizer_soil_note(soil_category, lang):
    if not soil_category:
        return ""
    if soil_category == "Sandy soil":
        text_en = "For sandy soil: split fertilizer into smaller doses to reduce leaching."
    elif soil_category in ("Clay soil", "Black soil"):
        text_en = "For clay-rich soil: avoid over-application; ensure drainage to prevent waterlogging."
    else:
        text_en = "For loamy soil: apply recommended dose based on crop stage."
    return _translate(text_en, lang) if lang and lang != "en" else text_en


def _combine_fertilizer_advice(rain_when_text, rain_advice, soil_note):
    parts = []
    if rain_when_text:
        parts.append(rain_when_text)
    if rain_advice:
        parts.append(rain_advice)
    if soil_note:
        parts.append(soil_note)
    return "\n".join([p for p in parts if p])


def _combine_irrigation_advice(climate_text, soil_text):
    parts = []
    if soil_text:
        parts.append(f"Soil: {soil_text}")
    if climate_text:
        parts.append(f"Weather: {climate_text}")
    return "\n".join([p for p in parts if p])


def _piper_speak_wav_bytes(text, lang=None):
    # Piper requires an installed binary + model file(s). This wrapper calls Piper via subprocess.
    # Env:
    # - PIPER_BIN: path to piper executable (e.g., ./piper or piper.exe)
    # - PIPER_MODEL: path to a .onnx model
    # - PIPER_CONFIG: optional path to a .json config (if required by your model build)
    piper_bin = os.getenv("PIPER_BIN")
    piper_model = os.getenv("PIPER_MODEL")
    piper_config = os.getenv("PIPER_CONFIG")

    if not piper_bin or not piper_model:
        return None, "Piper is not configured. Set PIPER_BIN and PIPER_MODEL."

    cmd = [piper_bin, "--model", piper_model]
    if piper_config:
        cmd.extend(["--config", piper_config])

    # Piper defaults to WAV on stdout in most builds when --output_file is omitted.
    # To be safe cross-platform, write to a temp file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        out_path = tmp.name

    cmd.extend(["--output_file", out_path])

    try:
        proc = subprocess.run(
            cmd,
            input=(text or "").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        if proc.returncode != 0:
            return None, (proc.stderr.decode("utf-8", errors="replace") or "piper_failed").strip()
        with open(out_path, "rb") as f:
            wav_bytes = f.read()
        return wav_bytes, None
    except Exception as exc:
        return None, str(exc)
    finally:
        try:
            os.remove(out_path)
        except Exception:
            pass


@app.errorhandler(Exception)
def _handle_unexpected_error(exc):
    # Ensure we always return JSON (no HTML error pages), and log details for Vercel logs.
    print(f"Unhandled error: {exc!r}")

    if isinstance(exc, HTTPException):
        return _json_error(exc.description, exc.code or 500)

    return _json_error("Internal server error", 500)


FERT_DATABASE = {
    "Paddy": [
        {"name": "Urea", "price": "242", "unit": "45kg"},
        {"name": "DAP", "price": "1350", "unit": "50kg"},
        {"name": "MOP", "price": "1700", "unit": "50kg"},
    ],
    "Coconut": [
        {"name": "Urea", "price": "242", "unit": "45kg"},
        {"name": "MOP", "price": "1700", "unit": "50kg"},
        {"name": "Complex (17:17:17)", "price": "1200", "unit": "50kg"},
    ],
    "Cotton": [
        {"name": "Urea", "price": "242", "unit": "45kg"},
        {"name": "NPK (19:19:19)", "price": "1150", "unit": "50kg"},
    ],
    "Vegetable": [
        {"name": "Urea", "price": "242", "unit": "45kg"},
        {"name": "NPK (19:19:19)", "price": "1150", "unit": "50kg"},
        {"name": "MOP", "price": "1700", "unit": "50kg"},
    ],
    "Pulse": [
        {"name": "DAP", "price": "1350", "unit": "50kg"},
        {"name": "NPK (19:19:19)", "price": "1150", "unit": "50kg"},
    ],
    "Oilseed": [
        {"name": "Urea", "price": "242", "unit": "45kg"},
        {"name": "DAP", "price": "1350", "unit": "50kg"},
        {"name": "NPK (19:19:19)", "price": "1150", "unit": "50kg"},
    ],
    "Generic": [
        {"name": "Urea", "price": "242", "unit": "45kg"},
        {"name": "DAP", "price": "1350", "unit": "50kg"},
        {"name": "MOP", "price": "1700", "unit": "50kg"},
        {"name": "NPK (19:19:19)", "price": "1150", "unit": "50kg"},
    ],
}


_CROP_FERTILIZER_PROGRAM = {
    # Cereals / staples
    "Paddy": "Paddy",
    "Wheat": "Paddy",
    "Maize": "Paddy",
    "Sugarcane": "Paddy",
    "Sorghum (Jowar)": "Paddy",
    "Pearl Millet (Bajra)": "Paddy",
    "Finger Millet (Ragi)": "Paddy",
    "Jute": "Paddy",
    # Oilseeds
    "Groundnut": "Oilseed",
    "Soybean": "Oilseed",
    "Mustard": "Oilseed",
    "Sunflower": "Oilseed",
    # Pulses
    "Chickpea (Gram)": "Pulse",
    "Pigeon Pea (Tur)": "Pulse",
    "Lentil": "Pulse",
    "Green Gram (Moong)": "Pulse",
    "Black Gram (Urad)": "Pulse",
    # Vegetables
    "Onion": "Vegetable",
    "Potato": "Vegetable",
    "Tomato": "Vegetable",
    "Chilli": "Vegetable",
    # Spices (closest fit)
    "Turmeric": "Oilseed",
    "Ginger": "Oilseed",
    # Plantation / fruits (closest fit)
    "Tea": "Coconut",
    "Coffee": "Coconut",
    "Rubber": "Coconut",
    "Banana": "Coconut",
    "Mango": "Coconut",
    "Grapes": "Coconut",
    "Apple": "Coconut",
    "Orange": "Coconut",
    # Fiber / commercial
    "Coconut": "Coconut",
    "Cotton": "Cotton",
}


def _normalize_crop(crop):
    if crop is None:
        return "Paddy"
    text = str(crop).strip()
    return text or "Paddy"


def _fertilizers_for_crop(crop):
    crop_norm = _normalize_crop(crop)
    program = _CROP_FERTILIZER_PROGRAM.get(crop_norm)
    ferts = (
        FERT_DATABASE.get(program) if program else None
    ) or FERT_DATABASE.get(crop_norm) or FERT_DATABASE.get("Generic") or FERT_DATABASE["Paddy"]
    return crop_norm, [dict(item) for item in (ferts or [])]


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    govt_key, weather_key = _get_api_keys()
    stt_model, stt_err = _get_whisper_model()
    tts_ready = bool(os.getenv("PIPER_BIN") and os.getenv("PIPER_MODEL"))
    intent_pipe, intent_err = _get_intent_pipeline()
    return jsonify(
        {
            "ok": True,
            "weather_key_set": bool(weather_key),
            "govt_key_set": bool(govt_key),
            "offline_mode": _is_offline_mode(),
            "stt_available": bool(stt_model) and not bool(stt_err),
            "stt_error": stt_err,
            "tts_available": tts_ready,
            "intent_available": bool(intent_pipe) and not bool(intent_err),
            "intent_error": intent_err,
        }
    )


@app.route("/api/intent", methods=["POST"])
def intent():
    # Intent detection for voice assistant. Accepts JSON: { text: "...", lang?: "..." }.
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if not text or not str(text).strip():
        return _json_error("Missing required field: text", 400)

    result = _predict_intent(str(text))
    return jsonify(
        {
            "text": str(text),
            "intent": result.get("intent"),
            "confidence": result.get("confidence"),
            "method": result.get("method"),
            "model_error": result.get("model_error"),
        }
    )


@app.route("/api/soil-scan", methods=["POST"])
def soil_scan():
    if Image is None or ImageStat is None:
        return _json_error("Server misconfigured: pillow is not installed", 500)

    crop = request.form.get("crop", "Paddy")
    requested_lang = request.form.get("lang")
    lang = _pick_lang(requested_lang)

    file = request.files.get("image")
    if not file:
        return _json_error("Missing required file: image", 400)

    try:
        pil_img = Image.open(file.stream)
    except Exception:
        return _json_error("Invalid image file", 400)

    photo_soil = _classify_soil_from_photo(pil_img)
    soil_display = photo_soil["soil_type"]
    soil_category = photo_soil["soil_type"]
    confidence = photo_soil["confidence"]
    soil_source = "photo_heuristic"
    soil_texture = None

    temp = request.form.get("temp")
    # Optional: try to use GPS-based texture from SoilGrids for more reliable classification.
    # This avoids guessing "alluvial" from photo alone.
    ip_place = None
    lat = request.form.get("lat")
    lon = request.form.get("lon")
    if (lat is None or lon is None) and not _is_offline_mode():
        ip_place = _infer_place_from_ip()
        if ip_place:
            lat, lon = ip_place.get("lat"), ip_place.get("lon")

    try:
        if lat is not None and lon is not None:
            lat = float(lat)
            lon = float(lon)
        else:
            lat = None
            lon = None
    except (TypeError, ValueError):
        lat = None
        lon = None

    if not _is_offline_mode() and lat is not None and lon is not None:
        sg, sg_err = _soilgrids_texture(lat, lon)
        if sg and not sg_err:
            soil_texture = sg
            soil_display = sg.get("texture_class") or soil_display
            soil_category = sg.get("category") or soil_category
            soil_source = "soilgrids"

    soil_irrigation, low, high = _soil_irrigation_text(
        soil_category, crop, lang, temp=temp, soil_display=soil_display
    )

    # Optional: also return current weather so the UI can keep the Weather card updated
    # even when the user only does a soil scan.
    weather_payload = {}
    weather_error = None
    next_rain = None
    rain_err = None
    rain_when = None
    fertilizer_advice = None
    climate_irrigation = None
    irrigation_combined = None
    # `ip_place`, `lat`, `lon` already computed above

    if not _is_offline_mode() and lat is not None and lon is not None:
        _, weather_key = _get_api_keys()
        if weather_key:
            w_url = (
                "https://api.openweathermap.org/data/2.5/weather"
                f"?lat={lat}&lon={lon}&appid={weather_key}&units=metric"
            )
            try:
                w_resp = requests.get(w_url, timeout=10)
                w_res = w_resp.json() if w_resp.headers.get("content-type", "").startswith("application/json") else {}
                if w_resp.status_code == 200 and isinstance(w_res, dict) and "main" in w_res:
                    temp_now = (w_res.get("main") or {}).get("temp")
                    hum_now = (w_res.get("main") or {}).get("humidity")
                    if temp_now is not None:
                        weather_payload["temp"] = temp_now
                    if hum_now is not None:
                        weather_payload["humidity"] = hum_now
                    weather_payload["location"] = {
                        "city": w_res.get("name") or ((ip_place or {}).get("city")),
                        "country": (w_res.get("sys") or {}).get("country") or ((ip_place or {}).get("country")),
                    }
                    climate_irrigation = _advice_text(temp_now, lang) if temp_now is not None else None
                else:
                    weather_error = (w_res.get("message") if isinstance(w_res, dict) else None) or "weather_error"
            except Exception as exc:
                weather_error = str(exc)
            if weather_error is None:
                next_rain, rain_err = _rain_forecast_next_event(lat, lon, weather_key, days=3)
                rain_when = _rain_summary_text(next_rain, lang) if rain_err is None else None
                fertilizer_rain_advice = _fertilizer_rain_advice_text(next_rain, lang)
                fertilizer_soil_note = _fertilizer_soil_note(soil_category, lang)
                fertilizer_advice = _combine_fertilizer_advice(
                    rain_when, fertilizer_rain_advice, fertilizer_soil_note
                )
        else:
            weather_error = "WEATHER_API_KEY_not_set"

    irrigation_combined = _combine_irrigation_advice(climate_irrigation, soil_irrigation)

    payload = {
        "soil_type": soil_display,
        "soil_category": soil_category,
        "soil_confidence": confidence,
        "water_liters_min": low,
        "water_liters_max": high,
        "soil_irrigation": soil_irrigation,
        "climate_irrigation": climate_irrigation,
        "irrigation_combined": irrigation_combined,
        "irrigation": irrigation_combined or soil_irrigation,
        "lang": lang,
        "soil_source": soil_source,
    }
    if soil_texture:
        payload["soil_texture"] = soil_texture
    payload.update(weather_payload)
    if weather_error:
        payload["weather_error"] = weather_error
    if rain_err:
        payload["rain_forecast_error"] = rain_err
    if next_rain is not None:
        payload["next_rain"] = next_rain
        payload["rain_expected_3d"] = bool(next_rain.get("expected"))
        payload["rain_when"] = rain_when
    if fertilizer_advice:
        payload["fertilizer_advice"] = fertilizer_advice

    return jsonify(payload)

def _compute_fertilizers_payload(crop, lang):
    crop_norm, ferts = _fertilizers_for_crop(crop)
    if not ferts:
        return None, ("No fertilizer pricing available for this crop", 404, {"crop": crop_norm})

    parts = []
    for f in ferts:
        name = (f or {}).get("name")
        price = (f or {}).get("price")
        unit = (f or {}).get("unit")
        if name and price and unit:
            parts.append(f"{name} {price} rupees per {unit}")
        elif name and price:
            parts.append(f"{name} {price} rupees")
        elif name:
            parts.append(str(name))

    text_en = f"Fertilizer prices for {crop_norm}: " + "; ".join(parts) + "."
    text = _translate(text_en, lang) if lang and lang != "en" else text_en
    return {"crop": crop_norm, "fertilizers": ferts, "text": text, "lang": lang}, None


@app.route("/api/fertilizers", methods=["POST"])
def fertilizers():
    # Crop-specific fertilizer price list (govt reference prices in this demo).
    # Accepts JSON: { crop: "...", lang?: "..." } and returns { crop, fertilizers, text }.
    data = request.get_json(silent=True) or {}
    crop = data.get("crop", "Paddy")
    requested_lang = data.get("lang")
    lang = _pick_lang(requested_lang) if requested_lang else "en"

    payload, err = _compute_fertilizers_payload(crop, lang)
    if err:
        msg, code, extra = err
        return _json_error(msg, code, **(extra or {}))
    return jsonify(payload)


def _stt_transcribe_filestorage(audio, requested_lang=None):
    model, err = _get_whisper_model()
    if err:
        return None, f"STT unavailable: {err}"

    lang = _pick_lang(requested_lang) if requested_lang else None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
        path = tmp.name
        audio.save(tmp)

    try:
        segments, info = model.transcribe(
            path,
            language=lang if lang and lang != "auto" else None,
            vad_filter=True,
        )
        text = "".join([seg.text for seg in segments]).strip()
        detected = getattr(info, "language", None) if info is not None else None
        prob = getattr(info, "language_probability", None) if info is not None else None
        return (
            {
                "text": text,
                "language": detected,
                "language_probability": prob,
                "engine": "faster-whisper",
            },
            None,
        )
    except Exception as exc:
        return None, f"STT failed: {exc}"
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


@app.route("/api/stt", methods=["POST"])
def stt():
    # Speech-to-text using faster-whisper. Accepts multipart/form-data with `audio` file.
    audio = request.files.get("audio")
    if not audio:
        return _json_error("Missing required file: audio", 400)

    requested_lang = request.form.get("lang") or request.args.get("lang")
    result, err = _stt_transcribe_filestorage(audio, requested_lang=requested_lang)
    if err:
        hint = None
        if "decod" in err.lower() or "ffmpeg" in err.lower():
            hint = "If this is an audio decoding error, install ffmpeg and try again."
        return _json_error(err, 500, hint=hint)
    return jsonify(result)


def _compute_analyze_payload(data):
    data = data or {}
    lat, lon = data.get("lat"), data.get("lon")
    crop = data.get("crop", "Paddy")
    requested_lang = data.get("lang")
    lang = _pick_lang(requested_lang)
    crop_norm, fertilizers = _fertilizers_for_crop(crop)
    soil_category = data.get("soil_category") or data.get("soil_type")
    ip_place = None

    if lat is None or lon is None:
        if _is_offline_mode():
            lat, lon = 0.0, 0.0
        else:
            ip_place = _infer_place_from_ip()
            if ip_place:
                lat, lon = ip_place["lat"], ip_place["lon"]
            else:
                return None, ("Enable location access or provide lat/lon", 400, None)

    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return None, ("lat and lon must be numbers", 400, None)

    if _is_offline_mode():
        temp = data.get("temp", 28.0)
        hum = data.get("humidity", 50)
        try:
            temp = float(temp)
            hum = int(hum)
        except (TypeError, ValueError):
            return None, ("temp and humidity must be numbers in offline mode", 400, None)

        climate_irrigation = _advice_text(temp, lang)
        soil_irrigation = None
        if soil_category:
            soil_irrigation, _, _ = _soil_irrigation_text(soil_category, crop_norm, lang, temp=temp)
        irrigation_combined = _combine_irrigation_advice(climate_irrigation, soil_irrigation)

        return (
            {
                "temp": temp,
                "humidity": hum,
                "irrigation": irrigation_combined or climate_irrigation,
                "soil_irrigation": soil_irrigation,
                "climate_irrigation": climate_irrigation,
                "irrigation_combined": irrigation_combined,
                "rain_expected_3d": None,
                "rain_when": _translate("Rain forecast is unavailable (offline mode).", lang),
                "fertilizer_advice": _translate(
                    "Rain forecast is unavailable (offline mode). Please enable forecast before fertilizing.",
                    lang,
                ),
                "rain_forecast_error": "offline_mode",
                "fertilizers": fertilizers,
                "crop": crop_norm,
                "offline": True,
                "lang": lang,
                "translated_by": "auto_translate",
                "location": ip_place,
            },
            None,
        )

    _, weather_key = _get_api_keys()
    if not weather_key:
        return None, ("Server misconfigured: WEATHER_API_KEY is not set", 500, None)

    if not requested_lang or str(requested_lang).strip().lower() in ("auto", "default"):
        browser_lang = _pick_lang(None)
        if browser_lang == "en":
            inferred = _infer_lang_from_place(lat, lon, weather_key)
            if inferred:
                lang = inferred

    w_url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={weather_key}&units=metric"
    )
    try:
        w_resp = requests.get(w_url, timeout=10)
    except requests.RequestException as exc:
        return None, ("Weather lookup failed", 502, {"details": str(exc)})

    try:
        w_res = w_resp.json()
    except ValueError:
        w_res = {"message": "Non-JSON response from weather service"}

    if w_resp.status_code != 200 or "main" not in w_res:
        details = w_res.get("message") or w_res.get("cod") or "Unknown error"
        return None, ("Weather lookup failed", 502, {"details": details})

    temp = w_res["main"].get("temp")
    hum = w_res["main"].get("humidity")
    if temp is None or hum is None:
        return None, ("Weather lookup failed: missing temp/humidity", 502, None)

    location = {
        "city": w_res.get("name") or ((ip_place or {}).get("city")),
        "country": (w_res.get("sys") or {}).get("country") or ((ip_place or {}).get("country")),
    }

    next_rain, rain_err = _rain_forecast_next_event(lat, lon, weather_key, days=3)
    rain_expected_3d = None if next_rain is None else bool(next_rain.get("expected"))
    rain_when = _rain_summary_text(next_rain, lang)

    fertilizer_rain_advice = _fertilizer_rain_advice_text(next_rain, lang)
    fertilizer_soil_note = _fertilizer_soil_note(soil_category, lang)
    fertilizer_advice = _combine_fertilizer_advice(rain_when, fertilizer_rain_advice, fertilizer_soil_note)

    climate_irrigation = _advice_text(temp, lang)
    soil_irrigation = None
    if soil_category:
        soil_irrigation, _, _ = _soil_irrigation_text(soil_category, crop_norm, lang, temp=temp)
    irrigation_combined = _combine_irrigation_advice(climate_irrigation, soil_irrigation)

    return (
        {
            "temp": temp,
            "humidity": hum,
            "irrigation": irrigation_combined or climate_irrigation,
            "soil_irrigation": soil_irrigation,
            "climate_irrigation": climate_irrigation,
            "irrigation_combined": irrigation_combined,
            "rain_expected_3d": rain_expected_3d,
            "rain_when": rain_when,
            "next_rain": next_rain,
            "fertilizer_advice": fertilizer_advice,
            "rain_forecast_error": rain_err,
            "fertilizers": fertilizers,
            "crop": crop_norm,
            "lang": lang,
            "translated_by": "auto_translate",
            "location": location,
        },
        None,
    )


@app.route("/api/tts", methods=["POST"])
def tts():
    # Text-to-speech using Piper. Accepts JSON: {text: "...", lang?: "..."} and returns WAV.
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if not text or not str(text).strip():
        return _json_error("Missing required field: text", 400)

    requested_lang = data.get("lang")
    lang = _pick_lang(requested_lang) if requested_lang else "en"

    wav_bytes, err = _piper_speak_wav_bytes(str(text), lang=lang)
    if err:
        return _json_error("TTS unavailable", 500, details=err)

    # Return raw WAV for playback in the browser.
    return (
        wav_bytes,
        200,
        {
            "Content-Type": "audio/wav",
            "Content-Disposition": "inline; filename=tts.wav",
        },
    )


@app.route("/api/assistant", methods=["POST"])
def assistant():
    # Voice assistant orchestration:
    # - Accepts either JSON { text, ... } or multipart/form-data with `audio`
    # - Detects intent (ML model if trained, else keyword fallback)
    # - Runs the matching app action and returns a TTS-friendly response text
    stt_meta = None

    if request.files.get("audio") is not None:
        audio = request.files.get("audio")
        if not audio:
            return _json_error("Missing required file: audio", 400)

        requested_lang = request.form.get("lang") or request.args.get("lang")
        stt_meta, stt_err = _stt_transcribe_filestorage(audio, requested_lang=requested_lang)
        if stt_err:
            hint = None
            if "decod" in stt_err.lower() or "ffmpeg" in stt_err.lower():
                hint = "If this is an audio decoding error, install ffmpeg and try again."
            return _json_error(stt_err, 500, hint=hint)

        text = (stt_meta or {}).get("text") or ""
        data = {
            "text": text,
            "lat": request.form.get("lat"),
            "lon": request.form.get("lon"),
            "crop": request.form.get("crop", "Paddy"),
            "lang": request.form.get("lang"),
            "soil_category": request.form.get("soil_category") or request.form.get("soil_type"),
            "temp": request.form.get("temp"),
            "humidity": request.form.get("humidity"),
        }
    else:
        data = request.get_json(silent=True) or {}
        text = data.get("text")
        if not text or not str(text).strip():
            return _json_error("Provide either JSON {text: ...} or upload audio as form-data field 'audio'", 400)
        text = str(text)

    lang = _pick_lang((data or {}).get("lang"))
    intent_res = _predict_intent(text)
    intent = intent_res.get("intent") or "unknown"

    action = None
    action_payload = None

    if intent == "fertilizer_prices":
        action = "fertilizers"
        crop = (data or {}).get("crop", "Paddy")
        fert_payload, err = _compute_fertilizers_payload(crop, lang)
        if err:
            msg, code, extra = err
            return _json_error(msg, code, **(extra or {}))
        action_payload = fert_payload
        response_text = (fert_payload or {}).get("text") or ""
    elif intent == "analyze_weather":
        action = "analyze"
        analyze_payload, err = _compute_analyze_payload(data)
        if err:
            msg, code, extra = err
            return _json_error(msg, code, **(extra or {}))
        action_payload = analyze_payload
        lang = (analyze_payload or {}).get("lang") or lang
        response_text = (
            (analyze_payload or {}).get("irrigation_combined")
            or (analyze_payload or {}).get("irrigation")
            or ""
        )
        fert_advice = (analyze_payload or {}).get("fertilizer_advice")
        if fert_advice:
            response_text = (response_text + " " + str(fert_advice)).strip()
    elif intent == "help":
        action = "help"
        response_text = _translate(
            "You can ask for irrigation advice (weather/rain/watering) or fertilizer prices (Urea/DAP/NPK).",
            lang,
        )
    elif intent == "greeting":
        action = "greeting"
        response_text = _translate(
            "Hello! Ask me about irrigation advice or fertilizer prices.",
            lang,
        )
    else:
        action = "unknown"
        response_text = _translate(
            "I didn't understand. Ask about irrigation advice (weather/rain/watering) or fertilizer prices (Urea/DAP/NPK).",
            lang,
        )

    return jsonify(
        {
            "text": text,
            "lang": lang,
            "intent": intent,
            "intent_confidence": intent_res.get("confidence"),
            "intent_method": intent_res.get("method"),
            "action": action,
            "response_text": response_text,
            "result": action_payload,
            "stt": stt_meta,
        }
    )


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    payload, err = _compute_analyze_payload(data)
    if err:
        msg, code, extra = err
        return _json_error(msg, code, **(extra or {}))
    return jsonify(payload)
