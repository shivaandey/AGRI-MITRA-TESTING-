FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime deps:
# - libsndfile1: required by `soundfile`
# - libgomp1: required by some ML wheels (OpenMP)
# - ffmpeg: useful for audio formats used by STT
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
