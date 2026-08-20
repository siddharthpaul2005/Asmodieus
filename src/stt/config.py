import os
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    raise ValueError("SARVAM_API_KEY is not set in the environment or .env file")

SARVAM_REALTIME_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
SARVAM_REST_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL_REALTIME = "saaras:v3-realtime"
SARVAM_MODEL_REST = "saaras:v3"

LANGUAGE_CODE = "auto"
SILENCE_DURATION_MS = 700
MIN_SPEECH_DURATION_MS = 200
HIGH_VAD_SENSITIVITY = True
