import os
from pathlib import Path
from dotenv import load_dotenv

# Always load .env from the project root (d:/Project/Asmodieus/.env),
# regardless of which directory uvicorn / Python is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    raise ValueError(
        f"SARVAM_API_KEY is not set. Looked for .env at: {_PROJECT_ROOT / '.env'}"
    )


SARVAM_REALTIME_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
SARVAM_REST_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL_REALTIME = "saaras:v3-realtime"
SARVAM_MODEL_REST = "saaras:v3"

LANGUAGE_CODE = "auto"
SILENCE_DURATION_MS = 2000
MIN_SPEECH_DURATION_MS = 200
HIGH_VAD_SENSITIVITY = False
