from .sarvam_stt import SarvamRealtimeSTT
from .base import STTProvider, TranscriptEvent

__all__ = ["SarvamRealtimeSTT", "STTProvider", "TranscriptEvent"]

# Convenience functions conforming to interface.py
async def stream_transcribe(audio_chunks):
    stt = SarvamRealtimeSTT()
    async for event in stt.stream(audio_chunks):
        yield {
            "text": event.text,
            "is_partial": event.is_partial,
            "language_code": event.language_code,
            "latency_ms": event.latency_ms,
            "request_id": event.request_id
        }

async def transcribe(audio_bytes, format="wav"):
    stt = SarvamRealtimeSTT()
    event = await stt.transcribe_batch(audio_bytes, format)
    return {
        "text": event.text,
        "is_partial": event.is_partial,
        "language_code": event.language_code,
        "latency_ms": event.latency_ms,
        "request_id": event.request_id
    }
