from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

@dataclass
class TranscriptEvent:
    text: str
    is_partial: bool
    language_code: str
    latency_ms: float
    request_id: str | None

class STTProvider(ABC):
    @abstractmethod
    async def stream(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[TranscriptEvent]:
        """Yields partial and final transcript events as audio streams in."""
        pass

    @abstractmethod
    async def transcribe_batch(self, audio_bytes: bytes, fmt: str = "wav") -> TranscriptEvent:
        """One-shot REST fallback."""
        pass
