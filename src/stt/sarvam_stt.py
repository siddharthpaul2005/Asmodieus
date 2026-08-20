import asyncio
import base64
import json
import time
from typing import AsyncIterator
import websockets
from sarvamai import SarvamAI
import httpx

from .base import STTProvider, TranscriptEvent
from . import config

class SarvamRealtimeSTT(STTProvider):
    def __init__(self):
        self.api_key = config.SARVAM_API_KEY
        self.ws_url = f"{config.SARVAM_REALTIME_URL}?model={config.SARVAM_MODEL_REALTIME}&language_code={config.LANGUAGE_CODE}"
        
        # We can append VAD params to the URL if supported by the endpoint docs.
        # Assuming they can be passed as query params for this example.
        if config.HIGH_VAD_SENSITIVITY:
            self.ws_url += "&high_vad_sensitivity=true"
        else:
            self.ws_url += f"&silence_duration_ms={config.SILENCE_DURATION_MS}&min_speech_duration_ms={config.MIN_SPEECH_DURATION_MS}"

        self.rest_client = SarvamAI(api_subscription_key=self.api_key)

    async def stream(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[TranscriptEvent]:
        headers = {"API-SUBSCRIPTION-KEY": self.api_key}
        start_time = time.perf_counter()

        try:
            async with websockets.connect(self.ws_url, additional_headers=headers) as ws:
                
                async def sender():
                    try:
                        async for chunk in audio_chunks:
                            payload = {
                                "event": "audio_input",
                                "audio": base64.b64encode(chunk).decode("utf-8")
                            }
                            await ws.send(json.dumps(payload))
                    except websockets.exceptions.ConnectionClosed:
                        pass # Normal on shutdown

                async def receiver():
                    try:
                        async for message in ws:
                            response = json.loads(message)
                            
                            # DEBUG: See what Sarvam actually sends back!
                            # print(f"DEBUG SARVAM RECV: {response}") 

                            # Different API endpoints (realtime vs legacy) have different response formats
                            text = None
                            is_partial = True
                            
                            event_type = response.get("event", "")
                            
                            if event_type.startswith("transcript."):
                                text = response.get("text")
                                is_partial = (event_type == "transcript.partial")
                            elif "transcript" in response:
                                text = response["transcript"]
                                is_partial = response.get("is_partial", True)
                            elif response.get("type") == "data" and "data" in response:
                                text = response["data"].get("transcript")
                                is_partial = response["data"].get("is_partial", True)

                            if text:
                                lang = response.get("language_code", config.LANGUAGE_CODE)
                                latency = (time.perf_counter() - start_time) * 1000
                                req_id = response.get("request_id")
                                
                                yield TranscriptEvent(
                                    text=text,
                                    is_partial=is_partial,
                                    language_code=lang,
                                    latency_ms=latency,
                                    request_id=req_id
                                )
                    except websockets.exceptions.ConnectionClosed:
                        pass # Normal on shutdown

                # Run sender and receiver concurrently
                # Since receiver yields, we need a way to properly interleave.
                # A common pattern is to wrap them in tasks and use a queue if needed,
                # but websockets allows simple yielding in the same task if sender is a background task.
                
                sender_task = asyncio.create_task(sender())
                
                async for event in receiver():
                    yield event
                    
                sender_task.cancel()
                
        except Exception as e:
            print(f"STT Streaming Error: {e}")
            raise

    async def transcribe_batch(self, audio_bytes: bytes, fmt: str = "wav") -> TranscriptEvent:
        start_time = time.perf_counter()
        
        # Write bytes to a temporary file for the SDK, or use raw httpx.
        # Since SDK takes a file path or file-like object, we'll try httpx for memory-only operations.
        headers = {"api-subscription-key": self.api_key}
        data = {
            "model": config.SARVAM_MODEL_REST,
            "mode": "transcribe"
        }
        files = {
            "file": (f"audio.{fmt}", audio_bytes, f"audio/{fmt}")
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(config.SARVAM_REST_URL, headers=headers, data=data, files=files)
            response.raise_for_status()
            res_json = response.json()
            
            latency = (time.perf_counter() - start_time) * 1000
            
            return TranscriptEvent(
                text=res_json.get("transcript", ""),
                is_partial=False,
                language_code=res_json.get("language_code", "en-IN"),
                latency_ms=latency,
                request_id=res_json.get("request_id")
            )
