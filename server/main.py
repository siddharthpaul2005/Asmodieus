import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _make_stt():
    """Lazily import and construct SarvamRealtimeSTT so a missing API key
    returns a descriptive error rather than crashing the whole server."""
    try:
        from src.stt.sarvam_stt import SarvamRealtimeSTT
        return SarvamRealtimeSTT(), None
    except Exception as e:
        return None, str(e)


def _get_guardrail():
    """Lazily import guardrail so an import error is surfaced cleanly."""
    try:
        from src.guardrails import run_all_guardrails
        return run_all_guardrails
    except Exception:
        return None

@app.get("/api/guardrails")
def get_guardrails():
    """Serve the centralized guardrails JSON file to the frontend."""
    project_root = Path(__file__).resolve().parent.parent
    guardrails_path = project_root / "data" / "guardrails.json"
    if not guardrails_path.exists():
        return JSONResponse(status_code=404, content={"error": "guardrails.json not found"})
    return FileResponse(guardrails_path)


@app.websocket("/ws/stt")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # ── Try to build the STT client ─────────────────────────────────────────
    stt, stt_error = _make_stt()
    if stt_error:
        print(f"STT INIT ERROR: {stt_error}")
        # API key missing or import error — tell the client immediately so it
        # can fall back to simulation instead of hanging.
        try:
            await websocket.send_json({
                "error": f"STT unavailable: {stt_error}",
                "stt_unavailable": True,
            })
        except Exception:
            pass
        await websocket.close()
        return

    run_all_guardrails = _get_guardrail()
    audio_queue: asyncio.Queue = asyncio.Queue()

    # Generator that feeds audio chunks into the Sarvam stream
    async def audio_stream_generator():
        while True:
            chunk = await audio_queue.get()
            if chunk is None:   # EOF sentinel
                break
            yield chunk

    async def receive_from_client():
        try:
            while True:
                data = await websocket.receive_bytes()
                await audio_queue.put(data)
        except WebSocketDisconnect:
            print("Client disconnected.")
        except Exception as e:
            print(f"Receive error: {e}")
        finally:
            await audio_queue.put(None)   # unblock the generator

    async def send_to_client():
        try:
            async for transcript in stt.stream(audio_stream_generator()):
                # ── Backend guardrail gate ──────────────────────────────────
                if not transcript.is_partial and run_all_guardrails:
                    guard = run_all_guardrails(transcript.text)
                    if guard.blocked:
                        await websocket.send_json({
                            "text": transcript.text,
                            "is_partial": False,
                            "latency_ms": transcript.latency_ms,
                            "guardrail": {
                                "blocked": True,
                                "tier": guard.tier,
                                "tier_label": guard.tier_label,
                                "category": guard.category,
                                "flagged_words": guard.flagged_words,
                                "latency_ms": guard.latency_ms,
                                "refusal": "Not ethically correct to answer this query.",
                            },
                        })
                        return   # Do NOT proceed to RAG
                # ────────────────────────────────────────────────────────────
                await websocket.send_json({
                    "text": transcript.text,
                    "is_partial": transcript.is_partial,
                    "latency_ms": transcript.latency_ms,
                })
        except Exception as e:
            print(f"STT stream error: {e}")
            # Inform the client so it can trigger simulation fallback
            try:
                await websocket.send_json({
                    "error": f"STT stream failed: {e}",
                    "stt_unavailable": True,
                })
            except Exception:
                pass

    recv_task = asyncio.create_task(receive_from_client())
    send_task = asyncio.create_task(send_to_client())
    await asyncio.gather(recv_task, send_task, return_exceptions=True)

