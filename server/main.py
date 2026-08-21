import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from src.stt.sarvam_stt import SarvamRealtimeSTT

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/stt")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_queue = asyncio.Queue()

    # Generator that yields chunks from the queue for Sarvam
    async def audio_stream_generator():
        while True:
            chunk = await audio_queue.get()
            if chunk is None: # EOF
                break
            yield chunk

    stt = SarvamRealtimeSTT()

    async def receive_from_client():
        try:
            while True:
                # Client (browser) sends raw Int16 PCM bytes
                data = await websocket.receive_bytes()
                await audio_queue.put(data)
        except WebSocketDisconnect:
            print("Client disconnected.")
        finally:
            await audio_queue.put(None) # Signal EOF to generator

    async def send_to_client():
        try:
            async for transcript in stt.stream(audio_stream_generator()):
                await websocket.send_json({
                    "text": transcript.text,
                    "is_partial": transcript.is_partial,
                    "latency_ms": transcript.latency_ms
                })
        except Exception as e:
            print(f"STT Stream error: {e}")
        finally:
            try:
                await websocket.close()
            except:
                pass

    # Run both tasks concurrently
    recv_task = asyncio.create_task(receive_from_client())
    send_task = asyncio.create_task(send_to_client())

    await asyncio.gather(recv_task, send_task, return_exceptions=True)
