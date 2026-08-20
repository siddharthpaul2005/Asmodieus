import asyncio
import json
import base64
import websockets
import os
from dotenv import load_dotenv

load_dotenv("d:/Copied/HHGOA/Asmodieus/.env")
api_key = os.getenv("SARVAM_API_KEY")

async def test_ws():
    url = "wss://api.sarvam.ai/speech-to-text-realtime/ws?model=saaras:v3-realtime&language_code=en-IN"
    headers = {"API-SUBSCRIPTION-KEY": api_key}
    
    print("Connecting...")
    async with websockets.connect(url, additional_headers=headers) as ws:
        print("Connected.")
        
        # Send one chunk of silence
        chunk = b"\x00" * 3200
        payload = {
            "event": "audio_input",
            "audio": base64.b64encode(chunk).decode("utf-8")
        }
        await ws.send(json.dumps(payload))
        
        # Wait for response
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                print("Received:", msg)
        except asyncio.TimeoutError:
            print("Timeout waiting for message.")
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    asyncio.run(test_ws())
