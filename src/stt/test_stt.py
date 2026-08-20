import asyncio
import sys
import queue
import sounddevice as sd
from src.stt import stream_transcribe

async def mic_audio_stream(sample_rate=16000, chunk_duration_ms=100):
    q = queue.Queue()
    loop = asyncio.get_event_loop()
    
    # 100ms * 16kHz = 1600 samples
    chunk_size = int(sample_rate * (chunk_duration_ms / 1000.0))

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    print(f"🎤 Microphone active! Speak now... (Press Ctrl+C to stop)")
    
    # 1 channel (mono), 16-bit PCM as expected by Sarvam
    with sd.RawInputStream(samplerate=sample_rate, blocksize=chunk_size,
                           dtype='int16', channels=1, callback=callback):
        while True:
            # Yield from queue asynchronously
            chunk = await loop.run_in_executor(None, q.get)
            yield chunk

async def main():
    print("Connecting to Sarvam STT Realtime...")
    try:
        async for transcript in stream_transcribe(mic_audio_stream()):
            text = transcript["text"]
            if transcript["is_partial"]:
                # \r moves cursor to start of line, \033[K clears the line
                print(f"\r\033[K[PARTIAL] {text}", end="", flush=True)
            else:
                print(f"\r\033[K[FINAL] {text} (Latency: {transcript['latency_ms']:.2f}ms)")
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError during streaming: {e}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExited.")
