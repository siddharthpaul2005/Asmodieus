// This AudioWorkletProcessor downsamples the native browser audio (e.g. 48kHz float32)
// down to 16kHz Int16 PCM, and posts it back to the main thread in chunks.

class PCMDownsamplerProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.targetSampleRate = 16000;
        this.chunkSize = 1600; // 100ms at 16kHz
        this.buffer = new Float32Array(this.chunkSize);
        this.bufferOffset = 0;
        this.lastSample = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input.length) return true;
        
        const channelData = input[0]; // mono
        
        // Calculate the downsampling ratio (e.g. 48000 / 16000 = 3)
        const ratio = sampleRate / this.targetSampleRate;
        
        for (let i = 0; i < channelData.length; i++) {
            // Simple decimation/interpolation would go here.
            // For a basic implementation, we just take every Nth sample if ratio is integer.
            // A more robust implementation would use a low-pass filter to prevent aliasing.
            
            // Increment logic to downsample
            this.lastSample += 1;
            if (this.lastSample >= ratio) {
                this.buffer[this.bufferOffset] = channelData[i];
                this.bufferOffset++;
                this.lastSample -= ratio;
            }

            if (this.bufferOffset >= this.chunkSize) {
                // Convert Float32 [-1.0, 1.0] to Int16 [-32768, 32767]
                const int16Buffer = new Int16Array(this.chunkSize);
                for (let j = 0; j < this.chunkSize; j++) {
                    let s = Math.max(-1, Math.min(1, this.buffer[j]));
                    int16Buffer[j] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                
                // Post the PCM buffer to the main thread
                this.port.postMessage(int16Buffer.buffer, [int16Buffer.buffer]);
                
                // Reset buffer
                this.bufferOffset = 0;
                this.buffer = new Float32Array(this.chunkSize);
            }
        }
        
        return true;
    }
}

registerProcessor('pcm-downsampler', PCMDownsamplerProcessor);
