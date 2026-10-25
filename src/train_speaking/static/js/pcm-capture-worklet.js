// AudioWorklet: downsample the microphone to 16 kHz mono int16 and emit fixed-size frames.
// Box-filter averaging over each output period doubles as a simple anti-aliasing filter.

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetRate = opts.targetRate || 16000;
    this.frameSamples = opts.frameSamples || 512;
    this.ratio = sampleRate / this.targetRate;
    this.frame = new Int16Array(this.frameSamples);
    this.filled = 0;
    this.acc = 0;
    this.accCount = 0;
    this.phase = 0;
    this.energy = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const channel = input[0];
    for (let i = 0; i < channel.length; i++) {
      this.acc += channel[i];
      this.accCount += 1;
      this.phase += 1;
      if (this.phase >= this.ratio) {
        this.phase -= this.ratio;
        const value = Math.max(-1, Math.min(1, this.acc / this.accCount));
        this.acc = 0;
        this.accCount = 0;
        this.energy += value * value;
        this.frame[this.filled++] = value * 32767;
        if (this.filled === this.frameSamples) {
          const rms = Math.sqrt(this.energy / this.frameSamples);
          this.port.postMessage({ pcm: this.frame.buffer, rms }, [this.frame.buffer]);
          this.frame = new Int16Array(this.frameSamples);
          this.filled = 0;
          this.energy = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor('pcm-capture', PcmCaptureProcessor);
