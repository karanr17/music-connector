// Downmixes the tab's audio to mono and hands it to the offscreen page in ~100 ms chunks.
// Small chunks keep audio close in time to the player events sent alongside it, so little
// of one song leaks into the next song's first window.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.size = Math.round(sampleRate / 10);
    this.buf = new Float32Array(this.size);
    this.n = 0;
  }

  process(inputs) {
    const channels = inputs[0];
    if (!channels || channels.length === 0) return true;
    const frames = channels[0].length;
    for (let i = 0; i < frames; i++) {
      let sum = 0;
      for (let c = 0; c < channels.length; c++) sum += channels[c][i];
      this.buf[this.n++] = sum / channels.length;
      if (this.n === this.size) {
        this.port.postMessage(this.buf.buffer, [this.buf.buffer]);
        this.buf = new Float32Array(this.size);
        this.n = 0;
      }
    }
    return true;
  }
}

registerProcessor('mc-capture', Capture);
