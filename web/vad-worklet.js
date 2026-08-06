/* Mic framer — runs on the AUDIO RENDER THREAD.
 *
 * Replaces the ScriptProcessor this UI used to run on the main thread. That mattered:
 * the main thread also draws the animated orb and decodes incoming speech, and every
 * GC pause or long paint there dropped mic frames — which made the frame-counting
 * endpointer fire early and cut callers off mid-sentence.
 *
 * Emits one message per hop carrying three things:
 *   pcm  — the ORIGINAL-rate samples of this hop; this is what is buffered and sent
 *          to Whisper, so it is never resampled twice.
 *   f16k — the same audio as exactly 512 samples at 16 kHz. That is precisely the
 *          frame size and rate Silero VAD expects, so the main thread can hand it
 *          straight to ONNX with no reshaping.
 *   rms  — loudness, for the adaptive-threshold VAD used when Silero isn't vendored.
 *
 * 512 samples @ 16 kHz = 32 ms per hop.
 */
/* One RBJ-cookbook biquad. Two of these in series give the speech band.
 * Filter state has to persist across hops or every frame boundary clicks. */
class Biquad {
  constructor(kind, f0, fs, q = 0.707) {
    const w0 = 2 * Math.PI * f0 / fs, c = Math.cos(w0), a = Math.sin(w0) / (2 * q);
    let b0, b1, b2;
    if (kind === 'hp') { b0 = (1 + c) / 2; b1 = -(1 + c); b2 = (1 + c) / 2; }
    else               { b0 = (1 - c) / 2; b1 = 1 - c;    b2 = (1 - c) / 2; }
    const a0 = 1 + a, a1 = -2 * c, a2 = 1 - a;
    this.b0 = b0 / a0; this.b1 = b1 / a0; this.b2 = b2 / a0;
    this.a1 = a1 / a0; this.a2 = a2 / a0;
    this.x1 = this.x2 = this.y1 = this.y2 = 0;
  }
  step(x) {
    const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2
              - this.a1 * this.y1 - this.a2 * this.y2;
    this.x2 = this.x1; this.x1 = x; this.y2 = this.y1; this.y1 = y;
    return y;
  }
}

class VadFramer extends AudioWorkletProcessor {
  constructor() {
    super();
    // `sampleRate` is a global in AudioWorkletGlobalScope (usually 48000).
    this.ratio = sampleRate / 16000;
    this.acc = 0;            // fractional resampler accumulator
    this.prev = 0;           // previous input sample, for linear interpolation
    this.out = new Float32Array(512);
    this.oi = 0;
    this.nat = [];           // native-rate samples accumulated since the last hop
    this.natN = 0;
    // Speech band, for the LEVEL decision only. A fan at 50Hz and hiss at 8kHz both
    // push full-band RMS up without being speech, so the energy VAD had to gate on a
    // number that noise contributes to as much as a voice does. This copy is measured
    // and thrown away — Whisper only ever receives the untouched `pcm` below, because
    // filtering what the RECOGNISER sees measurably hurts it (0.10 WER vs 0.00, and
    // speaker similarity 0.596 vs 0.675, with a real denoiser in that position).
    // MEASURED separation gain from this cascade, speech vs noise-only level:
    //   fan / AC rumble 4.61x · high hiss 2.80x · traffic 2.09x · broadband hiss 1.66x
    this.hp = new Biquad('hp', 300, sampleRate);
    this.lp = new Biquad('lp', 3400, sampleRate);
    this.bandSum = 0;        // Σ x² of the filtered signal for this hop
    this.bandN = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;                      // mic not delivering yet — stay alive
    this.nat.push(new Float32Array(ch));
    this.natN += ch.length;
    for (let i = 0; i < ch.length; i++) {
      const s = ch[i];
      const band = this.lp.step(this.hp.step(s));   // speech band, decision copy only
      this.bandSum += band * band; this.bandN++;
      this.acc += 1;
      // Emit one 16 kHz sample every `ratio` input samples, interpolating across the
      // fractional boundary so 44.1 kHz hardware doesn't drift against 48 kHz.
      while (this.acc >= this.ratio) {
        this.acc -= this.ratio;
        const f = this.acc / this.ratio;       // 0..1 position between prev and s
        this.out[this.oi++] = this.prev * f + s * (1 - f);
        if (this.oi === 512) { this._emit(); this.oi = 0; }
      }
      this.prev = s;
    }
    return true;
  }

  _emit() {
    const pcm = new Float32Array(this.natN);
    let o = 0;
    for (const c of this.nat) { pcm.set(c, o); o += c.length; }
    this.nat = []; this.natN = 0;

    const f16k = this.out.slice(0);
    let sum = 0;
    for (let i = 0; i < f16k.length; i++) sum += f16k[i] * f16k[i];
    const rmsBand = this.bandN ? Math.sqrt(this.bandSum / this.bandN) : 0;
    this.bandSum = 0; this.bandN = 0;

    // Transfer (not copy) both buffers — zero-copy hand-off to the main thread.
    // `pcm` is UNFILTERED: it is what Whisper receives.
    this.port.postMessage(
      { pcm, f16k, rms: Math.sqrt(sum / f16k.length), rmsBand, ms: 32 },
      [pcm.buffer, f16k.buffer]
    );
  }
}

registerProcessor('vad-framer', VadFramer);
