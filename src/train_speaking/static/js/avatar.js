// Animated SVG interviewer.
// Mouth shapes follow either the playback amplitude (default) or a server-provided
// viseme track aligned to the AudioContext clock. No camera input is used here.

const AMPLITUDE_SHAPES = [
  [0.012, 'rest'],
  [0.04, 'small'],
  [0.09, 'mid'],
  [Infinity, 'open'],
];
const SMOOTHING = 0.35;
const BLINK_MIN_MS = 3000;
const BLINK_MAX_MS = 6000;
const BLINK_DURATION_MS = 140;

export class Avatar {
  constructor(root) {
    this.root = root;
    this.shapes = new Map();
    root.querySelectorAll('.ts-mouth-shape').forEach((el) => this.shapes.set(el.dataset.viseme, el));
    this.eyelids = root.querySelector('#ts-eyelids');
    this.analyser = null;
    this.audioContext = null;
    this.samples = null;
    this.level = 0;
    this.visemes = [];
    this.current = null;
    this.setShape('rest');
    this.setState('idle');
    this.scheduleBlink();
    requestAnimationFrame(() => this.frame());
  }

  attachAudio(audioContext, analyser) {
    this.audioContext = audioContext;
    this.analyser = analyser;
    this.samples = new Float32Array(analyser.fftSize);
  }

  setState(state) {
    this.root.dataset.state = state;
  }

  addVisemes(track, startTime) {
    for (const [t, v] of track) this.visemes.push([startTime + t, v]);
    this.visemes.sort((a, b) => a[0] - b[0]);
  }

  clearVisemes() {
    this.visemes = [];
  }

  setShape(name) {
    if (name === this.current || !this.shapes.has(name)) return;
    this.shapes.forEach((el, key) => { el.style.display = key === name ? '' : 'none'; });
    this.current = name;
  }

  frame() {
    requestAnimationFrame(() => this.frame());
    const now = this.audioContext ? this.audioContext.currentTime : 0;
    if (this.visemes.length) {
      while (this.visemes.length > 1 && this.visemes[1][0] <= now) this.visemes.shift();
      const [t, v] = this.visemes[0];
      if (t <= now) {
        this.setShape(v);
        if (this.visemes.length === 1 && v === 'rest') this.visemes = [];
        return;
      }
    }
    if (!this.analyser) return this.setShape('rest');
    this.analyser.getFloatTimeDomainData(this.samples);
    let sum = 0;
    for (const s of this.samples) sum += s * s;
    const rms = Math.sqrt(sum / this.samples.length);
    this.level = this.level * (1 - SMOOTHING) + rms * SMOOTHING;
    const shape = AMPLITUDE_SHAPES.find(([limit]) => this.level < limit)[1];
    this.setShape(shape);
  }

  scheduleBlink() {
    const delay = BLINK_MIN_MS + Math.random() * (BLINK_MAX_MS - BLINK_MIN_MS);
    setTimeout(() => {
      if (this.eyelids) {
        this.eyelids.classList.add('ts-blink');
        setTimeout(() => this.eyelids.classList.remove('ts-blink'), BLINK_DURATION_MS);
      }
      this.scheduleBlink();
    }, delay);
  }
}
