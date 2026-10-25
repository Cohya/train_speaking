// Browser media for the interview page: microphone capture, camera preview, interviewer
// audio playback, and the /ws/audio connection. The camera stream is only shown locally;
// it is never sent to the server.

import { Avatar } from './avatar.js';

const HEADER_BYTES = 12;
const TARGET_RATE = 16000;
const FRAME_SAMPLES = 512;
const SCHEDULE_AHEAD_S = 0.05;
const RECONNECT_DELAYS_MS = [500, 1000, 2000, 4000, 8000, 8000];
const AVATAR_STATES = {
  interviewer_speaking: 'speaking',
  wrapping_up: 'speaking',
  listening: 'listening',
  user_speaking: 'listening',
  processing: 'thinking',
  recovering: 'thinking',
  evaluating: 'thinking',
  paused: 'idle',
};

class InterviewMedia {
  init(options) {
    this.options = options;
    this.el = {
      video: document.getElementById('ts-camera'),
      meter: document.getElementById('ts-mic-level'),
      micStatus: document.getElementById('ts-mic-status'),
      speaking: document.getElementById('ts-speaking'),
      question: document.getElementById('ts-question'),
      interim: document.getElementById('ts-interim'),
      notice: document.getElementById('ts-notice'),
    };
    this.avatar = new Avatar(document.getElementById('ts-avatar-svg'));
    this.ctx = new AudioContext();
    this.output = this.ctx.createGain();
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.output.connect(this.analyser);
    this.analyser.connect(this.ctx.destination);
    this.avatar.attachAudio(this.ctx, this.analyser);
    this.utterances = new Map();
    this.chunkStarts = new Map();
    this.pendingVisemes = new Map();
    this.nextTime = 0;
    this.micPaused = true;
    this.devicesReady = false;
    this.closedByUs = false;
    this.reconnectAttempt = 0;
    this.connect();
    window.addEventListener('beforeunload', () => { this.closedByUs = true; this.ws && this.ws.close(); });
  }

  // ------------------------------------------------------------------ websocket

  connect() {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${scheme}://${location.host}/ws/audio/${this.options.sessionId}?token=${encodeURIComponent(this.options.token)}`;
    this.ws = new WebSocket(url);
    this.ws.binaryType = 'arraybuffer';
    this.ws.onopen = () => { this.reconnectAttempt = 0; this.notice(''); };
    this.ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) this.onAudio(event.data);
      else this.onMessage(JSON.parse(event.data));
    };
    this.ws.onclose = () => {
      if (this.closedByUs || this.finished) return;
      const delay = RECONNECT_DELAYS_MS[this.reconnectAttempt];
      if (delay === undefined) {
        this.notice('Connection lost. Reload the page to continue; your session is kept on the server.');
        return;
      }
      this.reconnectAttempt += 1;
      this.notice('Connection lost. Reconnecting...');
      setTimeout(() => this.connect(), delay);
    };
  }

  send(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(message));
  }

  onMessage(msg) {
    switch (msg.type) {
      case 'snapshot':
        if (msg.question) this.el.question.textContent = msg.question;
        this.applyState(msg.state);
        break;
      case 'state':
        this.applyState(msg.state);
        break;
      case 'interviewer_text':
        this.utterances.set(msg.utterance_id, { sources: new Set(), ended: false, started: false });
        if (msg.kind !== 'ack') this.el.question.textContent = msg.text;
        break;
      case 'utterance_end':
        this.onUtteranceEnd(msg);
        break;
      case 'stop_playback':
        this.stopPlayback();
        break;
      case 'visemes':
        this.onVisemes(msg);
        break;
      case 'vad':
        this.el.speaking.classList.toggle('active', !!msg.speaking);
        if (!msg.speaking) this.el.interim.textContent = '';
        break;
      case 'interim_transcript':
        this.el.interim.textContent = msg.text;
        break;
      case 'notice':
      case 'error':
        this.notice(msg.message);
        break;
      case 'completed':
        this.finished = true;
        this.stopDevices();
        break;
      default:
        break;
    }
  }

  applyState(state) {
    this.state = state;
    this.avatar.setState(AVATAR_STATES[state] || 'idle');
    const listening = state === 'listening' || state === 'user_speaking';
    this.el.micStatus.textContent = listening ? 'Your turn - speak naturally' : (state === 'paused' ? 'Paused' : 'Interviewer\'s turn');
    this.el.micStatus.classList.toggle('listening', listening);
  }

  notice(text) {
    this.el.notice.textContent = text || '';
    this.el.notice.style.display = text ? '' : 'none';
  }

  // ------------------------------------------------------------------ playback

  onAudio(buffer) {
    const view = new DataView(buffer);
    const utteranceId = view.getUint32(0, true);
    const seq = view.getUint32(4, true);
    const sampleRate = view.getUint32(8, true);
    const pcm = new Int16Array(buffer, HEADER_BYTES);
    if (!pcm.length) return;
    const utterance = this.utterances.get(utteranceId) || { sources: new Set(), ended: false, started: false };
    this.utterances.set(utteranceId, utterance);

    const audioBuffer = this.ctx.createBuffer(1, pcm.length, sampleRate);
    const channel = audioBuffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;
    const source = this.ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.output);
    const startAt = Math.max(this.ctx.currentTime + SCHEDULE_AHEAD_S, this.nextTime);
    source.start(startAt);
    this.nextTime = startAt + audioBuffer.duration;
    utterance.sources.add(source);
    this.chunkStarts.set(`${utteranceId}:${seq}`, startAt);
    const pending = this.pendingVisemes.get(`${utteranceId}:${seq}`);
    if (pending) { this.avatar.addVisemes(pending, startAt); this.pendingVisemes.delete(`${utteranceId}:${seq}`); }

    if (!utterance.started) {
      utterance.started = true;
      const delayMs = Math.max(0, (startAt - this.ctx.currentTime) * 1000);
      setTimeout(() => this.send({ type: 'playback_started', utterance_id: utteranceId }), delayMs);
    }
    source.onended = () => {
      utterance.sources.delete(source);
      this.maybeFinished(utteranceId);
    };
  }

  onVisemes(msg) {
    const key = `${msg.utterance_id}:${msg.seq}`;
    const start = this.chunkStarts.get(key);
    if (start !== undefined) this.avatar.addVisemes(msg.track, start);
    else this.pendingVisemes.set(key, msg.track);
  }

  onUtteranceEnd(msg) {
    const utterance = this.utterances.get(msg.utterance_id) || { sources: new Set(), started: false };
    utterance.ended = true;
    this.utterances.set(msg.utterance_id, utterance);
    if (!msg.has_audio) {
      setTimeout(() => this.send({ type: 'playback_finished', utterance_id: msg.utterance_id }), msg.text_only_s * 1000);
      return;
    }
    this.maybeFinished(msg.utterance_id);
  }

  maybeFinished(utteranceId) {
    const utterance = this.utterances.get(utteranceId);
    if (utterance && utterance.ended && utterance.sources.size === 0) {
      this.utterances.delete(utteranceId);
      this.send({ type: 'playback_finished', utterance_id: utteranceId });
    }
  }

  stopPlayback() {
    for (const utterance of this.utterances.values()) {
      for (const source of utterance.sources) {
        source.onended = null;
        try { source.stop(); } catch (e) { /* already stopped */ }
      }
    }
    this.utterances.clear();
    this.chunkStarts.clear();
    this.pendingVisemes.clear();
    this.avatar.clearVisemes();
    this.nextTime = 0;
  }

  // ------------------------------------------------------------------ devices

  async enableDevices() {
    if (this.devicesReady) return true;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
        video: !!this.options.camera,
      });
    } catch (err) {
      const reason = err && err.name === 'NotAllowedError'
        ? 'Microphone permission was denied. Allow it in the browser address bar, or type your answers below.'
        : 'No microphone is available. Connect one, or type your answers below.';
      this.notice(reason);
      this.send({ type: 'client_error', kind: (err && err.name) || 'getUserMedia', message: reason });
      return false;
    }
    if (this.options.camera && this.el.video) this.el.video.srcObject = this.stream;
    await this.ctx.audioWorklet.addModule('/static/js/pcm-capture-worklet.js');
    const source = this.ctx.createMediaStreamSource(new MediaStream(this.stream.getAudioTracks()));
    this.worklet = new AudioWorkletNode(this.ctx, 'pcm-capture', {
      processorOptions: { targetRate: TARGET_RATE, frameSamples: FRAME_SAMPLES },
    });
    const sink = this.ctx.createGain();
    sink.gain.value = 0;
    source.connect(this.worklet);
    this.worklet.connect(sink);
    sink.connect(this.ctx.destination);
    this.worklet.port.onmessage = ({ data }) => {
      this.el.meter.style.width = `${Math.min(100, data.rms * 400)}%`;
      if (!this.micPaused && this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(data.pcm);
    };
    this.stream.getAudioTracks().forEach((track) => {
      track.onended = () => {
        this.notice('The microphone was disconnected. Reconnect it and reload, or type your answers.');
        this.send({ type: 'client_error', kind: 'mic_ended', message: 'microphone track ended' });
      };
    });
    this.devicesReady = true;
    return true;
  }

  stopDevices() {
    this.micPaused = true;
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
  }

  // ------------------------------------------------------------------ controls

  async start() {
    await this.ctx.resume();
    await this.enableDevices();
    this.micPaused = false;
    this.send({ type: 'start' });
  }

  pause() {
    this.micPaused = true;
    this.send({ type: 'pause' });
  }

  async resume() {
    await this.ctx.resume();
    this.micPaused = false;
    this.send({ type: 'resume' });
  }

  end() {
    this.send({ type: 'end' });
  }

  doneAnswering() {
    this.send({ type: 'done_answering' });
  }

  sendText(text) {
    if (text && text.trim()) this.send({ type: 'text_answer', text: text.trim() });
  }
}

window.tsMedia = new InterviewMedia();
