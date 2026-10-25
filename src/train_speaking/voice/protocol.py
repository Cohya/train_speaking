"""Wire format for audio on `/ws/audio`.

Client to server: binary messages of little-endian int16 mono PCM at 16 kHz. Any length is
accepted; the server re-chunks into VAD frames.

Server to client: binary messages made of a 12-byte header followed by int16 mono PCM:
`<uint32 utterance_id><uint32 seq><uint32 sample_rate>`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

INPUT_SAMPLE_RATE = 16_000
# Silero VAD requires exactly 512-sample frames at 16 kHz (32 ms).
VAD_FRAME_SAMPLES = 512
FRAME_MS = VAD_FRAME_SAMPLES * 1000 / INPUT_SAMPLE_RATE

_HEADER = struct.Struct("<III")
HEADER_BYTES = _HEADER.size
MAX_CLIENT_AUDIO_MESSAGE_BYTES = 64_000


@dataclass(frozen=True)
class AudioChunkHeader:
    utterance_id: int
    seq: int
    sample_rate: int


def encode_audio_chunk(header: AudioChunkHeader, pcm16: bytes) -> bytes:
    return _HEADER.pack(header.utterance_id, header.seq, header.sample_rate) + pcm16


def decode_audio_chunk(message: bytes) -> tuple[AudioChunkHeader, bytes]:
    if len(message) < HEADER_BYTES:
        raise ValueError("audio chunk shorter than its header")
    utterance_id, seq, sample_rate = _HEADER.unpack_from(message)
    return AudioChunkHeader(utterance_id, seq, sample_rate), message[HEADER_BYTES:]
