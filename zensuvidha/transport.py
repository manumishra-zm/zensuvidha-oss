"""The seam between a CALL and the thing carrying it.

Why this exists
---------------
Every piece of this system that touches audio assumes a browser. Echo cancellation,
voice activity detection, endpointing, the latch guard and the band-split filter all
live in `web/index.html` and all depend on `getUserMedia`. A phone call has none of
them: a carrier hands you 8 kHz μ-law with your own voice mixed in and nothing
cancelled, nothing gated, nothing endpointed.

That is the whole distance between "runs in a browser" and "has a phone number", and it
is not a model problem — the recognition, isolation, identity, grounding and synthesis
stages are all transport-agnostic already. What was missing was somewhere for a second
transport to plug in without touching them.

    browser    getUserMedia ──▶ AudioWorklet ──▶ Silero ──▶ WebSocket ──┐
                                                                        ├──▶ Session
    telephony  carrier ──▶ μ-law 8k ──▶ resample ──▶ Silero ──▶ ────────┘   (unchanged)
               (no AEC, no VAD, no endpointer — this module supplies them)

What a transport owes the engine
--------------------------------
Exactly four things, and nothing else:

    recv_audio()   16 kHz mono float32 PCM, one utterance at a time
    send_audio()   the same, back
    send_text()    what was said, for a transcript or a UI
    hangup()       end the call

Everything above that — isolation, the speaker gate, the guard, the router — is already
independent of where the audio came from, and stays untouched. This is a transport swap,
not a rewrite, and the tests pin that.

On Pipecat specifically
-----------------------
Evaluated twice. It implements no diarization and no voice isolation, so it cannot help
the person-filtering problem this codebase spent most of its effort on — but it has
Exotel and Plivo built in, which are the India-relevant carriers, and that is the one
thing here that cannot be written in an afternoon. So the adapter below deliberately
uses Pipecat for **transport only** and keeps every audio decision on this side.

It is an OPTIONAL dependency. `pipecat-ai` is not in requirements.txt, this module
imports cleanly without it, and `PipecatTransport` raises a message telling you what to
install rather than failing at import time and taking the browser path down with it.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

log = logging.getLogger("zensuvidha.transport")

SR = 16000              # everything above this seam is 16 kHz mono float32
TELEPHONY_SR = 8000     # what a carrier actually hands you


@runtime_checkable
class Transport(Protocol):
    """What a call looks like to the engine, whatever is carrying it."""

    async def recv_audio(self):
        """Yield one utterance at a time, as 16 kHz mono float32.

        A transport that has its own endpointing (a browser) yields what it decided; one
        that has none (a carrier) must supply it — see `Endpointer` below.
        """
        ...

    async def send_audio(self, pcm) -> None:
        """Play 16 kHz mono float32 to the caller."""
        ...

    async def send_text(self, text: str, role: str = "assistant") -> None:
        """Surface what was said. A UI shows it; a phone line may only log it."""
        ...

    async def hangup(self) -> None:
        ...


def resample(pcm, src_sr: int, dst_sr: int):
    """Linear resample. Adequate here and deliberately not more.

    Telephony is 8 kHz and everything above this seam is 16 kHz. A polyphase filter
    would be more correct, but the band that matters for speech is already inside 8 kHz
    and the measured cost of getting this wrong is far smaller than the cost of another
    dependency. If this ever shows up in a WER measurement, replace it — and measure.
    """
    import numpy as np
    x = np.asarray(pcm, dtype="float32").reshape(-1)
    if src_sr == dst_sr or x.size == 0:
        return x
    n = int(round(x.size * dst_sr / src_sr))
    if n <= 0:
        return np.zeros(0, dtype="float32")
    idx = np.linspace(0, x.size - 1, n, dtype="float64")
    return np.interp(idx, np.arange(x.size), x).astype("float32")


# μ-law, done with a table rather than `audioop`. The stdlib module was removed in
# Python 3.13 — importing it would have made the telephony path, the one thing here
# aimed squarely at the future, the first thing to break on a modern interpreter.
# G.711 is a 256-entry codec; a table is exact, portable and faster than the C call.
_ULAW_BIAS = 0x84
_ULAW_CLIP = 32635


def _build_ulaw_table():
    import numpy as np
    out = np.zeros(256, dtype="<i2")
    for byte in range(256):
        u = ~byte & 0xFF
        mantissa = u & 0x0F
        exponent = (u >> 4) & 0x07
        sign = u & 0x80
        value = ((mantissa << 3) + _ULAW_BIAS) << exponent
        value -= _ULAW_BIAS
        out[byte] = -value if sign else value
    return out


_ULAW_DECODE = None


def ulaw_to_pcm(data: bytes):
    """μ-law bytes → float32. What Exotel, Plivo and Twilio actually send."""
    global _ULAW_DECODE
    import numpy as np
    if _ULAW_DECODE is None:
        _ULAW_DECODE = _build_ulaw_table()
    idx = np.frombuffer(data, dtype=np.uint8)
    return (_ULAW_DECODE[idx].astype("float32") / 32768.0)


def pcm_to_ulaw(pcm) -> bytes:
    import numpy as np
    x = np.clip(np.asarray(pcm, dtype="float32").reshape(-1), -1.0, 1.0)
    s = (x * 32767.0).astype("<i4")
    sign = np.where(s < 0, 0x80, 0).astype(np.uint8)
    mag = np.minimum(np.abs(s), _ULAW_CLIP) + _ULAW_BIAS
    # The segment is the HIGHEST set bit at or above bit 7. Iterating downward let the
    # smallest matching exponent win instead of the largest, which put every sample in
    # the wrong segment — a round trip lost 0.40 of full scale rather than 0.02.
    exponent = np.zeros(mag.shape, dtype=np.int32)
    for e in range(0, 8):
        exponent = np.where((mag >> (e + 7)) & 1, e, exponent)
    mantissa = (mag >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4).astype(np.uint8)
              | mantissa.astype(np.uint8)) & 0xFF).astype(np.uint8).tobytes()


class Endpointer:
    """Turn a continuous stream into utterances, for transports that cannot.

    The browser does this in an AudioWorklet with Silero and a learned window. A carrier
    gives you an unbroken stream, so somebody has to decide where a turn ends — and if
    that somebody is not here, it is the LLM, which will happily answer half a sentence.

    It now uses the SAME ladder as the browser, from zensuvidha/turn.py, rather than a
    copy of its shape. That distinction mattered: this class had drifted to a flat
    800/1200ms with no learned pause, no per-slot extra and no prosody, so a caller on a
    phone line would have been endpointed by exactly the rules two rounds of live
    testing tuned away from ("it breaks my long sentences"). Same policy object, same
    eagerness setting, one place to change it.

    What a carrier still cannot give us is the LEXICAL and PROSODIC half — those need a
    transcript and a pitch read, which happen after this. So `learned_pause_ms` and
    `hold`/`tone` are accepted here and left for the caller to supply once it has them.
    """

    MAX_UTT_MS = 30000      # steady noise must not latch this open forever

    def __init__(self, sr: int = SR, is_speech=None, policy=None, eagerness=None):
        from . import turn as _turn
        self.policy = policy or _turn.policy(eagerness)
        self._turn = _turn
        self.learned_pause_ms = 0.0
        self.sr = sr
        self._is_speech = is_speech or self._energy_gate
        self._buf = []
        self._utt_ms = 0.0
        self._silence_ms = 0.0
        self._floor = 0.004

    @staticmethod
    def _energy_gate(frame, floor: float) -> bool:
        """The fallback when no VAD is supplied. Deliberately crude — a real deployment
        passes Silero in, and this only has to stop a silent line looking like speech."""
        import numpy as np
        x = np.asarray(frame, dtype="float32")
        return bool(x.size) and float((x * x).mean()) ** 0.5 > floor * 2.5

    def feed(self, frame):
        """Add a frame. Returns a finished utterance, or None."""
        import numpy as np
        x = np.asarray(frame, dtype="float32").reshape(-1)
        if not x.size:
            return None
        ms = 1000.0 * x.size / self.sr
        speech = self._is_speech(x, self._floor)

        if not self._buf and not speech:
            # adapt the floor on true silence only, so a long turn cannot drag it up
            rms = float((x * x).mean()) ** 0.5
            self._floor = 0.98 * self._floor + 0.02 * rms
            return None

        self._buf.append(x)
        self._utt_ms += ms
        self._silence_ms = 0.0 if speech else self._silence_ms + ms

        window = self._turn.window_ms(self.policy, utt_ms=self._utt_ms,
                                      learned_pause_ms=self.learned_pause_ms)
        if self._silence_ms >= window or self._utt_ms >= self.MAX_UTT_MS:
            return self.flush()
        return None

    def flush(self):
        import numpy as np
        if not self._buf:
            return None
        out = np.concatenate(self._buf)
        self._buf, self._utt_ms, self._silence_ms = [], 0.0, 0.0
        return out


class PipecatTransport:
    """Transport-only adapter for Pipecat (Exotel / Plivo / Twilio / Telnyx).

    Pipecat is used for the carrier connection and NOTHING else. Its own STT, LLM, TTS
    and turn-taking services are deliberately not wired in: this codebase's isolation,
    speaker gate and grounding guard are the parts worth keeping, and none of them exist
    over there.

    Optional dependency — `pip install pipecat-ai`. Constructing this without it raises
    something you can act on, rather than failing at import and taking the browser path
    down with it.
    """

    def __init__(self, carrier_stream, sr: int = TELEPHONY_SR, encoding: str = "ulaw"):
        try:
            import pipecat  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError(
                "PipecatTransport needs the optional dependency: pip install pipecat-ai\n"
                "It is not in requirements.txt because the browser path does not use it."
            ) from e
        self.stream = carrier_stream
        self.sr = sr
        self.encoding = encoding
        self.endpointer = Endpointer(sr=SR)
        self._open = True

    def _decode(self, chunk: bytes):
        pcm = ulaw_to_pcm(chunk) if self.encoding == "ulaw" else _pcm16(chunk)
        return resample(pcm, self.sr, SR)

    async def recv_audio(self):
        async for chunk in self.stream:
            if not self._open:
                return
            utt = self.endpointer.feed(self._decode(chunk))
            if utt is not None:
                yield utt
        tail = self.endpointer.flush()
        if tail is not None:
            yield tail

    async def send_audio(self, pcm) -> None:
        out = resample(pcm, SR, self.sr)
        payload = pcm_to_ulaw(out) if self.encoding == "ulaw" else _to_pcm16(out)
        await self.stream.send(payload)

    async def send_text(self, text: str, role: str = "assistant") -> None:
        log.info("[%s] %s", role, text)          # a phone line has nowhere to show it

    async def hangup(self) -> None:
        self._open = False
        close = getattr(self.stream, "close", None)
        if close:
            await close()


class LiveKitTransport:
    """Transport-only adapter for LiveKit (WebRTC in, SIP trunk to a phone number).

    Evaluated the same way Pipecat was, and the answer has the same shape: take the
    transport, leave the audio decisions here.

    WHAT LIVEKIT WOULD ACTUALLY GIVE US
      * A WebRTC transport that is not a raw WebSocket of WAV blobs: jitter buffering,
        packet-loss concealment and Opus, none of which this project has. On a real
        network that is the difference between a dropped word and a recovered one.
      * SIP, so the same agent answers an actual phone number.
      * A server that is Apache-2.0 and SELF-HOSTABLE, which is the only reason this is
        considered at all — the premise here is that everything runs on your own box.

    WHAT IT WOULD NOT
      * Its enhanced noise cancellation and background voice cancellation are Krisp
        models available only through LiveKit Cloud, and the self-hosting instructions
        say to remove that plugin. So the ONE feature that overlaps what this codebase
        spent most of its effort on is exactly the feature you cannot self-host. And on
        the recognition path we have already measured that denoising HURTS (DeepFilter
        won 1 of 30 SNR cells, net -400% word recall), so it is not a loss worth
        taking a cloud dependency for.
      * Its turn detector is a genuinely interesting model — but it is bound to the
        `livekit-agents` session object, ships under the LiveKit Model License rather
        than an OSI one, and covers 14 languages: Hindi yes, Telugu no. Our endpointer
        learns per caller and `guard.looks_incomplete()` reads the words; adopting
        theirs would mean restructuring into their agent loop to get a model that
        cannot serve the languages the GPU work is for.

    So: transport only, and everything above this seam — isolation, the speaker gate,
    the router, the grounding guard — stays exactly as it is.

    `audio_stream` is any async iterable of LiveKit audio frames (what `rtc.AudioStream`
    yields); `audio_source` is an `rtc.AudioSource` to play into; `room` is optional and
    used only for text and hangup. They are passed in rather than constructed here so
    the seam can be tested without the SDK, and so connecting a room stays the caller's
    business.
    """

    def __init__(self, audio_stream, audio_source=None, room=None,
                 sr: int = 48000, require_sdk: bool = True):
        if require_sdk:
            try:
                import livekit.rtc  # noqa: F401
            except ImportError as e:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "LiveKitTransport needs the optional dependency: "
                    "pip install livekit\n"
                    "It is not in requirements.txt because the browser path does not "
                    "use it. Note that the noise-cancellation plugin is LiveKit Cloud "
                    "only and is deliberately not used here."
                ) from e
        self.stream = audio_stream
        self.source = audio_source
        self.room = room
        self.sr = sr                     # WebRTC is 48k; everything above here is 16k
        self.endpointer = Endpointer(sr=SR)
        self._open = True

    @staticmethod
    def _frame_pcm(event):
        """LiveKit hands you a frame event; the frame holds int16 samples.

        Written defensively about the wrapper because the SDK has moved this around
        (`event.frame` vs the frame itself), and getting it wrong yields silence rather
        than an error — the failure mode this project keeps meeting.
        """
        import numpy as np
        frame = getattr(event, "frame", event)
        data = getattr(frame, "data", None)
        if data is None:
            return np.zeros(0, dtype="float32"), None
        arr = np.frombuffer(bytes(data), dtype="<i2").astype("float32") / 32768.0
        ch = int(getattr(frame, "num_channels", 1) or 1)
        if ch > 1:
            arr = arr.reshape(-1, ch).mean(axis=1)
        return arr, getattr(frame, "sample_rate", None)

    async def recv_audio(self):
        async for event in self.stream:
            if not self._open:
                return
            pcm, sr = self._frame_pcm(event)
            if not pcm.size:
                continue
            utt = self.endpointer.feed(resample(pcm, sr or self.sr, SR))
            if utt is not None:
                yield utt
        tail = self.endpointer.flush()
        if tail is not None:
            yield tail

    async def send_audio(self, pcm) -> None:
        if self.source is None:
            return
        import numpy as np
        out = resample(pcm, SR, self.sr)
        pcm16 = np.clip(out, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype("<i2")
        try:
            from livekit import rtc
            frame = rtc.AudioFrame(data=pcm16.tobytes(), sample_rate=self.sr,
                                   num_channels=1, samples_per_channel=len(pcm16))
        except ImportError:      # pragma: no cover - the stubbed path used by tests
            frame = pcm16.tobytes()
        await self.source.capture_frame(frame)

    async def send_text(self, text: str, role: str = "assistant") -> None:
        # LiveKit has a data channel, so unlike a phone line there IS somewhere to put
        # this. Failing to deliver a transcript must never take the call down with it.
        lp = getattr(self.room, "local_participant", None) if self.room else None
        if lp is None:
            log.info("[%s] %s", role, text)
            return
        try:
            await lp.publish_data(text.encode("utf-8"), topic=role)
        except Exception as e:  # noqa: BLE001
            log.warning("livekit: could not publish the transcript (%s)", e)

    async def hangup(self) -> None:
        self._open = False
        for obj in (self.stream, self.room):
            close = getattr(obj, "aclose", None) or getattr(obj, "disconnect", None)
            if close:
                try:
                    await close()
                except Exception as e:  # noqa: BLE001
                    log.warning("livekit: hangup on %s failed (%s)",
                                type(obj).__name__, e)


def _pcm16(data: bytes):
    import numpy as np
    return np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0


def _to_pcm16(pcm) -> bytes:
    import numpy as np
    x = np.clip(np.asarray(pcm, dtype="float32").reshape(-1), -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()
