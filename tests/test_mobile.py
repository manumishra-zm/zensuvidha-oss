"""The browser client on a PHONE, which is not a small laptop.

A laptop call runs in a foreground tab, on mains power, with a built-in microphone
nobody else wants and an AudioContext at 48kHz that never changes. None of those hold
on a phone, and every one of them is load-bearing:

  * the SAMPLE RATE is the device's, not a constant. Chrome on Android builds the
    AudioContext at whatever the current route runs at, and a Bluetooth headset on the
    HFP/SCO profile is 8kHz. `floatTo16kWav` only ever resampled DOWNWARD while writing
    16000 into the header unconditionally, so an 8k recording was shipped labelled 16k:
    Whisper read the caller at double speed an octave high and returned nothing. The VAD
    was unaffected — the worklet's resampler handles both directions — so turns opened
    and closed normally and simply transcribed to "", which lands on `commit_miss` and
    asks the caller to repeat into a microphone that will do it again.

  * the MICROPHONE can be taken away. The screen times out, a notification is glanced
    at, a real call arrives, a headset connects. Each ends with `callActive` still true,
    an orb still reading LISTENING, and a mic that will never deliver another frame.

  * NOISE SUPPRESSION was requested from the OS, which contradicts the measurement the
    whole audio pipeline is built on (zensuvidha/denoise.py: Whisper WER 0.00 -> 0.10,
    speaker similarity 0.675 -> 0.596). On a phone the suppressor is far more aggressive
    than a laptop's, and it sits upstream of everything, where it cannot be measured,
    A/B'd or switched off.

The resampler tests run the REAL function out of `web/index.html` in Node, like
tests/test_inspector.py — copying it here would pin the copy, not the page. The
lifecycle tests read the source, because the thing being asserted is that a handler is
wired at all, and a headless Node run has no visibilitychange to fire.

Run:  pytest -q tests/test_mobile.py
"""
import json
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "web", "index.html")
NODE = shutil.which("node")

SRC = open(INDEX, encoding="utf-8").read()


def _extract(src: str, name: str) -> str:
    """Return `function name(...){...}`, matched by counting braces."""
    start = src.index("function %s(" % name)
    depth, i, opened = 0, start, False
    while i < len(src):
        c = src[i]
        if c == "{":
            depth, opened = depth + 1, True
        elif c == "}":
            depth -= 1
            if opened and depth == 0:
                return src[start:i + 1]
        i += 1
    raise AssertionError("unbalanced braces in " + name)


def _run(js: str):
    prog = STUBS + _extract(SRC, "floatTo16kWav") + "\n" + js
    out = subprocess.run([NODE, "-e", prog], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout or "null")


# `floatTo16kWav` returns a Blob, which Node does not have and this test does not need:
# every assertion below is about the WAV bytes, so capture them instead.
STUBS = r"""
global.Blob = class { constructor(parts){ this.parts = parts; } };
// One second of a pure tone at the DEVICE's rate — the input a phone actually hands us.
function tone(hz, srcRate, seconds){
  const n = Math.round(srcRate * seconds), a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = Math.sin(2 * Math.PI * hz * i / srcRate);
  return a;
}
// Read the fields the SERVER reads: soundfile trusts the header, so a header that
// disagrees with the payload is the whole bug.
function wavHeader(res){
  const v = new DataView(res.blob.parts[0].buffer ? res.blob.parts[0].buffer
                                                  : res.blob.parts[0]);
  return { rate: v.getUint32(24, true),
           byteRate: v.getUint32(28, true),
           dataBytes: v.getUint32(40, true) };
}
// The pitch the server will hear, measured off the shipped samples the same way a
// listener would notice it: zero crossings per second at the DECLARED rate.
function pitchAt(res, declaredRate){
  const d = res.data;
  let zc = 0;
  for (let i = 1; i < d.length; i++) if ((d[i-1] < 0) !== (d[i] < 0)) zc++;
  return zc / 2 / (d.length / declaredRate);
}
"""

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


# ---------------------------------------------------------------- the sample rate

# Every rate a browser realistically hands us. 8000 is a Bluetooth headset on SCO,
# 16000 is the same headset on mSBC, 44100 is common on Android, 48000 is a laptop.
@pytest.mark.parametrize("src_rate", [8000, 16000, 22050, 44100, 48000])
def test_the_header_never_lies_about_the_rate(src_rate):
    """The declared rate and the payload must describe the same recording.

    This is the failure exactly: the header said 16000 whatever came in, so an 8k clip
    became a half-length 16k one. soundfile resamples correctly on the server — it was
    never told the truth to resample FROM.
    """
    out = _run("""
    const res = floatTo16kWav([tone(440, %d, 1.0)], %d);
    const h = wavHeader(res);
    console.log(JSON.stringify({rate: h.rate, seconds: res.seconds,
                                samples: res.data.length,
                                dataBytes: h.dataBytes, byteRate: h.byteRate}));
    """ % (src_rate, src_rate))
    assert out["rate"] == 16000, "the pipeline is 16k everywhere — see zensuvidha/pipeline.py"
    # One second in, one second out, whatever the device rate was.
    assert out["samples"] == pytest.approx(16000, abs=2)
    assert out["seconds"] == pytest.approx(1.0, abs=0.01)
    # …and the header agrees with the bytes, which is what soundfile actually reads.
    assert out["dataBytes"] == out["samples"] * 2
    assert out["byteRate"] == out["rate"] * 2


@pytest.mark.parametrize("src_rate", [8000, 16000, 44100, 48000])
def test_the_caller_keeps_their_own_pitch(src_rate):
    """A 440Hz tone must still be 440Hz after the trip.

    The bug had no effect on length alone — it doubled the PITCH, which is what made
    Whisper return nothing rather than return something wrong. Before the fix this
    reads ~880Hz at 8000 and passes at every other rate, which is precisely why it was
    invisible on a laptop.
    """
    out = _run("""
    const res = floatTo16kWav([tone(440, %d, 1.0)], %d);
    console.log(JSON.stringify({hz: pitchAt(res, 16000)}));
    """ % (src_rate, src_rate))
    assert out["hz"] == pytest.approx(440, rel=0.02)


def test_a_below_16k_mic_is_upsampled_rather_than_passed_through():
    """Guards the shape of the fix, not just its output.

    A future edit that drops the upsample branch but keeps the header write would put
    the original bug back with these length assertions still passing at 16k/44.1k/48k —
    so pin the one case that distinguishes them.
    """
    out = _run("""
    const res = floatTo16kWav([tone(440, 8000, 0.5)], 8000);
    console.log(JSON.stringify({samples: res.data.length, seconds: res.seconds}));
    """)
    assert out["samples"] == pytest.approx(8000, abs=2), \
        "0.5s at 8000Hz must become 8000 samples at 16k, not 4000 mislabelled ones"
    assert out["seconds"] == pytest.approx(0.5, abs=0.01)


def test_an_empty_or_silent_capture_does_not_crash():
    """The recovery paths below can hand this an empty buffer mid-rebuild."""
    out = _run("""
    const res = floatTo16kWav([new Float32Array(0)], 8000);
    console.log(JSON.stringify({samples: res.data.length}));
    """)
    assert out["samples"] <= 1


# ------------------------------------------------------------- losing the mic

def test_noise_suppression_is_not_requested():
    """The OS suppressor is the harm zensuvidha/denoise.py measured, applied upstream
    of every switch that exists to control it."""
    assert "noiseSuppression:false" in SRC.replace(" ", ""), \
        "requesting OS noise suppression contradicts denoise.py's own measurements"
    # AEC is NOT in the same category — barge-in depends on it, and it removes OUR
    # voice, never the caller's. A blanket "turn the processing off" would break it.
    assert "echoCancellation:true" in SRC.replace(" ", "")


def test_what_the_os_actually_gave_us_is_reported():
    """A phone may ignore any of these. A constraint that was silently overridden is
    worse than one never asked for, because the pipeline downstream assumes it held."""
    assert "getSettings" in SRC
    assert "logMicSettings" in SRC


def test_a_refused_constraint_set_still_gets_a_call():
    """Some phones reject the whole set rather than relaxing it. A call on the OS's
    own terms beats no call."""
    assert "getUserMedia({audio:true})" in SRC.replace(" ", ""), \
        "there must be a bare-constraint retry before giving up on the microphone"


@pytest.mark.parametrize("hook", ["onended", "onmute", "onunmute"])
def test_the_mic_track_is_watched(hook):
    """The OS ends the track when a real call takes the mic, and mutes it on a route
    change. Neither raises anything the page sees unless it is listening."""
    assert "t.%s" % hook in SRC


def test_backgrounding_is_handled():
    """iOS suspends the AudioContext when the page hides and never resumes it, and a
    wake lock is always released on hide. Both have to be re-taken on return."""
    assert "visibilitychange" in SRC
    assert "audioCtx.resume()" in SRC
    assert "wakeLock" in SRC


def test_a_route_change_is_handled():
    """A headset connecting mid-call changes the device behind the track."""
    assert "devicechange" in SRC


def test_recovery_rebuilds_without_ending_the_call():
    """The WebSocket, the session id and the conversation must all survive losing the
    microphone — only the audio graph is replaced. Ending the call would drop the
    caller's booking half-finished."""
    fn = _extract(SRC, "restartMic")
    assert "getUserMedia" in fn
    assert "callActive=false" not in fn.replace(" ", ""), \
        "restartMic must not end the call; it rebuilds the mic under a live session"
    # …but if it genuinely cannot get a mic back, it MUST say so rather than leave an
    # orb reading LISTENING beside a dead microphone.
    assert "endCall()" in fn


def test_recovery_cannot_run_twice_at_once():
    """`ended` and `visibilitychange` fire together when a call is answered mid-session;
    two concurrent rebuilds race for one AudioContext."""
    assert "micRecovering" in SRC


def test_hanging_up_disarms_the_watchers_before_stopping_the_tracks():
    """`stop()` fires `ended`. A handler still pointed at a stream being torn down is
    how a deliberate hang-up races a recovery."""
    fn = _extract(SRC, "endCall")
    stop = fn.index("getTracks().forEach")
    disarm = fn.index("onended=")
    assert disarm < stop, "clear the track handlers BEFORE stopping the tracks"
    assert "releaseWake()" in fn
