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
    """Return `function name(...){...}`, matched by counting braces.

    Keeps a leading `async`. Without it the extracted body still parses — right up until
    it contains an `await`, at which point Node rejects it for a reason that has nothing
    to do with the code under test.
    """
    start = src.index("function %s(" % name)
    if src[max(0, start - 6):start] == "async ":
        start -= 6
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
    """The recovery paths below can hand this an empty buffer mid-rebuild.

    And it must not produce NaN. The interpolator's tail reads flat[length-1], which on
    an empty buffer is `undefined` — a Float32Array stores that as NaN, and the same
    array is handed to the inspector to draw. A waveform of NaN renders as nothing,
    which looks exactly like a turn that was correctly silent.
    """
    out = _run("""
    const res = floatTo16kWav([new Float32Array(0)], 8000);
    console.log(JSON.stringify({samples: res.data.length,
                                finite: Array.from(res.data).every(Number.isFinite)}));
    """)
    assert out["samples"] <= 1
    assert out["finite"], "an empty capture must not resample into NaN"


# ------------------------------------------------------------- losing the mic

def test_noise_suppression_is_not_requested_anywhere():
    """The OS suppressor is the harm zensuvidha/denoise.py measured, applied upstream
    of every switch that exists to control it.

    Asserted as an ABSENCE across the whole file, not as the presence of one correct
    line. The first version of this test checked that `noiseSuppression:false` appeared
    somewhere and passed happily while the brand-voice recorder a thousand lines further
    down still asked for `true` — on the one clip whose entire job is carrying speaker
    identity, where suppression costs the most.
    """
    flat = SRC.replace(" ", "")
    assert "noiseSuppression:true" not in flat, \
        "some getUserMedia call still asks the OS to denoise"
    assert "noiseSuppression:false" in flat
    # AEC is NOT in the same category — barge-in depends on it, and it removes OUR
    # voice, never the caller's. A blanket "turn the processing off" would break it.
    assert "echoCancellation:true" in flat


def test_every_capture_path_uses_the_same_constraints():
    """A second constraint object is how the two drift. The call path and the clone
    recorder must ask for the same processing, or the voiceprint the agent enrols
    against and the reference it is cloned from were recorded through different front
    ends."""
    import re
    # Every getUserMedia with an inline audio object, other than the bare-`true` retries.
    inline = re.findall(r"getUserMedia\(\{audio:\s*\{", SRC.replace(" ", ""))
    assert not inline, "capture paths must share MIC_CONSTRAINTS, not inline their own"


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


def test_the_wake_lock_is_retaken_after_the_platform_releases_it():
    """The one that makes the wake lock worth having.

    The platform releases the lock whenever the page is hidden and leaves the sentinel
    in place with `released === true`. Guarding on `!wakeLock` alone therefore holds a
    DEAD lock forever: the first glance at a notification releases it, the re-acquire on
    return sees a non-null sentinel and does nothing, and the screen times out on every
    cycle after that — the exact cascade the lock exists to prevent.

    Run for real rather than grepped, because the bug is one truthy check and reads as
    correct.
    """
    prog = STUBS_WAKE + _extract(SRC, "keepAwake") + _extract(SRC, "releaseWake") + r"""
    (async () => {
      await keepAwake();
      const first = wakeLock;
      hide();                       // the platform drops it when the page is hidden
      await keepAwake();            // ...and we come back to the foreground
      console.log(JSON.stringify({
        gotOneAtAll: first !== null,
        heldAfterReturn: wakeLock !== null && wakeLock.released === false,
        requests: navigator.wakeLock.requests,
        differentSentinel: wakeLock !== first}));
    })();
    """
    out = subprocess.run([NODE, "-e", prog], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert got["gotOneAtAll"]
    assert got["heldAfterReturn"], "the lock was not re-taken after the platform released it"
    assert got["requests"] == 2, "a released lock must be re-requested, not reused"
    assert got["differentSentinel"]


STUBS_WAKE = r"""
let wakeLock = null;
// A sentinel that behaves like the real one: hiding the page releases it in place.
function makeSentinel(){
  const listeners = [];
  return { released:false,
           addEventListener:(k,f)=>{ if(k==='release') listeners.push(f); },
           release(){ this.released = true; listeners.forEach(f=>f()); },
           _drop(){ this.released = true; } };
}
// Node 21+ ships its own read-only `navigator` global, so a plain assignment is
// silently dropped and every assertion below would pass against an untested stub.
Object.defineProperty(globalThis, 'navigator', {configurable:true, writable:true,
  value: { wakeLock: { requests: 0,
    async request(){ navigator.wakeLock.requests++; return makeSentinel(); } } }});
// What the platform does on visibilitychange: releases the lock WITHOUT telling the
// page, leaving the object non-null. No 'release' listener fires in this shape, which
// is what makes the `!wakeLock` guard look sufficient.
function hide(){ if (wakeLock) wakeLock._drop(); }
"""


def test_recovery_clears_the_latch_state_too():
    """`maxGapMs` feeds the LATCH check — 7s of recording with no pause means "this is
    not a person, it is continuous noise". Carrying it across a rebuild would discard
    the caller's first sentence on the NEW microphone."""
    fn = _extract(SRC, "restartMic")
    for name in ("maxGapMs", "uttMs", "specSent", "commitTimer"):
        assert name in fn, "restartMic must clear %s" % name


def test_recovery_drops_the_speculation_before_it_clears_the_flag():
    """Ordering, and it reads as correct either way.

    `dropSpeculation()` returns early unless `specSent` is true, so resetting the
    counters first turns it into a silent no-op — and that message is the only thing
    that reaches `_void_speculation` on the server (zensuvidha/server.py, the `stt_hint`
    handler). Without it a speculative reply keeps generating against words from a
    microphone that no longer exists, and competes with the caller's next real turn.
    """
    fn = _extract(SRC, "restartMic")
    assert fn.index("dropSpeculation()") < fn.index("specSent=false"), \
        "dropSpeculation must run while specSent is still true"


def test_recovery_cancels_the_pending_unmute_check():
    """`stop()` does not clear a track's `muted`. A timer armed before the rebuild fires
    ~1.5s later, sees the old dead track still muted, and tears down the healthy
    microphone that was just rebuilt — discarding the utterance in progress."""
    fn = _extract(SRC, "restartMic")
    assert "_unmuteTimer" in fn


def test_recovery_gives_up_if_the_caller_hung_up_while_it_waited():
    """Getting a mic back on a phone takes hundreds of ms to seconds. If the caller taps
    End call in that window, endCall runs against a micStream restartMic already nulled
    — so it stops nothing — and everything after the await would build a LIVE graph on
    an ended call: recording indicator lit, orb reading LISTENING, and the next
    startCall overwrites micStream so the track is never stopped again."""
    fn = _extract(SRC, "restartMic").replace(" ", "")
    # The LAST getUserMedia in the function is the fallback retry; the guard must come
    # after every one of them, not merely somewhere in the body.
    after = fn[fn.rindex("getUserMedia"):]
    assert "if(!callActive)" in after, \
        "restartMic must re-check callActive after the await"
    guard = after[after.index("if(!callActive)"):]
    assert "stop()" in guard[:300], "and stop the track it just acquired"
    assert "return" in guard[:300]


def test_recovery_has_the_same_constraint_fallback_as_starting():
    """A route change is exactly when a phone starts refusing the constraint set.
    Failing the whole rebuild over it ends a call a bare request could have kept."""
    fn = _extract(SRC, "restartMic")
    assert "getUserMedia({audio:true})" in fn.replace(" ", "")


def test_frames_in_flight_cannot_reach_a_rebuilt_utterance():
    """`pump` is an async loop. Frames captured before the old node was disconnected can
    still be sitting in it, and would be appended to a buffer restartMic just cleared —
    half a sentence from a microphone that no longer exists, sent as a turn."""
    fn = _extract(SRC, "handleFrame")
    head = fn[:fn.index("micMode==='ptt'")]
    assert "micRecovering" in head, \
        "handleFrame must bail on a rebuild BEFORE it buffers anything"


# --------------------------------------------------------- which level opens a turn

def test_the_energy_gate_uses_the_speech_band():
    """`vad-worklet.js` computed `rmsBand` through its 300-3400Hz cascade, posted it
    every 32ms, and the gate read full-band `rms` — so the filter that was written and
    benchmarked was measured and discarded on every frame of every call.

    scripts/bench_vad.py, on the filter as it ships: separation improves on all five
    interference types, mean 3.97x, worst case traffic at 1.11x.
    """
    fn = _extract(SRC, "handleFrame")
    assert "rmsBand" in fn, "the speech-band level is still being thrown away"


def test_the_absolute_floor_moved_with_the_signal():
    """The half that makes the switch safe rather than merely better.

    The gate is max(floor*4, ABS). The adaptive half self-corrects because it tracks
    whatever it is fed; the ABSOLUTE half does not. Band-limiting takes speech itself to
    0.81 of its full-band level, so leaving 0.006 in place would have made the gate
    quietly deafer — a silent desensitisation, which is the failure mode this codebase
    keeps finding late.
    """
    fn = _extract(SRC, "handleFrame")
    assert "0.005" in fn, "the absolute floor must scale with the band-limited signal"
    assert "0.006" in fn, "...and the full-band path must keep its own measured floor"


def test_the_full_band_floor_survives_for_the_two_things_that_need_it():
    """`noiseFloor` is also read by the self-echo bar — compared against `agentLevel()`,
    which is full-band — and by the drop-tiny-utterance check, against a full-band peak.
    Feeding either a band-limited number would move two thresholds nothing measured."""
    assert "bandFloor" in SRC, "the speech gate needs its own floor"
    fn = _extract(SRC, "handleFrame")
    assert "noiseFloor=noiseFloor*0.97" in fn.replace(" ", ""), \
        "the full-band floor must still be learned"
    assert "bandFloor=bandFloor*0.97" in fn.replace(" ", ""), \
        "...and the band floor learned from the measure it is compared against"
    # The echo bar is the one that would break silently.
    assert "noiseFloor*6" in fn.replace(" ", "")


def test_the_gate_falls_back_when_the_band_level_is_absent():
    """The ScriptProcessor path posts no `rmsBand` (it has no filter). It must keep
    working on full-band rms with the floor that was measured for it."""
    fn = _extract(SRC, "handleFrame").replace(" ", "")
    assert "f.rmsBand!=null" in fn, "the band level has to be optional"


def test_the_vad_bench_measures_the_shipping_filter():
    """The stated lesson from the last time this cascade was measured: the ideal FFT
    said 4.47x and the real two-pole filter delivered 2.79x. A numpy port of the filter
    would be measuring something that does not ship."""
    bench = os.path.join(ROOT, "scripts", "bench_vad.py")
    src = open(bench, encoding="utf-8").read()
    assert "vad-worklet.js" in src
    assert "class Biquad" in src, "it must lift the real class, not reimplement it"


# ------------------------------------------------------------- the health tick

def test_there_is_a_periodic_health_check_at_all():
    """Before this the client had exactly two timers — the spectrogram and the mascot
    animation — and every recovery path was event-driven. That covers the failures the
    platform announces; the ones it does not announce are the ones that leave a call
    looking alive."""
    assert "healthTick" in SRC
    assert "setInterval(healthTick" in SRC.replace(" ", "")


def test_the_health_tick_notices_a_machine_that_slept():
    """A closed lid does not hide the tab, so `visibilitychange` never fires — the
    desktop equivalent of the mobile backgrounding case, and previously uncovered. A
    tick that arrives seconds late is the only evidence available."""
    fn = _extract(SRC, "healthTick")
    assert "SLEEP_JUMP_MS" in fn
    assert "restartMic" in fn


def test_the_health_tick_notices_a_stalled_worklet():
    """Frames stop arriving while `track.readyState` stays "live" — a known outcome of
    a route change, with no event for it."""
    fn = _extract(SRC, "healthTick")
    assert "lastFrameAt" in fn and "FRAME_STALL_MS" in fn
    assert "lastFrameAt=Date.now()" in SRC.replace(" ", ""), \
        "something must actually stamp the frame clock"


def test_the_health_tick_notices_a_half_open_socket():
    """`ws.readyState` stays 1 on a dead TCP connection — the normal result of a Wi-Fi
    to cellular handover. Every send guard in the client trusts that 1, so this is the
    only thing that can contradict it."""
    fn = _extract(SRC, "healthTick")
    assert "SOCKET_DEAD_MS" in fn
    assert "ws.close()" in fn.replace(" ", ""), \
        "closing it ourselves is what routes into the reconnect path that already works"
    assert "lastServerMsgAt=Date.now()" in SRC.replace(" ", "")


def test_the_server_answers_a_keepalive():
    """The half-open check above is only as good as the reply that proves it."""
    import inspect

    from zensuvidha import server
    src = inspect.getsource(server.ws)
    assert '"ping"' in src and '"pong"' in src


def test_the_receive_loop_exempts_keepalives_from_presence():
    """The trap this whole heartbeat walks into.

    The receive loop marks presence on EVERY inbound message. The browser pings every
    5s, which is inside both idle windows — so without an exemption the heartbeat
    silently disables `idle_prompt_seconds` AND `idle_hangup_seconds` for every browser
    caller. The feature would look like it worked; only the thing it broke is invisible.
    """
    import inspect

    from zensuvidha import server
    src = inspect.getsource(server.ws)
    assert "is_keepalive(msg)" in src, "a ping must not reset the idle timer"
    assert server.IDLE_PROMPT_S > 0 and server.IDLE_HANGUP_S > 0


@pytest.mark.parametrize("frame,expected", [
    ({"text": '{"type":"ping"}'}, True),
    # A caller who TYPES the word. Decided on the parsed type precisely so this keeps
    # resetting their own idle timer — otherwise the agent asks "are you still there?"
    # at somebody who is mid-conversation.
    ({"text": '{"type":"text","text":"ping"}'}, False),
    ({"text": '{"type":"text","text":"ping the doctor for me"}'}, False),
    ({"text": '{"type":"commit"}'}, False),
    ({"text": '{"type":"cancel"}'}, False),
    # Audio is the caller unambiguously present.
    ({"bytes": b"RIFF...."}, False),
    # Unparseable still came from a client that is doing something — presence is the
    # safe answer, and the body below handles the malformed frame on its own.
    ({"text": "not json at all"}, False),
    ({"text": "null"}, False),
    ({"text": "[]"}, False),
    ({"text": '{"no_type":1}'}, False),
    ({"type": "websocket.disconnect"}, False),
])
def test_only_a_real_keepalive_is_exempt(frame, expected):
    from zensuvidha.server import is_keepalive
    assert is_keepalive(frame) is expected


def test_the_keepalive_interval_is_inside_the_idle_window():
    """If the ping cadence ever drifts past the idle windows the exemption above stops
    mattering — and if it drifts past the client's own dead-socket timeout, every call
    reconnects on a loop. Pin the relationship, not the numbers."""
    import re

    from zensuvidha import server
    ping = int(re.search(r"PING_EVERY_MS\s*=\s*(\d+)", SRC).group(1))
    dead = int(re.search(r"SOCKET_DEAD_MS\s*=\s*(\d+)", SRC).group(1))
    assert ping * 2 <= dead, \
        "a single lost ping must not be enough to declare the socket dead"
    assert ping / 1000.0 < server.IDLE_HANGUP_S, \
        "the keepalive has to be more frequent than the hang-up it must not prevent"


def test_a_lost_line_is_shown_to_the_caller():
    """The turn was discarded in silence while the orb still read LISTENING, and the
    only sign was a word in a status line nobody on a phone is looking at."""
    fn = _extract(SRC, "endUtterance")
    assert "_lineDown" in fn
    assert "setCaption" in fn


def test_the_lost_line_notice_is_not_attributed_to_the_agent():
    """setCaption labels everything 'You' or 'AI'. A connection notice rendered as 'AI'
    puts words in the receptionist's mouth at exactly the moment the caller must not
    believe they were answered."""
    fn = _extract(SRC, "setCaption")
    assert "'sys'" in fn or '"sys"' in fn


def test_frame_errors_are_not_swallowed_silently():
    """A persistent throw in handleFrame means no turn is ever detected, with nothing
    anywhere to say so. Rate-limited, because at ~31 frames a second an unguarded
    console.error is the next bug."""
    fn = _extract(SRC, "pump")
    assert "console.error" in fn
    assert "%" in fn, "and rate-limited"


# ------------------------------------------------- getting onto the phone at all

MOBILE_SH = os.path.join(ROOT, "scripts", "run_mobile.sh")
OPENSSL = shutil.which("openssl")


def test_the_mobile_cert_names_the_address_you_actually_open():
    """Without this the phone path does not merely warn — it does not work.

    iOS/macOS have required subjectAltName since iOS 13 and ignore the Common Name
    entirely; Chrome has done the same since 58. The script issued
    `-subj "/CN=zensuvidha.local"` with no SAN at all, and you open it by LAN IP, so the
    certificate was wrong twice over. No secure context means `navigator.mediaDevices`
    is undefined and the page fails before it can say why.
    """
    sh = open(MOBILE_SH, encoding="utf-8").read()
    assert "subjectAltName" in sh
    assert "IP:$IP" in sh, "the SAN must name the LAN IP, not a hostname nobody types"
    # The IP has to be known BEFORE the certificate is issued for it. Compare the CODE,
    # not the file — the header comment explains all of this above either line.
    code = "\n".join(l for l in sh.split("\n") if not l.lstrip().startswith("#"))
    assert code.index("ipconfig getifaddr") < code.index("subjectAltName=IP:"), \
        "find the LAN IP before issuing a certificate that has to name it"


def test_the_mobile_cert_is_reissued_when_the_address_changes():
    """A laptop that moves between home and office Wi-Fi gets a new IP, and the cached
    certificate still names the old one — which fails on the phone with an error that
    says nothing about certificates."""
    sh = open(MOBILE_SH, encoding="utf-8").read()
    assert "-ext subjectAltName" in sh, "the cached cert must be checked against $IP"
    assert "checkend" in sh, "...and against its own expiry"


@pytest.mark.skipif(OPENSSL is None, reason="openssl is not installed")
def test_the_openssl_invocation_really_produces_that_san(tmp_path):
    """Run it, rather than trusting that the flags mean what they look like. `-addext`
    is silently ignored by some builds, which is exactly the failure this guards."""
    ip = "192.168.1.7"
    key, crt = tmp_path / "k.pem", tmp_path / "c.pem"
    r = subprocess.run([OPENSSL, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                        "-keyout", str(key), "-out", str(crt), "-days", "365",
                        "-subj", "/CN=%s" % ip,
                        "-addext", "subjectAltName=IP:%s,IP:127.0.0.1,DNS:localhost" % ip,
                        "-addext", "extendedKeyUsage=serverAuth"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    san = subprocess.run([OPENSSL, "x509", "-in", str(crt), "-noout",
                          "-ext", "subjectAltName"],
                         capture_output=True, text=True, timeout=60).stdout
    assert "IP Address:%s" % ip in san
    # And the guard in the script must not accept a prefix-similar address.
    assert "IP Address:192.168.1.70" not in san


def test_hanging_up_disarms_the_watchers_before_stopping_the_tracks():
    """`stop()` fires `ended`. A handler still pointed at a stream being torn down is
    how a deliberate hang-up races a recovery."""
    fn = _extract(SRC, "endCall")
    stop = fn.index("getTracks().forEach")
    disarm = fn.index("onended=")
    assert disarm < stop, "clear the track handlers BEFORE stopping the tracks"
    assert "releaseWake()" in fn
