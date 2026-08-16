"""A second recogniser must not be a second set of rules.

Every guard in stt.py exists because of a specific way a real call failed: a phantom
transcription of silence answered nobody, a latched combining mark ate a phone number,
"thank you for watching" was spoken back at a caller who said nothing. The danger with
a pluggable backend is not that the new one is worse — it is that it QUIETLY skips the
hardening, and the resulting failure looks like a bad model instead of a missing check.

These run without the binary or the 490MB model: the subprocess is stubbed, so the
contract is pinned on any machine.

Run:  pytest -q tests/test_stt_backends.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from zensuvidha import stt as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# the shared judge — one implementation, so the two backends cannot drift
# --------------------------------------------------------------------------- #
def test_unknown_confidence_is_not_the_same_as_confident():
    """whisper.cpp enforces the thresholds inside the binary and reports nothing back.
    Passing 0.0 for "I don't know" would read as certainty and switch the ambiguous
    half of the artifact list off without saying so."""
    kw = dict(reject_no_speech=0.85, reject_logprob=-1.6)
    # a phrase nobody ever says goes, with or without corroboration
    assert S.judge("subtitles by the amara.org community",
                   no_speech=None, avg_logprob=None, **kw) == ""
    # …and one that a caller genuinely might say survives when nothing corroborates it
    assert S.judge("bye bye", no_speech=None, avg_logprob=None, **kw) == "bye bye"
    # with evidence it looked like silence, the same phrase is dropped
    assert S.judge("bye bye", no_speech=0.7, avg_logprob=-0.2, **kw) == ""


def test_the_confidence_gate_still_runs_when_the_numbers_exist():
    kw = dict(reject_no_speech=0.85, reject_logprob=-1.6)
    assert S.judge("hello there", no_speech=0.9, avg_logprob=-0.1, **kw) == ""
    assert S.judge("hello there", no_speech=0.1, avg_logprob=-2.0, **kw) == ""
    assert S.judge("hello there", no_speech=0.1, avg_logprob=-0.1, **kw) == "hello there"


def test_a_runaway_tail_is_trimmed_not_dropped_on_either_backend():
    """The caller's phone number must survive Whisper latching onto a combining mark."""
    out = S.judge("मेरा नंबर है 8920429057ऽऽऽऽऽऽऽऽऽऽऽऽ",
                  no_speech=None, avg_logprob=None,
                  reject_no_speech=0.85, reject_logprob=-1.6)
    assert "8920429057" in out


# --------------------------------------------------------------------------- #
# whisper.cpp, with the binary stubbed
# --------------------------------------------------------------------------- #
class FakeRun:
    """Stands in for subprocess.run, writing the JSON whisper.cpp would produce."""

    def __init__(self, text="hello there", fail=None):
        self.text, self.fail, self.cmd = text, fail, None

    def __call__(self, cmd, **kw):
        self.cmd = cmd
        if self.fail == "timeout":
            raise subprocess.TimeoutExpired(cmd, 1)
        if self.fail == "error":
            raise subprocess.CalledProcessError(1, cmd, stderr=b"boom")
        of = cmd[cmd.index("-of") + 1]
        with open(of + ".json", "w", encoding="utf-8") as fh:
            json.dump({"transcription": [{"text": self.text}],
                       "result": {"language": "en"}}, fh)
        return subprocess.CompletedProcess(cmd, 0)


def _backend(tmp_path, **over):
    model = tmp_path / "ggml-small.bin"
    model.write_bytes(b"x")
    cfg = {"whispercpp_bin": sys.executable,        # any real file passes the check
           "whispercpp_model": str(model), "vad_filter": False}
    cfg.update(over)
    return S.WhisperCppSTT(cfg)


def _audio():
    return (np.sin(np.arange(16000) / 8) * 0.4).astype("float32")


def test_it_transcribes_and_reports_the_language(tmp_path, monkeypatch):
    b = _backend(tmp_path)
    fake = FakeRun("I would like to book an appointment")
    monkeypatch.setattr(subprocess, "run", fake)
    text, lang, _p = b.transcribe(_audio())
    assert text == "I would like to book an appointment"
    assert lang == "en"


def test_the_thresholds_are_handed_to_the_binary(tmp_path, monkeypatch):
    """They cannot be applied afterwards — the numbers only exist inside whisper.cpp.
    If they were not passed, the guard would be gone with nothing to show for it."""
    b = _backend(tmp_path, reject_no_speech=0.85, reject_logprob=-1.6)
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    b.transcribe(_audio())
    assert "-nth" in fake.cmd and fake.cmd[fake.cmd.index("-nth") + 1] == "0.85"
    assert "-lpt" in fake.cmd and fake.cmd[fake.cmd.index("-lpt") + 1] == "-1.6"


def test_auto_detect_is_passed_as_auto_not_as_empty(tmp_path, monkeypatch):
    """whisper.cpp reads an empty language as ENGLISH. `language: null` means detect,
    and getting that wrong flips every Indic call to the wrong language in silence."""
    b = _backend(tmp_path)
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    b.transcribe(_audio(), language=None)
    assert fake.cmd[fake.cmd.index("-l") + 1] == "auto"


def test_an_onnx_vad_model_is_never_passed(tmp_path, monkeypatch):
    """Handing --vad a Silero .onnx does not fail with a message — the binary ABORTS.
    This repo has three .onnx copies on disk and reaching for one is the obvious wrong
    move, so the candidate search must only ever yield ggml .bin files."""
    onnx = tmp_path / "silero_vad.onnx"
    onnx.write_bytes(b"x")
    cands = list(S.WhisperCppSTT._vad_candidates(ROOT, {}))
    assert all(c.endswith(".bin") for c in cands), cands


def test_vad_is_left_off_when_no_ggml_model_is_installed(tmp_path, monkeypatch):
    """Passing --vad with a path that is not there aborts the binary. On a machine
    where the VAD download has not been run, the recogniser must still work."""
    # The search is stubbed rather than pointed at a missing file: this repo now HAS a
    # real ggml-silero in models/whispercpp, so the fallback would legitimately find it
    # and the test would be asserting the developer's checkout, not the behaviour.
    monkeypatch.setattr(S.WhisperCppSTT, "_vad_candidates",
                        staticmethod(lambda root, cfg: iter(())))
    b = _backend(tmp_path, vad_filter=True)
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    b.transcribe(_audio())
    assert "--vad" not in fake.cmd, "a missing VAD model would abort the process"


def test_vad_is_switched_on_when_the_ggml_model_is_there(tmp_path, monkeypatch):
    vad = tmp_path / "ggml-silero-v5.1.2.bin"
    vad.write_bytes(b"x")
    b = _backend(tmp_path, vad_filter=True, whispercpp_vad_model=str(vad))
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    b.transcribe(_audio())
    assert "--vad" in fake.cmd and fake.cmd[fake.cmd.index("-vm") + 1] == str(vad)


def test_vad_is_off_entirely_when_the_config_says_so(tmp_path, monkeypatch):
    b = _backend(tmp_path, vad_filter=False)
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    b.transcribe(_audio())
    assert "--vad" not in fake.cmd


def test_a_wedged_binary_drops_the_turn_instead_of_hanging(tmp_path, monkeypatch):
    """Same reasoning as the `say` timeout: a child that never returns holds a
    threadpool worker forever, and enough of them stop the server answering anybody."""
    b = _backend(tmp_path)
    monkeypatch.setattr(subprocess, "run", FakeRun(fail="timeout"))
    assert b.transcribe(_audio()) == ("", None, 0.0)


def test_a_crash_drops_the_turn_rather_than_raising(tmp_path, monkeypatch):
    b = _backend(tmp_path)
    monkeypatch.setattr(subprocess, "run", FakeRun(fail="error"))
    assert b.transcribe(_audio()) == ("", None, 0.0)


def test_the_hardening_applies_to_this_backend_too(tmp_path, monkeypatch):
    """The whole point of the shared judge."""
    b = _backend(tmp_path)
    monkeypatch.setattr(subprocess, "run", FakeRun("Subtitles by the amara.org community"))
    assert b.transcribe(_audio())[0] == ""


def test_an_unknown_provider_falls_back_rather_than_disabling_voice(monkeypatch):
    """A typo in config, or a machine without the binary, must not silently leave the
    caller with no speech input at all."""
    monkeypatch.setattr(S, "WhisperCppSTT",
                        lambda cfg: (_ for _ in ()).throw(RuntimeError("no binary")))
    made = {}
    monkeypatch.setattr(S, "FasterWhisperSTT", lambda cfg: made.setdefault("fw", object()))
    assert S.get_stt({"provider": "whisper_cpp"}) is made["fw"]


def test_none_still_means_no_speech_input():
    assert S.get_stt({"provider": "none"}) is None


# --------------------------------------------------------------------------- #
# provider: auto — take the faster one when the machine actually has it
# --------------------------------------------------------------------------- #
def test_auto_prefers_the_faster_backend_when_it_is_installed(monkeypatch):
    """Measured 1.73-1.87x on the dominant cost of the audio path, at slightly better
    WER. Where it exists there is no argument for the slower one."""
    made = {}
    monkeypatch.setattr(S, "WhisperCppSTT", lambda cfg: made.setdefault("wc", object()))
    monkeypatch.setattr(S, "FasterWhisperSTT",
                        lambda cfg: pytest.fail("the slower backend was chosen"))
    assert S.get_stt({"provider": "auto"}) is made["wc"]


def test_auto_falls_back_silently_when_it_is_not(monkeypatch):
    """The premise of this project is that a clone runs. `auto` must never turn a
    missing optional dependency into a machine with no speech input."""
    monkeypatch.setattr(S, "WhisperCppSTT",
                        lambda cfg: (_ for _ in ()).throw(RuntimeError("no binary")))
    made = {}
    monkeypatch.setattr(S, "FasterWhisperSTT", lambda cfg: made.setdefault("fw", object()))
    assert S.get_stt({"provider": "auto"}) is made["fw"]


def test_auto_does_not_override_an_explicit_choice(monkeypatch):
    """Someone who pinned faster_whisper wants faster_whisper — most likely because
    they are comparing the two, which is exactly when a helpful swap is worst."""
    monkeypatch.setattr(S, "WhisperCppSTT",
                        lambda cfg: pytest.fail("an explicit provider was overridden"))
    made = {}
    monkeypatch.setattr(S, "FasterWhisperSTT", lambda cfg: made.setdefault("fw", object()))
    assert S.get_stt({"provider": "faster_whisper"}) is made["fw"]


# --------------------------------------------------------------------------- #
# regressions from the review of this work
# --------------------------------------------------------------------------- #
def test_a_punctuation_only_transcript_is_dropped_without_confidence_numbers():
    """THE ONE THAT MATTERED. `.` and `...` live in _MAYBE_SAID, which only_certain
    skips — so on every whisper.cpp turn a silent clip transcribing as "." came back
    TRUTHY, became a real caller turn, and the agent answered nothing at all. That is
    precisely the "randomly speaking" failure the guard was written for."""
    kw = dict(no_speech=None, avg_logprob=None,
              reject_no_speech=0.85, reject_logprob=-1.6)
    for junk in (".", "...", "।", "…", " . ", "a", "-"):
        assert S.judge(junk, **kw) == "", repr(junk)


def test_real_short_answers_still_survive_without_confidence_numbers():
    """The fix must not take the caller's shortest real turns with it — those are the
    ones the whole endpointer is tuned around."""
    kw = dict(no_speech=None, avg_logprob=None,
              reject_no_speech=0.85, reject_logprob=-1.6)
    for real in ("yes", "haan", "हाँ", "అవును", "ok", "bye bye", "10 am"):
        assert S.judge(real, **kw) == real, repr(real)


def test_an_unquantified_language_detection_is_not_reported_as_zero_confidence(
        tmp_path, monkeypatch):
    """whisper.cpp names a language and gives no score for it. Reporting 0.0 made the
    orchestrator discard the detection and guess from the script instead — which cannot
    separate Marathi from Hindi, the exact case Whisper's own detection is there for."""
    b = _backend(tmp_path)
    monkeypatch.setattr(subprocess, "run", FakeRun("नमस्ते"))
    _text, lang, prob = b.transcribe(_audio())
    assert lang == "en"
    assert prob is None, "0.0 reads as 'certainly not', which is not what happened"


def test_an_unquantified_detection_does_not_crash_the_reply_language():
    """`_last_det[1] >= 0.6` on a None is a TypeError, not a fallback — it would take
    down every reply-language decision on that backend."""
    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack
    s = Session(load_pack("clinic"), llm=None, stt=None, tts=None)
    s._last_det = ("hi", None)
    s.reply_language("नमस्ते")            # must not raise


def test_the_binary_is_serialised(tmp_path, monkeypatch):
    """The comment on _lock promises one process at a time; it was never acquired. Six
    concurrent turns spawning six whisper-cli processes contend for one GPU, and the
    measured 1.8x that justifies this backend evaporates when the box is busiest."""
    import inspect
    src = inspect.getsource(S.WhisperCppSTT.transcribe)
    assert "with self._lock:" in src, "the subprocess is not serialised"


# --------------------------------------------------------------------------- #
# the resident-model server, and picking the best local device
# --------------------------------------------------------------------------- #
def test_auto_prefers_the_server_over_the_cli(monkeypatch):
    """Measured 621ms against 841ms at 3.5s — and unlike the CLI it reports the
    confidence numbers, so it is the more accurate of the two as well as the faster."""
    order = []
    monkeypatch.setattr(S, "WhisperCppServerSTT",
                        lambda cfg: order.append("server") or object())
    monkeypatch.setattr(S, "WhisperCppSTT",
                        lambda cfg: pytest.fail("the CLI was preferred to the server"))
    S.get_stt({"provider": "auto"})
    assert order == ["server"]


def test_auto_walks_down_the_ladder_when_a_step_is_missing(monkeypatch):
    """A machine without whisper-server but with whisper-cli still gets the faster one;
    a machine with neither still gets a working recogniser."""
    monkeypatch.setattr(S, "WhisperCppServerSTT",
                        lambda cfg: (_ for _ in ()).throw(RuntimeError("no server")))
    made = {}
    monkeypatch.setattr(S, "WhisperCppSTT", lambda cfg: made.setdefault("cli", object()))
    assert S.get_stt({"provider": "auto"}) is made["cli"]

    monkeypatch.setattr(S, "WhisperCppSTT",
                        lambda cfg: (_ for _ in ()).throw(RuntimeError("no cli")))
    monkeypatch.setattr(S, "FasterWhisperSTT", lambda cfg: made.setdefault("fw", object()))
    assert S.get_stt({"provider": "auto"}) is made["fw"]


def test_a_pinned_device_is_never_second_guessed():
    assert S.best_local_device("cpu") == "cpu"
    assert S.best_local_device("cuda") == "cuda"
    assert S.best_local_device("mps") == "mps"      # somebody who typed it meant it


def test_auto_never_resolves_to_mps():
    """MEASURED: SpeechBrain's ECAPA fails outright on mps because it keeps internal
    CPU tensors, and CTranslate2 does not use it at all. Apple's GPU is reached through
    whisper.cpp's Metal backend, which is a separate process."""
    assert S.best_local_device(None) in ("cpu", "cuda")
    assert S.best_local_device("auto") in ("cpu", "cuda")


def test_thread_count_stays_off_the_efficiency_cores():
    """MEASURED on this M1: auto 1691ms, 4 threads 1601ms, 8 threads 2212ms, 10 threads
    3798ms. More threads is WORSE, so 'all of them' is the wrong default."""
    n = S.best_thread_count({})
    assert 2 <= n <= 8, n
    assert S.best_thread_count({"whispercpp_threads": 3}) == 3


def test_the_language_name_is_converted_to_a_code():
    """whisper.cpp says 'english' where faster-whisper says 'en'. Everything downstream
    — the language lock, the pack's a_hi/a_te answers — is keyed on CODES, so a name
    here means the lock silently never matches."""
    assert S._lang_code("english") == "en"
    assert S._lang_code("Hindi") == "hi"
    assert S._lang_code("telugu") == "te"
    assert S._lang_code("en") == "en"          # already a code
    assert S._lang_code(None) is None
    assert S._lang_code("klingon") is None     # unknown, not passed through as a name


# --------------------------------------------------------------------------- #
# two-pass: a tiny model for the guess, the accurate one for the turn
# --------------------------------------------------------------------------- #
class _Spy:
    def __init__(self, text, language=None):
        self.text, self.language, self.calls = text, language, []

    def transcribe(self, audio, hint=None, language="__cfg__", fast=None, denoise=None):
        self.calls.append({"fast": fast, "denoise": denoise})
        return self.text, "en", 0.9


def test_a_guess_goes_to_the_tiny_model_and_a_turn_does_not():
    """MEASURED: 171ms against 722ms for the same clip. The guess exists to size the
    endpoint window and show live text, and all three turn-taking signals computed from
    it were gated behind a full recognition."""
    commit, part = _Spy("the real transcript"), _Spy("the guess")
    two = S.TwoPassSTT(commit, part)
    assert two.transcribe(b"x", partial=True)[0] == "the guess"
    assert two.transcribe(b"x")[0] == "the real transcript"
    assert len(part.calls) == 1 and len(commit.calls) == 1


def test_the_partial_model_never_becomes_the_transcript():
    """`tiny` is measurably worse and its errors must not reach a booking. The engine
    enforces this too — nothing reaches the LLM until the endpoint is confirmed — but a
    provider that returned the guess for a committed turn would defeat that."""
    commit, part = _Spy("nine eight two zero"), _Spy("nine eight too zero")
    two = S.TwoPassSTT(commit, part)
    assert two.transcribe(b"x")[0] == "nine eight two zero"


def test_the_language_is_read_off_the_model_that_makes_the_transcript():
    commit, part = _Spy("x", language="hi"), _Spy("x", language="en")
    assert S.TwoPassSTT(commit, part).language == "hi"


def test_a_provider_that_predates_the_partial_kwarg_still_works():
    """stt.py is a pluggable interface; adding a required kwarg broke five tests and
    every custom adapter once already."""
    class Old:
        language = None

        def transcribe(self, audio, hint=None, language="__cfg__", fast=None):
            return "old provider", "en", 0.9
    two = S.TwoPassSTT(Old(), Old())
    assert two.transcribe(b"x", partial=True)[0] == "old provider"


def test_no_partial_model_configured_means_one_model_for_both(monkeypatch):
    made = []
    monkeypatch.setattr(S, "_build_stt", lambda cfg: made.append(cfg) or object())
    out = S.get_stt({"provider": "faster_whisper"})
    assert not isinstance(out, S.TwoPassSTT)
    assert len(made) == 1


def test_a_missing_partial_model_degrades_to_one_model_not_to_none(monkeypatch):
    """A file that is not there must cost the two-pass speedup, never voice input."""
    calls = {"n": 0}

    def build(cfg):
        calls["n"] += 1
        if cfg.get("whispercpp_model") == "nope.bin":
            raise RuntimeError("missing")
        return object()
    monkeypatch.setattr(S, "_build_stt", build)
    out = S.get_stt({"provider": "auto", "partial_model": "nope.bin"})
    assert out is not None and not isinstance(out, S.TwoPassSTT)


def test_closing_the_pair_closes_both_children():
    """Two whisper-server children now, and neither may outlive the app — the first
    version of the single one left 490MB listening after shutdown."""
    closed = []

    class C:
        language = None

        def close(self):
            closed.append(1)
    S.TwoPassSTT(C(), C()).close()
    assert len(closed) == 2


def test_two_pass_forwards_the_whole_provider_surface():
    """THE ONE THAT WOULD HAVE CAUGHT IT.

    `TwoPassSTT` wraps two providers and has to be a complete stand-in for one. It was
    not: `_decode` was missing, and `Session.clean_audio` passes `self.stt._decode` into
    the audio pipeline — so EVERY microphone turn raised AttributeError before reaching
    the recogniser. The agent stopped hearing anyone at all, and the failure was buried
    in a threadpool traceback rather than surfacing as a dropped turn.

    The list is DERIVED from the source rather than written out here. A hand-kept list
    would have been just as incomplete as the wrapper it is checking — the next
    attribute the engine starts reading would be missing from both.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "zensuvidha"
    wanted = set()
    for src in root.glob("*.py"):
        text = src.read_text(encoding="utf-8")
        wanted |= set(re.findall(r"(?:self\.stt|session\.stt|_stt)\.([A-Za-z_]\w*)", text))
        wanted |= set(re.findall(
            r'getattr\(\s*(?:self\.stt|session\.stt|_stt)\s*,\s*"([A-Za-z_]\w*)"', text))
    wanted -= {"transcribe"}          # exercised directly by the tests above
    assert wanted, "the scan found nothing — has the engine stopped using the provider?"

    class Provider:
        language = "en"
        denoiser = None

        def _decode(self, audio):
            return audio

        def transcribe(self, audio, **kw):
            return "x", "en", 1.0

        def close(self):
            pass

    two = S.TwoPassSTT(Provider(), Provider())
    missing = [a for a in sorted(wanted) if not hasattr(two, a)]
    assert not missing, (
        "TwoPassSTT does not forward %s — the engine reads it off the provider, so a "
        "turn using it will raise instead of being transcribed" % missing)


def test_a_phone_turn_is_partialled_by_the_accurate_model():
    """MEASURED: the tiny partial model changes the endpoint decision on 3 of 8 shapes,
    and the disagreement is not random —

        "I would like to book an"             tiny HOLD    small normal   tiny right
        "my mobile number is 9820429057"      tiny normal  small CLOSE    small right
        "my mobile number is eight nine two"  tiny HOLD    small normal   tiny right

    Tiny is BETTER on dangling words and WORSE on digits. It mishears them, so
    `looks_complete` never sees a full ten-digit number and a caller who has just
    finished reading one out sits through the whole window instead of being confirmed.
    Digits are also where the window is widest (phone gets +600ms), so the error costs
    most exactly where it occurs."""
    from zensuvidha.orchestrator import Session
    from zensuvidha.packs import load_pack

    calls = []

    class Spy:
        language = None
        denoiser = None

        def _decode(self, a):
            return a

        def transcribe(self, audio, **kw):
            calls.append(kw.get("partial", False))
            return "x y z", "en", 0.9

    s = Session(load_pack("clinic"), llm=None, stt=Spy(), tts=None)
    s.transcribe(b"a", partial=True)
    assert calls[-1] is True, "an ordinary partial did not use the cheap model"

    s.pending_slot = "phone"
    s.transcribe(b"a", partial=True)
    assert calls[-1] is False, "a phone partial was answered by the model that mishears digits"

    s.pending_slot = "name"
    s.transcribe(b"a", partial=True)
    assert calls[-1] is True, "only phone turns should pay for the accurate model"
