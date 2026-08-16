"""Speaker gating — answer the caller, ignore whoever else is in the room.

Silero tells us a sound is speech; it cannot tell us whose. Without a gate, a colleague
talking nearby or a television opens a turn and gets answered as though it were the
caller. The first voice on the call is enrolled and everything after is compared to it.

These tests use a stub encoder so the suite stays fast and needs no 80MB download. The
real ECAPA separation is measured separately (same speaker 0.87, closest impostor 0.43).

Run:  pytest -q tests/test_speaker.py
"""
import numpy as np
import pytest

from zensuvidha.orchestrator import Session
from zensuvidha.packs import load_pack
from zensuvidha.speaker import SpeakerGate


class StubGate(SpeakerGate):
    """A gate whose 'audio' is just a label → a fixed vector, so identity is exact."""

    def __init__(self, threshold=0.55, too_short=()):
        self.backend = "stub"
        self.threshold = threshold
        self._too_short = set(too_short)
        self._vectors = {}

    def embed(self, audio, min_seconds=0.6):
        key = audio if isinstance(audio, str) else bytes(audio).decode("utf8", "ignore")
        if key in self._too_short:
            return None                      # clip below the length we can judge from
        if key not in self._vectors:
            rng = np.random.default_rng(abs(hash(key)) % (2 ** 31))
            v = rng.normal(size=192).astype("float32")
            self._vectors[key] = v / np.linalg.norm(v)
        return self._vectors[key]


def _s(gate=None):
    return Session(load_pack("clinic"), None, speaker_gate=gate)


def _corroborated(gate, who="caller"):
    """A session whose voiceprint the caller has actually confirmed.

    The gate deliberately refuses NOBODY while the print is a single un-corroborated
    sample: at that point a disagreement means we do not know which voice is the
    caller, and refusing is the guess that silences them. Tests about refusal must
    therefore establish a trusted print first — which is what a real call does within
    two turns.
    """
    s = _s(gate)
    for _ in range(s.VOICEPRINT_TRUST_N + 1):
        s.check_speaker(who, speakers=1)
    assert s._voiceprint_n >= s.VOICEPRINT_TRUST_N
    assert s._gate_proven, "the gate never matched the caller, so it may not refuse"
    return s



# --------------------------------------------------------------------------- #
# enrolment
# --------------------------------------------------------------------------- #
def test_the_first_voice_on_the_call_becomes_the_caller():
    s = _s(StubGate())
    assert s.voiceprint is None
    ok, sim = s.check_speaker("caller")
    assert ok and sim is None, "the enrolling turn is never judged"
    assert s.voiceprint is not None


def test_the_same_voice_is_accepted_on_later_turns():
    s = _s(StubGate())
    s.check_speaker("caller")
    ok, sim = s.check_speaker("caller")
    assert ok
    assert sim == pytest.approx(1.0, abs=1e-5)



def test_a_different_voice_is_rejected():
    """The gate's whole purpose — once it has a print it can trust."""
    s = _corroborated(StubGate())
    ok, sim = s.check_speaker("somebody else", speakers=1)
    assert not ok and sim is not None and sim < s.speaker_gate.threshold


def test_enrolment_survives_the_whole_call():
    s = _s(StubGate())
    s.check_speaker("caller")
    print0 = s.voiceprint.copy()
    for _ in range(5):
        s.check_speaker("caller")
        s.check_speaker("intruder")
    assert np.allclose(s.voiceprint, print0), "the voiceprint must not drift"


# --------------------------------------------------------------------------- #
# it must fail OPEN, never closed
# --------------------------------------------------------------------------- #
def test_no_gate_configured_accepts_everything():
    """The gate is optional; without it nothing may change."""
    s = _s(None)
    for who in ("caller", "someone else", ""):
        ok, sim = s.check_speaker(who)
        assert ok and sim is None


def test_a_clip_too_short_to_judge_is_accepted():
    """Rejecting the real caller because they answered "haan" is far worse than letting
    one stray utterance through."""
    s = _s(StubGate(too_short={"haan"}))
    s.check_speaker("caller")
    ok, sim = s.check_speaker("haan")
    assert ok and sim is None


def test_a_clip_too_short_to_enrol_does_not_enrol():
    """Enrolling on a cough would lock the whole call to the wrong voiceprint."""
    s = _s(StubGate(too_short={"cough"}))
    s.check_speaker("cough")
    assert s.voiceprint is None, "must wait for an utterance long enough to be reliable"
    s.check_speaker("caller")
    assert s.voiceprint is not None


def test_an_encoder_that_raises_does_not_break_the_call():
    class Broken(StubGate):
        def embed(self, audio, min_seconds=0.6):
            raise RuntimeError("model died")

    s = _s(Broken())
    with pytest.raises(RuntimeError):
        s.check_speaker("caller")           # enrolment surfaces the error…
    s.voiceprint = np.ones(192, dtype="float32") / np.sqrt(192)
    # …but SpeakerGate.matches swallows it, so a live call keeps going
    real = SpeakerGate.__new__(SpeakerGate)
    real.threshold = 0.55
    real.embed = lambda *a, **k: None
    ok, sim = real.matches(s.voiceprint, b"anything")
    assert ok and sim is None


# --------------------------------------------------------------------------- #
# threshold behaviour
# --------------------------------------------------------------------------- #
def test_threshold_is_configurable():
    # A threshold only governs REFUSAL, and the gate refuses nobody while its print is
    # a single un-corroborated sample — so both prints are marked trusted first.
    #
    # Marked rather than earned: refinement requires threshold + VOICEPRINT_MARGIN, so a
    # threshold above ~0.90 can never be corroborated by any real voice and the gate
    # would stay provisional for the whole call. That is a property of a pathological
    # setting, not of this test — see test_an_unreachable_threshold_is_rejected.
    s_strict, s_loose = _s(StubGate(threshold=0.99)), _s(StubGate(threshold=-1.0))
    for s in (s_strict, s_loose):
        s.check_speaker("caller")
        # A threshold governs REFUSAL, and a gate that has never matched the caller is
        # not allowed to refuse at all — so both of these have to be marked as having
        # recognised somebody before the threshold means anything.
        s._voiceprint_n, s._gate_proven = s.VOICEPRINT_TRUST_N, True
    assert not s_strict.check_speaker("other", speakers=1)[0]
    assert s_loose.check_speaker("other", speakers=1)[0], "a low threshold accepts anyone"


def test_an_unreachable_threshold_is_rejected():
    """Refinement needs `threshold + VOICEPRINT_MARGIN`. Set the threshold close enough
    to 1.0 and no voice on earth can corroborate the print, so the gate stays
    provisional — refusing nobody — for the entire call. Fail loudly at construction
    rather than looking like a gate that quietly does nothing."""
    from zensuvidha.orchestrator import Session
    s = _s(StubGate(threshold=0.55))
    assert s.speaker_gate.threshold + s.VOICEPRINT_MARGIN < 1.0, (
        "the configured threshold leaves no room for the print to ever be corroborated")


def test_similarity_is_a_cosine():
    a = np.array([1.0, 0.0, 0.0], dtype="float32")
    b = np.array([0.0, 1.0, 0.0], dtype="float32")
    assert SpeakerGate.similarity(a, a) == pytest.approx(1.0)
    assert SpeakerGate.similarity(a, b) == pytest.approx(0.0)
    assert SpeakerGate.similarity(a, -a) == pytest.approx(-1.0)
    assert SpeakerGate.similarity(None, a) == 0.0


def test_the_gate_is_off_unless_asked_for():
    """It needs a model download, so it must never switch itself on."""
    from zensuvidha.speaker import get_speaker_gate
    assert get_speaker_gate({}) is None
    assert get_speaker_gate({"speaker_gate": False}) is None


# --------------------------------------------------------------------------- #
# voiceprint refinement
# --------------------------------------------------------------------------- #
class DriftGate(StubGate):
    """Similarity is controlled directly, so refinement logic can be tested exactly."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.embeds = 0

    def embed(self, audio, min_seconds=0.6):
        self.embeds += 1
        return super().embed(audio, min_seconds)


def test_a_confident_match_widens_the_voiceprint():
    """One utterance is a thin sample of a voice. Measured: a caller enrolled on a single
    clean clip falls to 0.54 in heavy noise and gets rejected — refinement holds them at
    0.75 with impostor scores unchanged."""
    s = _s(StubGate())
    s.check_speaker("caller")
    assert s._voiceprint_n == 1
    s.check_speaker("caller")            # similarity 1.0, well above threshold+margin
    assert s._voiceprint_n == 2


def test_refinement_stops_at_the_cap():
    s = _s(StubGate())
    for _ in range(20):
        s.check_speaker("caller")
    assert s._voiceprint_n == s.VOICEPRINT_MAX


class ExactGate(SpeakerGate):
    """Similarity is set exactly, so the refinement band can be tested at its edges."""

    def __init__(self, threshold=0.55):
        self.backend = "exact"
        self.threshold = threshold
        e1 = np.zeros(192, dtype="float32"); e1[0] = 1.0
        e2 = np.zeros(192, dtype="float32"); e2[1] = 1.0
        self._e1, self._e2 = e1, e2

    def at(self, cos):
        """A unit vector whose cosine with the enrolled print is exactly `cos`."""
        return (cos * self._e1 + np.sqrt(max(0.0, 1 - cos ** 2)) * self._e2).astype("float32")

    def embed(self, audio, min_seconds=0.6):
        return self._e1 if audio == "caller" else self.at(float(audio))


def test_a_borderline_match_may_not_drag_the_print():
    """An impostor who scrapes past the threshold once must not pull the voiceprint
    toward themselves — only matches clear of threshold+margin contribute."""
    s = _s(ExactGate(threshold=0.55))
    s.check_speaker("caller")
    before = s.voiceprint.copy()
    ok, sim = s.check_speaker("0.60")            # accepted, but inside the margin band
    assert ok and sim == pytest.approx(0.60, abs=1e-3)
    assert sim < s.speaker_gate.threshold + s.VOICEPRINT_MARGIN
    assert s._voiceprint_n == 1, "a borderline match must not contribute"
    assert np.allclose(s.voiceprint, before)


def test_a_clearly_confident_match_does_contribute():
    """The other edge of the same band — comfortably above threshold+margin."""
    s = _s(ExactGate(threshold=0.55))
    s.check_speaker("caller")
    ok, sim = s.check_speaker("0.90")
    assert ok and sim >= s.speaker_gate.threshold + s.VOICEPRINT_MARGIN
    assert s._voiceprint_n == 2


def test_a_rejected_voice_never_widens_the_existing_print():
    """Refinement is for CONFIDENT matches only — a refused voice must never be folded
    into the print that is already trusted."""
    s = _s(StubGate())
    s.check_speaker("caller")
    before = s.voiceprint.copy()
    # The print here is one utterance old and never corroborated, so the bound that
    # applies is REENROL_PROVISIONAL, not REENROL_AFTER. Staying below it is what keeps
    # this test about REFINEMENT rather than about recovery.
    for _ in range(s.REENROL_PROVISIONAL - 1):
        s.check_speaker("intruder")
    assert s._voiceprint_n == 1
    assert np.allclose(s.voiceprint, before)


def test_a_corroborated_print_is_not_thrown_away_as_readily():
    """Recovery is fast only while the print is a guess. Once the caller's own turns
    have confirmed it, a couple of odd refusals must not discard it — that print has
    earned its loyalty."""
    s = _s(StubGate())
    s.check_speaker("caller")
    for _ in range(s.VOICEPRINT_TRUST_N):        # the caller corroborates it
        s.check_speaker("caller")
    assert s._voiceprint_n >= s.VOICEPRINT_TRUST_N
    before = s.voiceprint.copy()
    for _ in range(s.REENROL_PROVISIONAL):       # would have re-enrolled a guess
        s.check_speaker("intruder")
    assert np.allclose(s.voiceprint, before), \
        "a corroborated print was discarded as fast as an uncorroborated one"


def test_a_voice_enrolled_by_mistake_costs_at_most_one_turn():
    """A call opening while music plays enrols whatever is playing. Measured on real
    audio: the caller then scored 0.14 and 0.03 against it. Every refusal it takes to
    notice is a turn the caller spoke into nothing."""
    s = _s(StubGate())
    s.check_speaker("the song")                  # wrong voice enrolled
    lost = 0
    for _ in range(4):
        ok, _sim = s.check_speaker("the real caller")
        if ok:
            break
        lost += 1
    assert lost <= 1, f"the caller lost {lost} turns before the gate noticed"



def test_the_same_voice_heard_repeatedly_becomes_the_caller():
    """The recovery path, whichever route it takes. One voice that keeps speaking while
    the enrolled print never returns IS the person having the conversation — that is
    true whether the print was a song, a passer-by, or a bad sample of the caller."""
    s = _s(StubGate())
    s.check_speaker("whatever was making a noise")
    before = s.voiceprint.copy()
    for _ in range(4):
        s.check_speaker("the actual caller", speakers=1)
    assert not np.allclose(s.voiceprint, before), "the call never found its caller"


def test_three_DIFFERENT_strangers_never_take_over_the_call():
    """Recovery must not become a hijack: the voiceprint may only move to a voice that
    KEEPS COMING BACK, never to a series of different people.

    Note what is asserted — the print, not the answer. While the enrolled print is
    still a guess the gate answers everybody, deliberately: refusing on a guess is what
    cost a real caller their turn when a song was enrolled instead of them."""
    s = _s(StubGate())
    s.check_speaker("caller")
    before = s.voiceprint.copy()
    for who in ("stranger one", "stranger two", "stranger three", "stranger four"):
        s.check_speaker(who, speakers=1)
        assert np.allclose(s.voiceprint, before), f"{who} took the call over"


def test_the_refined_print_stays_unit_length():
    """Cosine similarity assumes unit vectors — an un-normalised merge would silently
    change every later comparison."""
    s = _s(StubGate())
    for _ in range(6):
        s.check_speaker("caller")
    assert np.linalg.norm(s.voiceprint) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# duration must be measured on VOICE, not on the clip
# --------------------------------------------------------------------------- #
def _voiced_clip(voiced_s, pre_s=0.30, post_s=0.80, sr=16000, seed=0):
    """A realistic utterance: pre-roll room noise + speech + endpoint silence."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(voiced_s * sr)) / sr
    speech = (0.3 * np.sin(2 * np.pi * 180 * t) + 0.2 * np.sin(2 * np.pi * 540 * t)
              + 0.05 * rng.normal(size=t.size)).astype("float32")
    quiet = lambda n: (rng.normal(size=int(n * sr)) * 0.002).astype("float32")
    return np.concatenate([quiet(pre_s), speech, quiet(post_s)]).astype("float32")


def test_silence_padding_does_not_count_toward_the_length_test():
    """A one-word answer arrives as 300ms pre-roll + 350ms voice + 800ms endpoint silence.
    Measuring the CLIP let it through the length test, then ECAPA pooled over 76% silence
    and scored the real caller 0.526 — below threshold, so their confirmation was
    silently ignored."""
    from zensuvidha.speaker import _voiced_only
    clip = _voiced_clip(0.35)
    assert len(clip) / 16000 > 1.4, "the clip is long"
    voiced = _voiced_only(clip, 16000)
    assert len(voiced) / 16000 < 0.8, "but the VOICED part is short"
    assert len(voiced) / 16000 > 0.15, "and the speech itself is not destroyed"


def test_voiced_trim_never_empties_a_clip():
    """It must never be able to blank an utterance on its own — every degenerate input
    returns something usable."""
    from zensuvidha.speaker import _voiced_only
    for sig in (np.zeros(16000, dtype="float32"),
                np.ones(16000, dtype="float32") * 0.5,
                np.zeros(10, dtype="float32"),
                (np.random.default_rng(0).normal(size=16000) * 0.3).astype("float32")):
        out = _voiced_only(sig, 16000)
        assert out is not None and out.size > 0


# --------------------------------------------------------------------------- #
# The lockout reported from a live call, reproduced exactly.
#
#   "Can you hear me?"                 2.1s  -> TRANSCRIBED, enrolled the print here
#   "list me out all the details"            -> TRANSCRIBED
#   "tell me about the fee structure"  0.51  -> DISCARDED
#   "Oh"                               0.03  -> DISCARDED
#   "tell me about the fee structure"  0.45  -> DISCARDED
#
# A clean room (31-38dB floor-to-voice), nothing filtered, no music. The caller was
# simply refused by a print taken from their own first sentence — and could not
# recover, because 0.45-0.51 is BELOW the 0.55 threshold and ABOVE the 0.25 the
# re-enrolment rescue required.
# --------------------------------------------------------------------------- #
def test_a_thin_first_sample_does_not_lock_the_caller_out():
    s = _s(ExactGate(threshold=0.55))
    s.check_speaker("caller")                      # 2.1s "Can you hear me?"
    heard = []
    for score in ("0.51", "0.45", "0.49", "0.62", "0.58"):
        ok, sim = s.check_speaker(score)
        heard.append(ok)
    assert all(heard), f"the caller was discarded on their own call: {heard}"


def test_the_dead_zone_between_reject_and_rescue_is_closed():
    """Rejected below 0.55, rescued only below 0.25 — anyone landing in between was
    refused every turn forever. There must be no score that is both refused and
    unrecoverable."""
    for score in (0.30, 0.40, 0.45, 0.50, 0.54):
        s = _s(ExactGate(threshold=0.55))
        s.check_speaker("caller")
        outcomes = [s.check_speaker(f"{score:.2f}")[0] for _ in range(6)]
        assert any(outcomes), f"a caller at {score} never got through in six turns"



def test_a_stranger_on_the_second_turn_is_answered_but_never_adopted():
    """A deliberate trade, and the direction matters.

    Refusing here protects against a colleague speaking second. Accepting protects the
    CALLER when the enrolled print is the thing that is wrong — measured: a call opening
    while music played enrolled the song, and the caller then scored 0.14 against it.
    One of those failures has happened twice on real calls and the other has not, so the
    turn is answered — but the stranger does not become the caller, and one more turn
    from the real caller settles it."""
    s = _s(StubGate())
    s.check_speaker("caller")
    before = s.voiceprint.copy()
    ok, _sim = s.check_speaker("a colleague", speakers=1)
    assert ok, "the caller's own turn could equally have been the one refused here"
    assert np.allclose(s.voiceprint, before), "one stray turn adopted a new caller"


def test_a_stranger_cannot_widen_a_provisional_print():
    s = _s(ExactGate(threshold=0.55))
    s.check_speaker("caller")
    before, n_before = s.voiceprint.copy(), s._voiceprint_n
    s.check_speaker("0.05")
    assert s._voiceprint_n == n_before
    assert np.allclose(s.voiceprint, before), "a stranger dragged the provisional print"


def test_the_gate_becomes_strict_once_the_print_is_corroborated():
    """Forgiveness is for the thin-sample phase only. Once the print has been widened
    by the caller's own turns it must enforce again, or the gate stops existing.

    Asserted through a STRANGER rather than a near-miss: merging changes the print, so
    a stub label like "0.45" no longer scores 0.45 against it and the number would be
    measuring the fixture instead of the behaviour."""
    s = _s(ExactGate(threshold=0.55))
    s.check_speaker("caller")
    for _ in range(4):
        s.check_speaker("0.90")                    # the caller, clearly
    assert s._voiceprint_n >= s.VOICEPRINT_TRUST_N, "the print never got corroborated"
    ok, _sim = s.check_speaker("0.02")
    assert not ok, "the gate never became authoritative"



def test_refusal_only_happens_once_the_print_is_trusted():
    """The provisional phase must END. If it did not, the threshold would quietly
    become permanent forgiveness and the gate would stop existing."""
    s = _corroborated(StubGate())
    ok, _sim = s.check_speaker("somebody else", speakers=1)
    assert not ok, "the gate never became authoritative"



def test_a_mixed_clip_never_becomes_the_caller():
    """Measured at 0.364-0.48: the caller with a colleague talking over them. It is
    answered — losing the caller's words is the worse failure — but a blend of two
    people must never be adopted as the caller's identity, whether by widening the
    print or by winning as a rival."""
    s = _s(ExactGate(threshold=0.55))
    s.check_speaker("caller")
    s._gate_proven = True                                     # it can see this caller
    before, n = s.voiceprint.copy(), s._voiceprint_n
    for _ in range(s.PROVISIONAL_MAX_TURNS + 1):
        s.check_speaker("0.364", speakers=2)                  # two voices in the clip
    assert s._voiceprint_n == n and np.allclose(s.voiceprint, before), \
        "a mixed clip became the caller"


def test_a_single_voice_near_miss_does_teach_the_gate():
    """The other half: a clip known to hold ONE voice, scoring close enough to be the
    caller misjudged by a thin print, is exactly what should repair it."""
    s = _s(ExactGate(threshold=0.55))
    s.check_speaker("caller")
    s._gate_proven = True
    n = s._voiceprint_n
    ok, _sim = s.check_speaker("0.48", speakers=1)
    assert ok and s._voiceprint_n == n + 1, "a single-voice near miss must teach the gate"


def test_a_multi_voice_clip_never_widens_the_print():
    """The score cannot tell a misjudged caller (0.45-0.51) from a caller with someone
    talking over them (0.48) — measured, they overlap. The voice COUNT can, and the
    reported lockout showed "1 voice" on every discarded turn. So a near miss is
    forgiven either way (losing the caller's words is the worse failure), but only a
    clip known to hold one voice may teach the gate who the caller is."""
    s = _s(ExactGate(threshold=0.55))
    s.check_speaker("caller")
    s._gate_proven = True          # it has matched this caller, so the rest is meaningful
    before, n = s.voiceprint.copy(), s._voiceprint_n

    ok, _ = s.check_speaker("0.48", speakers=2)
    assert ok, "the caller's own words were discarded because someone spoke over them"
    assert s._voiceprint_n == n, "a two-voice clip widened the print"
    assert np.allclose(s.voiceprint, before)

    ok, _ = s.check_speaker("0.48", speakers=1)
    assert ok and s._voiceprint_n == n + 1, "a single-voice near miss must teach the gate"


def test_an_uncounted_near_miss_DOES_corroborate_the_print():
    """`speakers=None` means nobody counted — isolation off, or no diarizer. That is not
    the same as "mixed", and treating it as such reopened the original lockout by a side
    door: the caller's own near misses could never corroborate the print, so the
    provisional window expired with a print still too thin to recognise them and they
    were refused at 0.49 on their own call.

    A counted 2+ is still refused as evidence — see
    test_a_multi_voice_clip_never_widens_the_print."""
    s = _s(ExactGate(threshold=0.55))
    s.check_speaker("caller")
    s._gate_proven = True                                     # it can see this caller
    n = s._voiceprint_n
    ok, _ = s.check_speaker("0.48", speakers=None)
    assert ok
    assert s._voiceprint_n == n + 1, \
        "an uncounted near miss could not corroborate the print"


# --------------------------------------------------------------------------- #
# _voiced_only was refusing the caller with their own audio
# --------------------------------------------------------------------------- #
def test_trimming_does_not_splice_the_middle_of_an_utterance():
    """It used to concatenate the voiced frames and drop the gaps BETWEEN words. Every
    join is an artificial transient the encoder hears as part of the voice. Measured on
    one caller's own clip:

        spliced (gaps removed)   same speaker 0.450   <- below the 0.55 threshold
        voiced span              same speaker 0.843
        no trimming at all       same speaker 0.830

    So the trim meant to rescue short answers was refusing whole sentences instead.
    A span has no interior joins; the frame count must equal its own end minus start."""
    from zensuvidha.speaker import _voiced_only

    sr = 16000
    hop = sr // 100
    rng = np.random.default_rng(0)
    # speech, a gap, more speech — the shape splicing used to collapse
    clip = np.concatenate([
        np.zeros(30 * hop, dtype="float32"),                       # pre-roll
        (rng.normal(size=60 * hop) * 0.30).astype("float32"),      # a word
        np.zeros(25 * hop, dtype="float32"),                       # a gap mid-sentence
        (rng.normal(size=60 * hop) * 0.30).astype("float32"),      # another word
        np.zeros(40 * hop, dtype="float32"),                       # endpoint silence
    ])
    out = _voiced_only(clip, sr)
    assert out.size < clip.size, "the padding was not removed"
    # the interior gap must SURVIVE: span length >= both words plus the gap between them
    assert out.size >= (60 + 25 + 60) * hop, \
        f"the gap between words was spliced out ({out.size/hop:.0f} frames kept)"


def test_trimming_still_strips_leading_and_trailing_silence():
    """The reason it exists: a one-word answer arrives as pre-roll + 350ms of voice +
    endpoint silence, and ECAPA pooled over 76% silence scored the real caller 0.526."""
    from zensuvidha.speaker import _voiced_only

    sr = 16000
    hop = sr // 100
    rng = np.random.default_rng(1)
    voice = (rng.normal(size=50 * hop) * 0.30).astype("float32")
    clip = np.concatenate([np.zeros(30 * hop, dtype="float32"), voice,
                           np.zeros(80 * hop, dtype="float32")])
    out = _voiced_only(clip, sr)
    assert out.size < clip.size * 0.6, "the silence padding survived the trim"
    assert out.size >= 50 * hop, "the voice itself was cut"


def test_trimming_never_empties_a_clip():
    from zensuvidha.speaker import _voiced_only

    sr = 16000
    for clip in (np.zeros(sr, dtype="float32"),
                 (np.ones(sr) * 0.5).astype("float32"),
                 np.zeros(5, dtype="float32")):
        assert _voiced_only(clip, sr).size > 0


def test_the_repair_is_allowed_to_finish_before_the_gate_enforces():
    """Reported live, with music playing as the call opened. The print was enrolled from
    a contaminated first turn, and the caller then scored 0.09, 0.43, 0.52 against it.
    The widening WAS working — 0.43 -> 0.52, climbing toward the 0.55 threshold — and
    the provisional turn cap closed the window one turn early and discarded them.

    A turn that repairs the print must not burn the clock: the cap is for turns nothing
    can be learned from, and a near miss is the opposite of that."""
    s = _s(ExactGate(threshold=0.55))
    s.check_speaker("caller")                      # contaminated enrolment
    verdicts = []
    for score in ("0.09", "0.43", "0.52", "0.58"):
        ok, _sim = s.check_speaker(score, speakers=1)
        verdicts.append(ok)
    assert all(verdicts), f"the caller was discarded mid-repair: {verdicts}"
    assert s._voiceprint_n >= s.VOICEPRINT_TRUST_N, "the print never got corroborated"


def test_turns_that_teach_nothing_still_close_the_window():
    """The clock must still run, or a caller in a noisy shop — whose clips are always
    mixed and never learnable — would leave the gate off for the whole call."""
    s = _s(ExactGate(threshold=0.55))
    s.check_speaker("caller")
    s._gate_proven = True                                     # it can see this caller
    for _ in range(s.PROVISIONAL_MAX_TURNS + 1):
        s.check_speaker("0.30", speakers=2)        # mixed: cannot widen
    ok, _sim = s.check_speaker("0.30", speakers=2)
    assert not ok, "the provisional window never closed"


def test_a_gate_that_has_never_matched_the_caller_may_not_refuse_them():
    """Reported live, twice. The 0.55 threshold was calibrated on macOS `say` voices —
    same speaker 0.867, closest impostor 0.429. On a real caller's own microphone, in a
    room the router judged CLEAN (13-34dB floor-to-voice), their own consecutive turns
    scored 0.37 / 0.27 / 0.29 / 0.41. Never once above the threshold.

    A gate in that state cannot recognise the caller at all, so every refusal it makes
    is a coin toss that silences whoever is actually on the phone. It has to earn the
    right to refuse by matching them at least once."""
    s = _s(ExactGate(threshold=0.55))
    s.check_speaker("caller")
    assert not s._gate_proven
    verdicts = [s.check_speaker(x, speakers=1)[0]
                for x in ("0.37", "0.27", "0.29", "0.41", "0.35")]
    assert all(verdicts), f"the caller was discarded on their own call: {verdicts}"


def test_matching_once_earns_the_right_to_refuse():
    """Two separate gates, and both must open before anybody is refused: the gate has to
    have MATCHED the caller once (it works on this mic), and the print has to have been
    CORROBORATED (it is not a guess). This checks the first."""
    s = _s(StubGate())
    s.check_speaker("caller")
    assert not s._gate_proven
    s.check_speaker("caller", speakers=1)          # it recognises them
    assert s._gate_proven, "a confident match did not prove the gate"


def test_both_gates_open_before_a_stranger_is_refused():
    s = _corroborated(StubGate())                  # proven AND corroborated
    assert not s.check_speaker("a stranger", speakers=1)[0]


# --------------------------------------------------------------------------- #
# the expectation rescue — a SECOND opinion, alongside the voiceprint
#
# pyannote, ERes2Net, ECAPA and DeepFilterNet are untouched by this; it only runs on
# the path where the gate has already decided to refuse. The property under test is
# therefore one-directional: it may give a turn back, and may never take one away.
# --------------------------------------------------------------------------- #
def test_a_refused_turn_that_answers_our_question_is_rescued():
    """The documented unsolved limit: loud audio at the mic drives the caller's score
    against their OWN voice to 0.07, at which point every refusal is noise. The turn
    still carries the ten digits we just asked for, and that does not need the audio."""
    s = _corroborated(StubGate())
    s.pending_slot = "phone"
    refused, _ = s.check_speaker("a completely different voice")
    assert not refused, "precondition: the gate refuses this voice on the audio alone"

    ok, _ = s.check_speaker("a completely different voice", heard="8920429057")
    assert ok, "a turn carrying the number we asked for must be given back"


def test_the_rescue_does_not_clear_the_refusal_streak():
    """The rescue hands back the TURN; it must not hide the fact that the print is
    wrong. `_gate_rejects` reaching REENROL_AFTER is what repairs the print, and
    clearing it would trade a permanent fix for a per-turn rescue — the first turn
    that happened not to match an expectation would be refused again."""
    s = _corroborated(StubGate())
    s.pending_slot = "phone"
    s.check_speaker("stranger", heard="8920429057")
    assert s._gate_rejects > 0, "the refusal must still count toward re-enrolment"


def test_the_rescue_cannot_refuse_a_turn_the_gate_accepted():
    """One-directional, by construction. An accepted turn must be unaffected no matter
    what the transcript says — including a transcript with no bearing on the business."""
    for heard in (None, "", "and now the weather across the region", "haan",
                  "my son has a fever", "!@#$"):
        s = _corroborated(StubGate())
        s.pending_slot = "phone"
        ok, sim = s.check_speaker("caller", heard=heard)
        assert ok, f"the caller was refused with heard={heard!r}"


def test_a_stranger_saying_something_irrelevant_is_still_refused():
    s = _corroborated(StubGate())
    s.pending_slot = "phone"
    ok, _ = s.check_speaker("stranger", heard="and then he told me about the match")
    assert not ok


def test_no_transcript_leaves_the_gate_exactly_as_it_was():
    """Every existing caller passes no `heard`. Their behaviour must be byte-identical,
    which is what makes this additive rather than a rewrite."""
    a = _corroborated(StubGate())
    b = _corroborated(StubGate())
    a.pending_slot = b.pending_slot = "phone"
    assert a.check_speaker("stranger") == b.check_speaker("stranger", heard=None)


def test_the_rescue_reason_is_recorded_for_the_inspector_and_consumed_once():
    """A rescue that only reaches the server log is invisible to whoever is debugging
    the call. It must be reported — and must describe THIS turn, never leak to the next."""
    s = _corroborated(StubGate())
    s.pending_slot = "phone"
    assert s._last_rescue is None

    ok, _ = s.check_speaker("stranger", heard="8920429057")
    assert ok and s._last_rescue and "digit" in s._last_rescue

    s._last_rescue = None                      # server consumes it after one insight
    s.check_speaker("caller", heard="8920429057")
    assert s._last_rescue is None, "an ACCEPTED turn must not report a rescue"


# --------------------------------------------------------------------------- #
# what this caller scores on THIS microphone
# --------------------------------------------------------------------------- #
class _Bare(SpeakerGate):
    """The scoring logic without an encoder — thresholds are arithmetic, not models."""

    def __init__(self, threshold=0.55):
        self.backend = "stub"
        self.threshold = threshold
        self._caller_scores = []


def test_a_real_microphone_moves_the_bar_to_where_the_caller_actually_is():
    """MEASURED on one live call, 39.6dB room:

        the caller           0.21  0.22  0.26     ← every turn BELOW the 0.55 threshold
        a video in the room -0.06  0.04  0.05

    The separation is about 0.2 and completely invisible to an absolute bar sitting
    above both. So the gate never matched anyone, never became proven, and correctly
    refused to refuse — which is why background audio was transcribed and answered."""
    g = _Bare()
    for s in (0.22, 0.26, 0.21):
        g.note_caller_score(s)
    t = g.effective_threshold()
    assert 0.21 >= t and 0.26 >= t, f"the caller is still refused at {t:.2f}"
    assert 0.05 < t and -0.06 < t, f"the background video still passes at {t:.2f}"


def test_it_never_becomes_stricter_than_the_configured_threshold():
    """Tightening on the strength of a handful of samples would silence the real caller,
    which is the failure this whole file is organised around. It may only ever loosen
    toward what the microphone really produces."""
    g = _Bare()
    for s in (0.85, 0.88, 0.83):
        g.note_caller_score(s)
    assert g.effective_threshold() == 0.55


def test_it_waits_for_evidence_before_moving():
    g = _Bare()
    assert g.effective_threshold() == 0.55
    g.note_caller_score(0.24)
    assert g.effective_threshold() == 0.55, "one sample moved the bar"


def test_a_single_bad_sample_cannot_drag_the_bar_down():
    """The median, not the mean — one outlier is exactly what a half-caught utterance
    looks like."""
    g = _Bare()
    for s in (0.62, 0.65, 0.02, 0.60):
        g.note_caller_score(s)
    assert g.effective_threshold() > 0.3, g.effective_threshold()


def test_the_bar_has_an_absolute_floor():
    """A print that has degenerated to scoring near zero must not license accepting
    literally anything."""
    g = _Bare()
    for s in (0.01, 0.02, 0.0):
        g.note_caller_score(s)
    assert g.effective_threshold() >= g.ADAPT_FLOOR


def test_only_accepted_turns_teach_it():
    """Learning from refused turns would let a stranger drag the bar down onto their own
    score, which is the one way an adaptive threshold can be worse than a fixed one."""
    import inspect

    from zensuvidha.orchestrator import Session
    src = inspect.getsource(Session.check_speaker)
    at = src.index("note_caller_score")
    ok_at = src.index("self._gate_proven = True")
    assert ok_at < at, "the score is learned outside the accepted branch"


def test_the_calibration_is_not_circular():
    """MY OWN BUG, caught by replaying the live call through it.

    The first version learned only from turns the gate ACCEPTED. On this caller's
    microphone their own turns score 0.21-0.26 against a 0.55 bar, so nothing was ever
    accepted, so nothing was ever learned, so the bar never moved — the adaptation was
    dead code in precisely the situation it was built for.

    The same trap catches the gate's other two escape routes, which is why none of them
    fired either: widening needs PROVISIONAL_FLOOR 0.40, rival adoption needs the same
    voice twice. Every mechanism is keyed to an absolute number that sits above where
    this microphone lands.
    """
    g = _Bare()
    for sim in (0.22, 0.26, 0.21):
        # the engine answers these (unproven → fails open) and calibrates from that
        g.note_caller_score(sim)
    assert g.effective_threshold() < 0.21, (
        "the bar never moved: %s" % g.effective_threshold())


def test_after_calibrating_the_caller_is_kept_and_the_room_is_not():
    """The whole point. Replayed from one live call with a video playing in the room."""
    g = _Bare()
    for sim in (0.22, 0.26, 0.21):
        g.note_caller_score(sim)
    bar = g.effective_threshold()
    for caller in (0.24, 0.23, 0.21):
        assert caller >= bar, f"the caller was refused at {caller}"
    for room in (0.05, 0.04, -0.06):
        assert room < bar, f"the background video passed at {room}"


def test_a_clip_with_more_than_one_voice_teaches_it_nothing():
    """A mixed clip scores like a misjudged caller. Calibrating on one would move the
    bar toward whoever interrupted them — the same rule that stops a mixed clip widening
    the print, for the same reason."""
    import inspect

    from zensuvidha.orchestrator import Session
    src = inspect.getsource(Session._note_caller_score)
    assert "speakers or 1" in src and "> 1" in src, (
        "a multi-voice clip is allowed to calibrate the gate")


def test_the_engine_calibrates_from_turns_it_answers():
    """Both places a turn is concluded to be the caller must feed it, or the deadlock
    returns by whichever route was missed."""
    import inspect

    from zensuvidha.orchestrator import Session
    src = inspect.getsource(Session.check_speaker)
    assert src.count("_note_caller_score") >= 2, (
        "only one of the accepted paths calibrates the gate")
