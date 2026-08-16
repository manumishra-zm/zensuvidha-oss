# ZenSuvidha OSS — explained, part by part

This is the walkthrough: what every component is for, **why that one and not the
alternatives**, and — in [part 5](#5-how-the-audio-models-actually-work) — how the audio
models actually work at the algorithm level, since "it filters the voice" is where most
explanations stop.

Diagrams here are ASCII so they render in a terminal, in a diff, and on GitHub without
anything installed. The same system is drawn as
[Mermaid](DIAGRAMS.md), as a [laid-out technical sheet](../ARCHITECTURE.html), and as two
[editable drawings](../ARCHITECTURE.excalidraw). The measurements behind every number are
in [ARCHITECTURE.md](../ARCHITECTURE.md).

**Contents**

1. [The problem, and the constraint that shapes everything](#1-the-problem)
2. [The whole system](#2-the-whole-system)
3. [One turn, end to end](#3-one-turn-end-to-end)
4. [Each part: role, choice, rejected alternatives](#4-each-part)
5. [How the audio models actually work](#5-how-the-audio-models-actually-work)
6. [Why a call can never wedge](#6-why-a-call-can-never-wedge)
7. [Operating envelope and latency](#7-operating-envelope-and-latency)
8. [What makes it different](#8-what-makes-it-different)
9. [What it cannot do](#9-what-it-cannot-do)

---

## 1. The problem

An Indian small business — a clinic, a salon, a restaurant — misses calls. Existing voice
AI needs a cloud API, a per-minute bill, a telephony account, and handles Hindi or Telugu
badly. This runs the whole thing **on one laptop**: no API key, no GPU, no internet, no
per-call cost, and the caller's data never leaves the machine.

That constraint is what makes the design interesting. When you cannot throw a GPU or a
frontier model at a problem, you have to actually solve it — and you have to measure,
because there is no headroom to waste on a component that turns out not to help.

---

## 2. The whole system

```
┌─────────────────────────────────────────────────────────────────────────┐
│  BROWSER  ·  web/index.html                                             │
│                                                                          │
│   🎙 mic ──▶ AudioWorklet ──▶ Silero VAD v5 ──▶ endpointer ──▶ latch     │
│   AGC OFF    biquad 300-3400   "is this speech?"  800/1200/    guard     │
│              (level gate only)  2.3 MB ONNX       →2000 ms    7s = drop  │
│                                                                          │
│   audio inspector ◀── spectrogram + one row per turn                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │  ONE WebSocket per call
                    WAV frames up │ JSON + audio down
┌────────────────────────────────▼────────────────────────────────────────┐
│  SERVER  ·  FastAPI + uvicorn  ·  one Python process                    │
│                                                                          │
│   pipeline.prepare ──▶ STT ──▶ speaker gate ──▶ LLM ──▶ guard ──▶ TTS   │
│   isolate → denoise                                                      │
│        │                                          ▲                      │
│        │                            Industry Pack ┘ (YAML + RAG-lite)   │
│        │                                          │                      │
│        │                                     SQLite (bookings)          │
└────────┼─────────────────────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────────────────────┐
│  MODELS  ·  all local, all on disk, ~700 MB + 2.5 GB LLM                │
│                                                                          │
│   faster-whisper   pyannote-seg   ERes2Net   ECAPA-TDNN   Qwen3  Kokoro │
│   460 MB · MIT     7 MB · MIT     38 MB      80 MB        2.5 GB  ~80MB │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. One turn, end to end

Everything indented is an **exception** — it fires only when its condition holds. The
gates exist so the ordinary turn stays cheap and the expensive stages are reached rarely
and on purpose.

```
 caller speaks
     │
 ┌───▼─────────────────────────────────────────────── BROWSER ───────────┐
 │ 1. capture            AEC on, noise suppression on, AGC OFF           │
 │ 2. band split         biquad 300–3400 Hz → level gate only            │
 │                       raw PCM still goes to the server untouched      │
 │ 3. Silero VAD v5      "is this speech?" — never "whose?"              │
 │ 4. endpoint           short answer 800ms · long utterance 1200ms      │
 │                       learns up to 2000ms from turns it cut off       │
 │                       └ words say "unfinished"? +900ms                │
 │ 5. latch guard        7s recorded, never a 260ms pause → DISCARD      │
 │                       people breathe; a loudspeaker does not          │
 └───┬───────────────────────────────────────────────────────────────────┘
     │  ═══════════════ WebSocket ═══════════════
 ┌───▼─────────────────────────────────────────────── SERVER ────────────┐
 │ 6. ISOLATE on RAW     pyannote segmentation + ERes2Net clustering     │
 │      ├ no voiceprint (turn 1)   → skip entirely           0 ms       │
 │      ├ 1 voice, < 4s            → done                   ~49 ms       │
 │      ├ 1 voice, ≥ 4s            → thirds gate, 3× ECAPA ~124 ms       │
 │      ├ ambiguous                → window rescan ≤8      ~340 ms       │
 │      └ 2+ voices    → score each cluster, trim to the caller,         │
 │                       re-check the survivor for a merge               │
 │                                                                       │
 │ 7. denoise            DeepFilterNet — OFF by default    226–475 ms    │
 │ 8. silence strip      Silero v6 (ships inside faster-whisper)  free   │
 │ 9. recognise          faster-whisper                     ~1070 ms ◀── │
 │                       3 rejections: no_speech>0.85, logprob<−1.6,     │
 │                       degeneracy (TRIMS the runaway, keeps the words) │
 │10. speaker gate       ECAPA-TDNN                           ~34 ms     │
 │      ├ no print         → enrol, accept                               │
 │      ├ never matched    → ACCEPT (hasn't earned the right to refuse)  │
 │      ├ not corroborated → forgive near misses, remember rivals        │
 │      ├ proven + corroborated → now it may refuse…                     │
 │      └ …unless the TURN answers what we just asked → rescue (§4.3b)   │
 │11. LLM                Qwen3-4B, forced JSON, streamed   400–600 ms    │
 │12. GUARD per sentence kind→echo→repetition→degeneracy→               │
 │                       ungrounded numbers→language                     │
 │13. TTS                routed by SCRIPT, not preference     385 ms     │
 └───┬───────────────────────────────────────────────────────────────────┘
     ▼
 browser plays sentence 1 while sentence 2 is still being written
```

---

## 4. Each part

### 4.1 Voice activity — Silero v5 (browser) + v6 (server)

**Role.** Decide when a turn starts and stops. No push-to-talk.

```
   browser: Silero v5  ──▶  "should I OPEN a turn?"
   server:  Silero v6  ──▶  "what should Whisper SEE?"
                            (ships free inside faster-whisper)
```

**Why two copies.** They answer different questions, and the second is free — faster-whisper
bundles `silero_vad_v6.onnx` and runs it as `vad_filter`. The browser copy decides whether
to start recording; the server copy strips silence so Whisper does not hallucinate from it.

**Why Silero.** MIT, 2.3 MB, runs as ONNX-WASM in the browser so there is no round-trip per
frame. Loud white noise scores `0.023`, which closes the noise→hallucination path at the
source.

**The trap.** v5 requires the *previous* frame's last 64 samples prepended (input becomes
`[1, 576]`). Without it the model returns ~`0.11` on loud clear speech instead of ~`0.99` —
and looks perfectly healthy while doing it. v6 has a different interface again
(`h`/`c` state, no `sr`, output named `speech_probs`), so the browser detects the version by
output name rather than assuming.

---

### 4.2 Voice isolation — pyannote-segmentation-3.0 + 3D-Speaker ERes2Net

**Role.** A colleague or a TV talking in a *gap* used to drag whole-clip similarity from
`0.867` down to `0.34` — so the caller's own turn was thrown away, **and** the stranger's
words landed in the transcript. This removes the other person before Whisper sees anything.

```
  raw audio ──▶ segmentation ──▶ clustering ──▶ score each cluster
                (where voices     (group into    against the caller's
                 start & stop)     speakers)      voiceprint
                                                       │
                            keep only the caller ◀─────┘
                                     │
                            ONE cluster kept?
                                     │ yes
                            re-check IT for a merge
                            (a "cluster" can be two people)
```

**Why not source separation** (SepFormer, MossFormer2). Measured first, then not built.
True *overlap* — both talking at once — turns out to be survivable: similarity 0.594–0.833,
the gate accepts, Whisper still transcribes the caller. The real failure is **sequential**,
someone speaking in a pause. Separation solves a problem this system does not have.

**Why not ClearerVoice-Studio.** Its only released target-speaker-extraction checkpoint is
`AV_MossFormer2_TSE_16K` — **audio-visual**. It needs a face video. A phone line has none.

**Why sherpa-onnx rather than PyTorch pyannote.** The ONNX form is ungated: no HuggingFace
token, no torch, no GPU, 7 MB.

**The hard-won part.** pyannote is **order-sensitive**. Same two voices, same gap, only the
order reversed:

```
  caller then stranger    → 2 segments, 2 speakers    ✓ trimmed correctly
  stranger then caller    → ONE segment, whole clip   ✗ total leak
  stranger then caller    → 2 segments, BOTH labelled ✗ total leak
    with a 1.0s gap          speaker 1
```

So the clusterer's *labels* cannot be trusted on their own. The fix is to use the diarizer
for **boundaries** and ECAPA — the model actually tuned here — for **identity**.

And the decisive detail: one person speaking two sentences **dips** at their own pause;
a stranger's turn is a **run**.

```
  one speaker, two sentences   0.77 0.78 0.74 0.71 [0.50] 0.66 0.67 0.73
  caller then a stranger       0.77 0.78 0.74 0.71 [0.45  0.52  0.35  0.23]
                                                    └── a RUN, not a dip ──┘
```

Splitting on the dip deleted 0.8 s of the caller's own words, so a re-split needs a run of
at least two low windows, judged both relative to the best window and in absolute terms.

---

### 4.3 Speaker gate — ECAPA-TDNN (SpeechBrain)

**Role.** Decide whether the voice on this turn is the caller. Answer the caller, ignore
the television.

**Why ECAPA.** Apache-2.0, ungated (unlike Indic-Parler), 80 MB, 192-dim embedding, cosine
similarity.

**Why it is the most interesting component.** It has caused more user-visible failures than
anything else here, and *every* repair moved it toward refusing **less**. One measurement
explains all of them:

| tested on | same speaker | closest impostor | verdict at 0.55 |
|---|---|---|---|
| macOS `say` voices | 0.867 | 0.429 | separates them |
| **a real microphone** | **0.27 – 0.41** | — | **refuses the caller** |

The threshold was calibrated on synthetic audio. In a room the noise router measured
*clean*, a caller's own consecutive turns scored 0.37 / 0.27 / 0.29 / 0.41 — never once
above threshold. The gate could not recognise them at all, so **every refusal it made was
noise**.

The fix is a principle, not a number — **a gate must earn the right to refuse**:

```
   NoPrint ──enrol──▶ Provisional ──matched once──▶ Proven
                          │                            │
                     mismatch?                    corroborated
                     ANSWER it,                    (≥3 utterances)
                     remember as RIVAL                  │
                          │                             ▼
                  same rival twice?                 Enforcing
                  → it IS the caller            (now strangers refused)
```

Two independent conditions must hold before anyone is refused: **proven** (it has matched
this caller at least once *on this call*) and **corroborated** (the print is more than one
un-corroborated sample). Until both hold, a mismatch is answered and the voice remembered
as a rival. *The voice having the conversation is the caller — a song does not follow up on
its own question.*

**The root cause of two total lockouts was our own code.** `_voiced_only` kept voiced frames
and *spliced out the gaps between words*. Every join is an artificial transient the encoder
hears as part of the voice:

```
   spliced (gaps removed)   same speaker 0.450  ← BELOW the 0.55 threshold
   voiced SPAN kept         same speaker 0.843
   no trimming at all       same speaker 0.830
```

The tell was that the score got **worse with more audio** (3.0 s → 0.507 against
2.0 s → 0.818). When a metric degrades as data increases, suspect the preprocessing, not
the model.

---

### 4.3b Expectation rescue — a second opinion, alongside the voiceprint

**Nothing above changes.** pyannote, ERes2Net, ECAPA and DeepFilterNet all do exactly
what they did; this adds one more signal, consulted only where the gate has already
decided to refuse.

**Role.** The acoustic path has one failure it cannot see from the inside: loud audio at
the microphone drives the caller's score against their **own** voice to `0.07`. At that
point the similarity number is not imprecise, it is uninformative — and every refusal
made on it is noise. This asks something the acoustics cannot:

```
   the gate refused on the audio
              │
              ▼
   does this turn ANSWER the question we just asked?
              │
     ┌────────┴────────┐
     │                 │
   yes                 no
     │                 │
  give the turn      nothing happens —
  back               the gate's verdict stands
```

Ten digits arriving right after *"what mobile number should we use?"* is the caller,
whatever ECAPA thinks of the recording — and that judgement never touches the voiceprint,
so it survives exactly the case that breaks it.

**Why it can only rescue.** Callers say things with no bearing on the business at all —
*"hello?"*, *"can you hear me?"*, *"haan"*, *"my son has a fever"*. A relevance threshold
that rejected those would repeat the mistake the 0.55 speaker threshold made, in a domain
where the caller has no way to try harder. So there is no path from this module to a
rejection, and the test suite pins that structurally rather than assuming it.

**What it cannot do.** Two containment properties, both pinned by tests rather than
asserted:

```
  it never reaches the audio       only import is `re`; only input is a transcript STT
                                   has already produced; runs AFTER every filtering
                                   decision → it cannot remove one sample of the caller

  a rescue never moves the print   the voiceprint is what isolation trims against, so
                                   moving it on TEXT evidence would spread into what
                                   LATER turns keep. It does not.
```

The one indirect path, stated plainly rather than hidden: rescued turns still count
toward the refusal streak, so a *run* of them can trigger the pre-existing re-enrolment
repair. That is intended — it is how a caller whose print was destroyed gets a correct
one back. It is not a new hole either: the repair has always needed the same voice
refused repeatedly, and it still refuses to learn from a clip known to hold more than one
voice, so a blend of two people cannot become somebody's identity.

**What counts, and how much.** Scores accumulate, so two independent medium signals can
rescue where no single weak one can:

```
   STRONG  1.00   the exact number we asked for · one of the pack's own proper nouns
   MEDIUM  0.50   a time expression on cue · two of the business's own terms
   WEAK    0.25   a short non-question when we asked for a name · one business term
                                                              rescue at >= 1.00
```

**The part that nearly made it useless.** `pending_slot` is deliberately left `None`
whenever more than one field is outstanding — filing an answer against the wrong slot is
worse than not filing it. Correct for collection, and it meant the rescue would almost
never have fired, because a booking spends most of its turns with several fields missing.
A separate `last_asked_slot`, derived from the line the caller actually *heard*, fixes
that without touching the stricter semantics collection depends on. Found by an
end-to-end probe; the unit tests had been setting the slot by hand and passed either way.

---

### 4.4 Noise reduction — DeepFilterNet, and why it ships **off**

```
  audio ──▶ measure floor-to-voice gap ──▶ auto_denoise enabled?
            (numpy percentiles, FREE)          │
                                        no ────┴──── yes
                                        │             │
                                     skip 0ms    gap < 10 dB?
                                        │         │        │
                                        └────┬────┘   DeepFilterNet
                                             │        226–475 ms
                                         Whisper
```

The measurement always runs, so the inspector can report the room either way. Only the
*reduction* is off.

**Why.** Swept across 6 interference types × 5 SNRs — 30 conditions:

```
   DeepFilter WON 1 cell · LOST 8 · tied 21        net −400% word recall
   white hiss  −50%      fan/AC  −67%      music+vocals  −100%
```

It is also the most expensive stage in the pipeline. On the recognition path it costs time
**and** accuracy — the same verdict `noisereduce` received (0.00 WER delta down to 0 dB SNR,
and it made the speaker gate *worse* in every condition tested).

**Why that happens.** Whisper trained on 680,000 hours of messy audio and learned its own
robustness. Denoisers optimise for *"sounds clean to a human"*, which strips exactly the
fine detail Whisper and ECAPA use. The general lesson: **do not put a denoiser in front of
a modern neural model without measuring.**

**And the deeper point.** Instrumental music does **not** hurt Whisper — measured, WER
identical (0.27) from clean through to music at equal loudness. What breaks it is a
background *voice*, and DeepFilterNet is a speech *enhancer*, so it protects the singer too.
**No denoiser can remove a voice. Only speaker isolation can.** Section 5 explains why that
is structural rather than a tuning problem.

**Installation note.** Always the precompiled Rust binary, never the pip wheel: the wheel
pins `numpy<2`, which silently downgraded numpy and broke SpeechBrain while pytest stayed
green, and its `df/io.py` imports a torchaudio backend removed in modern torchaudio.

---

### 4.5 Speech recognition — faster-whisper on CTranslate2

**Why.** C++ inference core, int8 on CPU. `small` on a laptop, `large-v3` on GPU.

**Why not Vosk** — weaker for Indic. **Why not Parakeet** — 25 European languages, no Telugu.

**Why `large-v3` matters.** `whisper-small` transcribes spoken **Telugu into Devanagari** —
the wrong script entirely.

**Three rejections before a transcript is trusted:** `no_speech > 0.85`,
`avg_logprob < −1.6`, and a degeneracy check that **trims** a runaway character run rather
than dropping the turn — condemning the whole turn once cost a caller their phone number.
The confidence gate is duration-weighted across segments rather than worst-segment, or one
brief pause would kill a long real sentence.

---

### 4.6 The language model — Qwen3-4B via Ollama

**Why Qwen3 and not Qwen2.5.** Qwen2.5 supports 29 languages and **Telugu is not among
them**. Qwen3 supports 119. Never trade the model family for size — a 14B Qwen2.5 is *worse*
at Telugu than a 4B Qwen3 on CPU.

**Why Ollama and not MLX.** Benchmarked: MLX `39.6 tok/s` against Ollama `38.7`. Ollama is
already 100% Metal on this machine. **Why not AirLLM** — it fits a 70B model into 4 GB by
loading one layer at a time. That is a *memory* tool; it would be slower, not faster.

**The 6,089-token prompt is deliberate:**

```
   ┌──────────────────────────────────────────┐
   │  6,089-token system prompt               │  BYTE-IDENTICAL every turn
   │  persona · services · policies · KB      │  → Ollama reuses its KV cache
   └──────────────────────────────────────────┘  → 640 ms steady-state turn
   ┌──────────────────────────────────────────┐
   │  conversation history (this call)        │  the only part that changes
   └──────────────────────────────────────────┘
```

It carries weight twice. The cache-stable prefix is what makes turns 3 onward take 640 ms,
**and** the full English knowledge base is what lets Hindi and Telugu callers get answers at
all, because English tags cannot retrieve against a native-script query. Trimming the prompt
to "optimise" it breaks both at once.

**The Indic ceiling is tokenizer inflation, not runtime.** Hindi 3.5×, Telugu 6.2×, Odia
7.1×. At 38.7 tok/s the same 40-token reply is 1.03 s in English, 3.62 s in Hindi and
**6.41 s in Telugu**. No inference library changes that, which is why MLX was benchmarked
and rejected rather than adopted on principle. A GPU is the answer.

Also: `num_ctx` must be **constant** across calls. Varying it per language makes Ollama
reallocate the KV cache and reload the model — 3 s becomes 16 s per reply.

---

### 4.7 The grounding guard — the component that makes it usable

**Role.** Prompt rules do not hold a 4B model. Asked for a service the clinic does not
offer, it invents one, with a plausible price.

```
  token stream
      │
      ▼
  ┌─ kind? ────────── unknown / out_of_scope ──▶ pre-written refusal (12 langs)
  ├─ echo? ────────── answering AS the caller ─▶ safe line
  ├─ repeats? ─────── same as last turn ───────▶ safe line
  ├─ degenerate? ──── a clause looping ────────▶ close stream, retry
  ├─ numbers? ─────── a price in NEITHER the ──▶ safe line
  │                   pack nor the caller's words
  └─ language? ────── wrong script ────────────▶ safe line
      │
      ▼  all clear
    SPEAK IT
```

It runs **on the stream** — a bad sentence is intercepted before synthesis, because on a
call you cannot un-say a price.

**Why this is the part that matters most.** Probed with 30 questions real callers ask, the
agent answered 28/30 — but **6 of those answers were fabricated**: SMS reminders, fitness
certificates, travel vaccinations, proxy report collection. None had ever been in the pack.
Counting non-shrug replies as success measures the wrong thing; always diff the claims
against the source of truth.

Worse, `_base.yaml` literally instructed the model to *"give price/time ranges when
unsure"* — **an instruction to invent**. A range is a made-up number with a second made-up
number beside it, and the caller hears it as the answer. Removed.

The fix that actually worked was having the fact: 39 → 51 knowledge entries (EN+HI+TE), then
30/30 grounded. **The trade-off worth knowing:** every fact added *widens* the set of
grounded numbers and weakens the guard. Knowledge and tight grounding pull against each
other.

---

### 4.8 Text to speech — routed by script, not preference

```
  reply text ──▶ dominant script?
                     ├── Latin        ──▶ Kokoro af_heart      385 ms
                     ├── Devanagari   ──▶ Kokoro hf_alpha      583 ms
                     └── Telugu, Tamil, Kannada, Bengali…
                              └──▶ system voice               1065 ms
                                     └── no voice? → silent + mute_reason
                                         (Malayalam, Gujarati, Punjabi,
                                          Odia, Urdu on macOS)
```

**Why Kokoro.** It runs in-process, so cost scales with the text. macOS `say` is
**~1064 ms of fixed process spawn** plus 0.4 ms/char — "Hi." costs 1065 ms and a 112-char
sentence costs 1105 ms, so 99% of a normal first sentence is overhead and splitting the
reply into chunks buys nothing.

**Why routing by script rather than by a config setting.** Fed Telugu, an English Kokoro
pipeline produced **2.1 MB and 6.5 seconds** of confident nonsense. It does not fail loudly.
So the script is checked *before* anything is synthesised, and a provider signals "not my
script" so only that case falls through to the fallback — a provider that merely *failed*
must not silently reroute the rest of the call.

---

### 4.9 Industry packs — the business as data

```
  packs/clinic.yaml     persona · services · prices · policies ·
  packs/salon.yaml      situation playbooks · 51 knowledge entries
  packs/restaurant.yaml (EN + HI + TE) · booking slots · aliases
  packs/laundry.yaml
  packs/gym.yaml        → adding an industry is a YAML file.
  packs/hotel.yaml         Zero code.
```

Storing pack knowledge **in the caller's language** (`a_hi`/`a_te` answers with `k_hi`/`k_te`
keywords) is the single biggest quality-and-latency win on a small model: it quotes instead
of translating. A Hindi fee question went from 27 s to 5 s. Only the top ~4 entries by
keyword overlap are injected — the whole knowledge base in Telugu overflows the context and
makes the model refuse answerable questions.

---

## 5. How the audio models actually work

"It filters the voice" is where most explanations stop. Here is the mechanism, because the
difference between these two families is exactly what the measurements in 4.2 and 4.4
reflect.

### 5.1 The one distinction that explains everything

```
  DeepFilterNet   MASKS   — decides "speech or noise?" per time-frequency bin,
                            then multiplies the spectrum by a gain and rebuilds audio.
                            Output: NEW audio, same length.

  pyannote        CUTS    — decides "who is speaking when?" per ~16 ms frame,
                            then we DELETE the samples belonging to other people.
                            Output: SHORTER audio, untouched samples.
```

```
   DeepFilterNet (mask)          pyannote + clustering (cut)
   ┌─────────────────┐           ┌─────────────────────────┐
   │▓▓▒▒░░▓▓▒▒░░▓▓▒▒│  in       │ AAAA BBBB AAAA BBBB     │  in
   └────────┬────────┘           └───────────┬─────────────┘
            │ × gain mask                    │ keep A, drop B
   ┌────────▼────────┐           ┌───────────▼─────────────┐
   │████░░████░░████░│  out      │ AAAA      AAAA          │  out (shorter)
   └─────────────────┘           └─────────────────────────┘
   every sample still there,      B's samples are GONE
   just re-weighted
```

**A mask cannot remove a second human**, because a second human *is* speech and the mask's
entire job is to keep speech. A cut can, because it never asks what the sound is — only who
it belongs to.

That is the whole reason auto-denoise ships off and isolation ships on.

### 5.2 pyannote-segmentation-3.0

**What it answers:** *between second 0.0 and second 10.0, which frames contain speaker 1,
speaker 2, speaker 3, or overlap?* Not *"whose voice is this"* — it has no idea who anyone
is. It only knows *"these frames are a different person from those frames."*

**Representation: the raw waveform, not a hand-made spectrogram.**

```
   16 kHz mono waveform (10-second window)
        │
        ▼
   ┌──────────────────────────────────────────┐
   │  learned convolutional frontend          │  the "spectrogram" is
   │  (band-pass-like filters, LEARNED        │  learned from data rather
   │   from data rather than a fixed FFT)     │  than a fixed transform
   └──────────────────┬───────────────────────┘
                      ▼
   ┌──────────────────────────────────────────┐
   │  recurrent / temporal layers             │  builds context — you cannot
   │  (sequence modelling over frames)        │  tell speakers apart from
   └──────────────────┬───────────────────────┘  one 16 ms frame alone
                      ▼
   per-frame classification, ~16 ms resolution
```

**The algorithm: powerset multi-label classification.** This is the distinctive part of
version 3.0. Instead of predicting one independent binary "is speaker N active" per speaker,
it predicts a single class from the *powerset* of up to three speakers:

```
   class 0 : silence
   class 1 : speaker 1                ┐
   class 2 : speaker 2                │ single-speaker
   class 3 : speaker 3                ┘
   class 4 : speakers 1 AND 2         ┐
   class 5 : speakers 1 AND 3         │ OVERLAP is its own class
   class 6 : speakers 2 AND 3         ┘
```

Modelling overlap as an explicit class — rather than letting two independent binary heads
both fire — is why it handles two people talking at once instead of collapsing them.

**What comes out is timestamps and labels, not audio:**

```
   0.00 ──── 2.31   speaker_1
   2.31 ──── 2.68   (silence)
   2.68 ──── 4.10   speaker_2
```

Those labels are **local to one window** and arbitrary: "speaker_1" in one window has no
relationship to "speaker_1" in the next. That is what the clustering stage exists to fix,
and the order-sensitivity in 4.2 is a consequence of it.

> Confidence note: the powerset output and the learned-frontend structure are the parts
> this design depends on and they are well established. The exact frontend block in 3.0
> versus earlier pyannote releases is not something this repo verified, and nothing here
> relies on it.

### 5.3 ERes2Net and ECAPA-TDNN — the identity models

These **do** use a spectrogram: specifically a **mel filterbank**, an FFT magnitude
spectrogram warped onto a perceptual frequency scale.

```
   audio ──▶ STFT ──▶ |magnitude| ──▶ 80 mel filters ──▶ log
                                            │
                                            ▼
                                   80 × T feature matrix
                                            │
                                            ▼
                          ┌─────────────────────────────────┐
                          │  ECAPA-TDNN                     │
                          │   · Res2Net blocks (multi-scale │
                          │     receptive fields in one     │
                          │     layer)                      │
                          │   · Squeeze-Excitation (channel │
                          │     attention)                  │
                          │   · multi-layer aggregation     │
                          │   · ATTENTIVE STAT POOLING ◀────┼── the important bit
                          └─────────────────┬───────────────┘
                                            ▼
                              192-dim embedding (ONE vector
                              for the WHOLE clip)
```

**Attentive statistics pooling** collapses a variable-length clip into one fixed vector by
taking a *weighted* mean and standard deviation across time — the network learns which
frames carry identity and which do not. Identity is then cosine similarity between two
vectors:

```
   similarity = (v₁ · v₂) / (|v₁| |v₂|)      → 1.0 same, 0.0 unrelated
```

**Why this pooling explains our worst bug.** Pooling averages over *whatever frames you give
it*. Two consequences, both measured:

```
   silence padding:  300ms pre-roll + 350ms voice + 800ms endpoint silence
                     → 76% of the pooled frames are silence
                     → real caller scored 0.526, REJECTED

   spliced gaps:     removing inter-word gaps creates artificial transients
                     at every join, which pooling averages IN as "voice"
                     → same speaker 0.450 (spliced) vs 0.843 (span kept)
```

**Why two identity models.** ERes2Net is the same family — mel filterbank in, embedding out
— with multi-scale feature fusion. It does the *clustering*; ECAPA does the *arbitration*.
The clusterer's labels are an opinion, so they get verified:

```
   pyannote:   "segment A, segment B, segment C"     (boundaries — trusted)
   ERes2Net:   "A and C are the same person"         (opinion — verified)
   ECAPA:      "…and that person is the caller"      (decision — tuned here)
```

### 5.4 DeepFilterNet

This is the one that genuinely lives in the spectrogram, and its algorithm is unusual.

**The problem it solves.** A naive denoiser predicts a real-valued gain per time-frequency
bin and multiplies. That fixes the **magnitude** but leaves the **phase** wrong, which is
why classic spectral gating sounds watery and smears speech detail — and why it strips the
fine structure ECAPA needs.

**The two-stage algorithm:**

```
   48 kHz audio
        │
        ▼  STFT
   complex spectrum  X(f, t)          f = frequency bin, t = frame
        │
   ╔════╪═══════════════════════════════════════════════════════╗
   ║ STAGE 1 — ERB gains  (coarse, whole band)                   ║
   ║                                                              ║
   ║   group the FFT bins into ~32 ERB bands                     ║
   ║   (ERB = Equivalent Rectangular Bandwidth — narrow bands     ║
   ║    at low frequency, wide at high, matching the ear)         ║
   ║                                                              ║
   ║   predict ONE real gain per band  →  shapes the ENVELOPE     ║
   ║                                                              ║
   ║   ▁▂▄█▆▃▂▁▁▂▃▁  ──gains──▶  ▁▁▂█▆▂▁▁▁▁▁▁                    ║
   ╚════╪═══════════════════════════════════════════════════════╝
        │
   ╔════╪═══════════════════════════════════════════════════════╗
   ║ STAGE 2 — "Deep Filtering"  (fine, low frequencies only)     ║
   ║                                                              ║
   ║   for each low-frequency bin, predict a set of COMPLEX       ║
   ║   coefficients and apply them across a few NEIGHBOURING      ║
   ║   TIME FRAMES:                                               ║
   ║                                                              ║
   ║        Y(f,t) = Σ  c_i(f,t) · X(f, t−i)                     ║
   ║                 i                                            ║
   ║                                                              ║
   ║   i.e. a short per-bin FIR filter along time with complex    ║
   ║   taps — so it can correct PHASE, not just magnitude, and    ║
   ║   reconstruct the periodic harmonic structure of voiced      ║
   ║   speech that a real-valued mask destroys.                   ║
   ╚════╪═══════════════════════════════════════════════════════╝
        │
        ▼  iSTFT
   cleaned 48 kHz audio
```

**Why split it that way.** The ear needs fine resolution at low frequencies, where pitch and
harmonics live, and tolerates coarse resolution higher up. Stage 1 handles the broad noise
floor cheaply across the whole band; stage 2 spends the expensive complex-valued modelling
only where it pays.

**Why it cost us accuracy anyway.** Its training objective is *"sound clean to a human."*
Both stages actively remove content the downstream neural models were using:

| what DeepFilter removes | what it costs us |
|---|---|
| low-energy spectral detail | ECAPA identity 0.675 → 0.596 |
| noise entangled with speech | Whisper WER 0.00 → 0.10 |
| **any non-speech** | **nothing, if the interferer is a VOICE** |

That last row is the −100% cell on music-with-vocals: the vocal is speech, so it is
*preserved*, which is exactly wrong for our purpose.

There is one position where it is transformative, and it is not the recognition path: fed to
the **VAD decision**, separation went from 1.99 raw to 110.6 — 55×. That was the original
"denoise the decision, not the recognition" finding. It is not shipped only because the VAD
runs in the browser worklet and DeepFilterNet has no WASM build; streaming every 32 ms frame
to the server for a yes/no would defeat barge-in.

### 5.5 The one classic spectral filter that survives

```
   browser AudioWorklet:
                       ┌────────────────────────┐
   mic ──┬────────────▶│ 2-pole biquad IIR      │──▶ rmsBand ──▶ level gate
         │             │ band-pass 300–3400 Hz  │              (2.79× rejection)
         │             └────────────────────────┘
         │
         └──────────── raw PCM, untouched ─────────────────────▶ server → Whisper
```

Two poles, no model, ~0 latency, zero dependencies. It **only** feeds the energy gate that
decides *"is anything happening?"*. Whisper never sees band-limited audio. Same principle as
everything else here: **filter the decision, never the recognition.**

Measured on the real shipping filter: fan/AC 4.61×, high hiss 2.80×, traffic 2.09×,
broadband hiss 1.66×, mean **2.79×**. An ideal brick-wall FFT benchmark predicted 4.47× —
always re-measure the thing that actually ships.

---

## 5.6 Sounding like a person, and reaching a phone

Five changes aimed at the two things that are not model quality: how it *feels*, and
whether it can be reached at all.

### The repair line says why

`room_snr_db()` runs on every turn — it was previously computed *inside* the denoise
branch, so with the toggle off (the default) it never ran at all, and the inspector
showed no room reading either. Hoisted, it costs two numpy percentiles and it buys this:

```
   before   "Sorry, could you say that again?"         every failure, identical
   after    < 6 dB   "there's a lot of background noise — could you move somewhere
                      quieter?"                        because repeating will not help
            < 11 dB  "the line isn't very clear…"
            else     the plain line
            unknown  the plain line — never invent a diagnosis
```

Twelve languages plus romanised Hindi. Deliberately **not** applied to the
missing-STT path: blaming the caller's background noise for our absent model would be a
confident lie.

### Closing the turn when the words are finished

`looks_incomplete()` already reads the words to *extend* the window. `looks_complete()`
is the inverse and is much stricter, because the two mistakes are not symmetrical:

```
   a false "incomplete"   costs the caller a pause
   a false "complete"     CHOPS THEIR SENTENCE IN HALF   ← the failure already tuned
                                                            for twice
```

So it fires on exactly two shapes that admit no continuation — a phone number that has
reached full length when a phone is what we asked for, and a bare yes/no alone. Not a
name ("Manu" then "Mishra"), not a time ("tomorrow" then "morning"). Those close at
400 ms instead of 800–1200.

### A listening noise

A filler covers *our* silence while we think. A backchannel is what a listener says
while the *other* person is still talking, and its absence is the most machine-like
thing about a long turn. Pre-loaded to the client at greeting time so it lands in the
pause itself rather than after it, then gated on four conditions — once per turn, only
past 2.6 s, only into a real gap, never while we are speaking. Each one removes a way it
becomes an interruption, which is far worse than staying silent.

### Hearing ourselves

```
   real AEC   y[n] − ŷ[n]    subtract a prediction, keep the residual
   this       keep or drop   a whole frame, on the evidence
```

Normalised cross-correlation against what we played. Measured across 16 conditions:

```
   our own voice — any delay, any attenuation, plus room noise    0.99 – 1.00
   the caller, a fan, white noise, the caller OVER our echo       0.01 – 0.27
                                                    threshold 0.62, in the gap
```

Refusing a frame can only lose audio that was mostly echo; subtracting badly can corrupt
audio that was mostly the caller. Given the browser already has real AEC underneath,
the conservative one is right — and it composes with a proper AEC later rather than
blocking it. Fails open everywhere.

*Found while building it:* probing the head of the frame scored a 300 ms-delayed echo at
0.451 and passed it straight through. Echo is **always** delayed, so that was the only
case that mattered and it was the one that failed. Probing the loudest window instead
made it independent of where in the frame the echo begins.

### Answering before they finish

STT already comes off the critical path when the guess is right. This takes the LLM off
it too: generate against the speculative transcript, adopt the result only if the
committed transcript is **identical**, discard otherwise.

The constraint that shapes it: **a guess never mutates session state.** `begin_user`
appends history and folds the caller's numbers into the grounding set, and this system
has already been broken once by letting a guess mutate state — a voiceprint enrolled
from a speculative fragment locked a caller out for a whole call. So the speculative
path builds its messages by hand, streams into a buffer, and touches nothing; the reply
is then replayed through the *same* loop the live stream uses, so the guard, the
sentence splitter and the history behave identically.

*Measured and rejected first:* speculative **prefill** — warming the KV cache with the
guessed turn — saves only 116–127 ms, because the 6,089-token prefix is already cached
and the user turn is 20–40 tokens. Not worth an extra round-trip.

### A seam for a phone number

```
   recv_audio()   16 kHz mono float32, one utterance at a time
   send_audio()   the same, back
   send_text()    what was said
   hangup()
```

Four methods. Everything above — isolation, the gate, the guard, the router — is already
transport-agnostic and stays untouched. What a browser supplies free and a carrier does
not is supplied here: μ-law, resampling, and an endpointer using the *same* windows the
browser learned from real callers being cut off.

The μ-law codec is hand-written rather than `audioop`, which was removed in Python 3.13 —
the telephony path is the part aimed squarely at the future and should not be the first
thing to break on a modern interpreter. Pinned at 99.5 % byte-identical to the stdlib
implementation while that still exists to compare against.

Pipecat is an **optional** dependency used for transport only. It has no diarization and
no voice isolation, so wiring its STT/LLM/TTS would trade away the parts of this codebase
that took the most work — but Exotel and Plivo are built in, and that is the one piece
here that cannot be written in an afternoon.

---

## 5.7 What a full audit found

Five defects, three of them introduced by the naturalness work in 5.6 — which is the
argument for auditing after building rather than trusting the tests that shipped with it.

**The adaptive noise floor never adapted.** `noiseFloor` was written in exactly one
place, inside the *energy-fallback* branch of the VAD — but read by the self-echo bar and
the drop-tiny-utterance check, both of which run whichever detector decided. So the
moment you installed Silero, which the setup recommends, it froze at its initial `0.004`
for the whole call and two "adaptive" thresholds became constants. In a loud room the
guards that should have risen with the noise never moved. Now learned on both paths, from
Silero's verdict, which is the better one.

**The latch guard reported 0.0 s, always.** `uttMs` was zeroed before the line that read
it, on the one path where the server never sees the clip and the inspector is the
caller's only explanation. The same "capture before zeroing" mistake the file had already
fixed twenty lines above.

**A speculative reply outlived its call.** Cancelled on four paths but not in `finally`,
so a client dropping mid-guess left an Ollama request running for a call that had ended —
competing with live callers for the model. The comment directly above it warns about
exactly this, for the idle watcher.

**`EchoSuppressor.reset()` was never called.** Written with a docstring explaining it
must run on barge-in "so a stale tail cannot explain away the words they interrupted us
with", and nothing called it. Its own test passed. Now wired into `cancel_current()`.

**The backchannel was invisible to the echo suppressor at the moment it played.**
Recorded as ordinary output at greeting time, then murmured minutes later — by which
point the 2 s rolling reference had long forgotten it. Measured: recognised right after
preload, invisible ten seconds on. Audio the client holds and plays on its own schedule
now goes to a separate reference that does not age out, and survives `reset()` because
the client still has it.

### And two changes the audit argued for

**The endpoint window is sized to the question.** People do not pause the same way for
every answer — reading a phone number aloud has long gaps between digit groups. One
learned per-caller number either cut those in half or made every yes/no wait for them.
The server already knew what it had asked; it now says so.

**A latched turn is offered for isolation before being thrown away.** The guard exists
because continuous sound is not a person — but a caller talking *over* continuous noise
never gives a clean pause either, so they hit it too and were discarded and asked to
repeat into the same noise. Isolation exists to pull one voice out of exactly that and
never got the chance, because the turn died in the browser.

It is offered, not forced: with no voiceprint the server cannot trim against anything, so
it answers with the SNR-aware repair instead. And a salvaged turn may never *teach* the
voiceprint — `may_learn=False` gates all four learning paths — because the thing that
made it latch is exactly the thing that would poison the print. Measured before that
guard: a latched clip scored the real caller **0.07** against their own voice.

---

## 5.8 Answering without the model

Some questions are asked almost word-for-word, constantly — *"what are your timings"*,
*"what is the consultation fee"* — and the pack already carries the answer written out,
in every language it supports. Generating that sentence is strictly worse than quoting
it:

```
   fast path (quote the pack)        0.45 ms
   LLM generation, same answer       3219 ms
                                     ────────
   saved                             3218 ms     ~7000×

   …and the same reply in Telugu, which inflates 6.2×:
   generate  ≈ 20 s        quote  = 0 s   (the pack carries `a_te`)
```

Three things follow from skipping the model, and each is worth more here than it would
be elsewhere: it **cannot invent**, because the sentence *is* the fact — the grounding
guard has nothing to catch. It costs **no generation**, which matters most in exactly
the languages where latency hurts. And it is **deterministic**, so the same question
gets the same answer every time.

### What decides whether it fires

Calibrated from the distribution, not guessed — the 0.55 speaker threshold was chosen on
synthetic audio and proved wrong by more than 2×:

```
   English, near-verbatim    0.63 – 1.00      "what are the timings", "what's the fee"
   off-topic                 0.15 – 0.41      "tell me a joke", "who won the election"
   INDIC, near-verbatim      0.21 – 0.49      ← overlaps off-topic
                                        threshold 0.55, plus a 0.15 margin over runner-up
```

So it answers English confidently and **declines for Indic** — which is the correct
answer rather than a shortcoming to paper over. An Indic question is matched against the
pack's `k_hi` / `k_te` *keyword lists*, and a natural sentence does not look like a
keyword list. Those callers take the normal path, exactly as they do today.

The honest irony: the fast path helps English most, and Indic is where latency actually
hurts. Closing that is precisely what a neural backend is for — and the measurement
above is the argument for enabling one, not a claim that the default already does it.

### Why char n-grams are the default

```
   char n-gram TF-IDF     no dependency · every script · handles Indic morphology
   neural embeddings      better at PARAPHRASE · ~1.2GB · opt-in
```

"సమయం" and "సమయంలో" share every n-gram and are different words — the shipping retriever
needed a hand-written prefix rule for exactly this. What n-grams *cannot* reach is true
paraphrase: *"how much to see a doctor"* against *"what is the consultation fee"* shares
almost no characters. That is the gap the neural backend closes, and requiring a gigabyte
of weights to answer *"what are your timings"* is the wrong default for a project whose
whole premise is running on a laptop.

**Measured with Qwen3-Embedding-0.6B, on this machine:**

```
   cold load           243 s        once, at startup
   warm query           40 ms mps · 150 ms cpu
   index 60 entries    1.8 s        once, cached with the pack
```

It does close the Indic gap the lexical backend cannot:

```
                       char n-gram              Qwen3-Embedding
   Indic near-verbatim   0.21 – 0.49              0.47 – 0.69
   off-topic             0.15 – 0.41              0.25 – 0.40
                          OVERLAPS                 SEPARABLE
```

**But only when used correctly, and that turned out to be the whole story.** These
models are trained with an instruction prefix, and without it they compress everything
into one high-similarity band:

```
   no prefix                indic 0.54 – 0.81   off-topic 0.43 – 0.66   OVERLAPS
   WITH instruction prefix  indic 0.47 – 0.69   off-topic 0.25 – 0.40   separable
```

The first version of `NeuralBackend` did not use it. Measured that way the model looked
useless — which would have been the wrong conclusion, drawn from the wrong usage.

**And there is a conflict worth knowing before you enable it.** Qwen3-Embedding needs
`transformers >= 4.51`; `parler-tts` pins `== 4.46.1`, and parler is the Indic TTS. In
one environment you can have Indic *retrieval* or Indic *speech*, not both. On the GPU
preset, prefer the speech — a caller who cannot hear an answer is worse off than one
whose answer took the normal path to reach them.

### What made it affordable

`load_pack()` re-read and re-merged two YAML files on **every session** — 51 ms, paid at
exactly the moment a caller is waiting for a greeting — and returned a fresh dict, which
threw away everything derived from it. The semantic index and the expectation vocabulary
both live on the pack, so both rebuilt per call.

```
   load_pack   51.07 ms  →  0.13 ms      cached on (path, mtime)
   index_for    4.17 ms  →  0.12 ms      no longer rebuilt per session
   test suite     ~26 s  →   2.45 s      as a side effect
```

Keyed on modification time rather than name, so editing a pack during development still
takes effect on the next call without a restart. Packs are now **shared between
concurrent calls**, which is only safe because nothing writes per-session state into
them — a test walks the AST of every module to keep it that way.

### The other two uses of the same index

**Ranking.** Measured on the shipping retriever: rank 1 was always the business *name*
and rank 2 often the *address*, with about half the injected facts irrelevant — which is
precisely the documented failure of a 4B model here, that it answers the easy question
(the address) and drops the one that was asked. We were feeding it the distraction.
Reordering is free: those facts are appended *after* the cache-stable prefix either way.

**Rescue.** When the model is unreachable or breaks mid-stream, the reply today is
"I don't have that detail" — sometimes to a question written down three lines away in
the knowledge base. Now the pack is consulted before giving up.

### What it will not do

It never fires **mid-booking**. The caller is answering *our* questions, and a knowledge
fact dropped into slot collection abandons the collection — the same rule
`recovery_line` already follows.

---

## 5.9 Seeing it: the audio inspector, drawn

Every claim in section 5 is a measurement somebody has to take on trust. The docked
inspector narrowed that a little — it reports whether anything was removed from a turn
— but "was anything removed?" and "removed from WHERE?" are different questions, and
only the second one tells you whether the isolation cut the right person.

The full-page inspector (🔬 in the sidebar) draws the turn:

    ████████░░░░░░░░████████████████        green  = you, kept
      you    stranger      you              red    = another voice
                                            faded  = never reached the recogniser

Three decisions in it are worth stating, because each was a way to get it quietly wrong.

**The recording is the browser's own copy.** Nothing is sent back from the server. The
caller's audio stays on the caller's machine, the page still works after the call has
ended, and there is no second encode to drift out of alignment with the timestamps.

**Colour follows the voiceprint match, not the cluster id.** The diarizer numbers its
clusters in the order it happens to find them, so colouring by that id makes "green"
mean the caller only when the caller spoke first. Ordering by similarity makes green
mean *you* on every turn, which is the whole point of looking.

**Shading follows the exact kept ranges, not a per-segment flag.** The second-pass
rescan cuts INSIDE a segment — that is what it is for. A per-segment boolean would
either lose that cut or claim the whole segment went, and both read as a picture of
something that did not happen. `keep_matching_speaker` therefore reports two things:
the boundaries it found, and the ranges that survived them, both in the timeline of
the clip the caller recorded. The second-pass ranges have to be mapped back through
the concatenation to get there — in the trimmed timeline the gaps are already closed
up, so 1.0s into it is not 1.0s into the recording on screen.

Clicking a segment plays that moment on its own and says whether its words are in the
transcript or were removed before transcription. There is deliberately no per-segment
transcript: Whisper is given the kept audio joined up, so there is only one transcript
for the turn, and inventing a split of it would be exactly the kind of confident
fabrication the grounding guard exists to stop.

One thing the page exposes rather than hides: a turn answered from the speculative
transcript is never analysed as a whole, so it has no segments to draw. It says so.

---

## 5.10 A second recogniser, and what it bought

STT is the dominant cost on the audio path — 1070ms at 3s of speech, against 66ms for
isolation and 34ms for the speaker gate — and every CPU-side lever had been measured
and rejected: more CTranslate2 threads is WORSE on an M1 (auto 1691ms, 8 threads
2212ms, 10 threads 3798ms), and greedy decoding was not reliably faster. The remaining
unused resource is the GPU already in the machine.

    clip     faster-whisper (CPU int8)   whisper.cpp (Metal)   speedup
    3.5s                       1562ms                 833ms     1.87x
    7.0s                       1488ms                 797ms     1.87x
   10.5s                       1519ms                 876ms     1.73x

Mean WER over the same four clips: 0.25 against 0.29 — so unlike DeepFilterNet, the
speed is not paid for in accuracy. It is still opt-in, because it needs a system
package and a 490MB model that a clone does not bring, and this project's premise is
that cloning and running works.

Two things had to be right for it to be a real second backend rather than a second set
of rules:

**The hardening is shared, not reimplemented.** `stt.judge()` is now one function that
both backends pass through. Each guard in it exists because of a specific way a real
call failed, and a backend that quietly skipped them would reintroduce all of it while
looking like a bad model rather than a missing check.

**"Unknown" is not "confident".** whisper.cpp enforces the no-speech and log-probability
thresholds inside the binary — the same values — but does not report what it measured.
Passing 0.0 for those would read as certainty and switch off the half of the artifact
list that needs corroboration. `judge()` takes None and knows the difference: phrases a
caller would never say ("subtitles by the amara.org community") go regardless; phrases
they might ("bye bye") survive without evidence to the contrary.

Found while building it, and worth recording because the obvious move is the wrong one:
**pointing `--vad` at one of the three Silero `.onnx` files already in this repo does
not fail with a message — it ABORTS the process.** whisper.cpp wants Silero in its own
ggml format. From Python that abort looks like a broken recogniser, not a wrong file.

---

## 5.11 LiveKit, evaluated

Assessed the way Pipecat was, and the verdict has the same shape: **transport only.**

What would genuinely be gained is a WebRTC transport instead of a WebSocket carrying WAV
blobs — jitter buffering, packet-loss concealment, Opus — plus SIP, so the same agent
answers a real phone number. The server is Apache-2.0 and self-hostable, which is the
only reason it can be considered here at all.

What would not:

- **Its noise cancellation and background voice cancellation are Krisp models available
  only through LiveKit Cloud**; the self-hosting instructions say to remove that plugin.
  The one feature overlapping what this codebase spent most of its effort on is exactly
  the one you cannot self-host — and on the recognition path denoising is already
  measured to hurt here (DeepFilter won 1 of 30 SNR cells, net −400% word recall).
- **Its turn detector** is a real model and a real idea, but it is bound to the
  `livekit-agents` session object, ships under the LiveKit Model License rather than an
  OSI one, and covers 14 languages: Hindi yes, **Telugu no.** Our endpointer learns per
  caller and `guard.looks_incomplete()` reads the words. Adopting theirs would mean
  restructuring into their agent loop to get a model that cannot serve the languages the
  GPU work is for.

`LiveKitTransport` implements the same four methods as everything else at that seam, so
isolation, the speaker gate, the router and the guard are untouched by it. A test
asserts none of those services leak into the adapter — checking the code, not the
docstring, since the docstring has to be free to name them.

---

## 5.12 Turn-taking: a third signal, from the voice

Deciding when the caller has finished has been the most re-tuned thing in this project.
Three signals were in play, and all three count or read rather than listen:

    how long they have been talking   →  short answers close fast, sentences wait
    how THIS caller pauses            →  learned per call, from pauses they spoke through
    what the words say                →  looks_incomplete / looks_complete

The gap they all share: none of them can tell a finished sentence from a caller who
stopped after a grammatically complete clause to think. "I need an appointment" is a
whole sentence and also the first half of "I need an appointment tomorrow morning".

The signal nobody was reading is the one every human listener uses. **Pitch and energy
fall together through the end of an utterance, and are held when the speaker means to
carry on.** It is not a learned fact about English — it is what happens when a speaker
runs out of breath support — and it holds in Hindi and Telugu declaratives too. It also
costs nothing: 1.6ms of numpy on audio that has already been decoded.

**Why not a learned turn detector.** LiveKit ships one and it is the right idea. But it
is bound to their agent session, ships under a non-OSI licence, and covers 14 languages
of which Telugu is not one. Training our own needs the real call recordings this project
does not have yet. Declination is the part of the signal available without data.

**The measurement is why the design looks the way it does.** The first version tested
the detector against clips of `say` speaking "My name is" — and scored 3/8, calling
every fragment finished. The fixture was wrong, not the arithmetic: asking a synthesiser
for a fragment gets you a fragment synthesised AS a complete sentence, with a textbook
final fall. Speaking the whole sentence and truncating it where the caller would still
be talking is what produces genuinely mid-utterance audio.

Measured properly, neither half of the signal works alone:

    clip                            score   slope st/s   decay   verdict
    "My name is Manu Mishra."        10.2         -6.6    0.64   unsure
    …the same, cut after "is"         7.0         -6.7    0.96   unsure
    "I would like to book an appt."   8.1         -7.9    0.97   unsure
    …the same, cut after "an"         5.1         -2.5    0.74   unsure
    "Yes that is correct."           36.4        -27.8    0.14   finished
    "What are your charges?"         32.2        -25.2    0.30   finished
    "I need an appointment" (cut)     3.2         -2.5    0.93   holding
    "Are you open on Sunday?"         3.1         -0.9    0.78   holding

Read rows one and two: the same slope, −6.6 against −6.7, and opposite answers. Row
three has a *higher* energy decay than three of the four unfinished clips. Either signal
alone is a coin toss on those. Combined — `-slope + 10 × (1 − decay)` — the finished
clips score 8.1 to 36.4 and the unfinished 3.1 to 7.0.

That separates, but by 8.1 against 7.0, and nine clips of synthetic speech is not a
distribution. **So there is a dead band.** Between 5 and 12 the detector returns
"unsure", which changes nothing at all. On the calibration set it acts on four clips and
is wrong on none; the five it declines are exactly the ones where a threshold would have
been guessing. *Wrong* is the only number that matters here, because this runs on the
path that decides whether to cut somebody off.

**It is never the decider.** It scales the window by 0.80 or 1.25 within bounds the other
signals already set, and `holdForMore` — the words — always wins: a caller who says "my
mobile number is" with a perfect falling contour has still not given us the number. The
shortening is smaller than the lengthening, because the two mistakes are not symmetrical.
The first version had 0.75/1.25, which cut by more than it waited and contradicted its
own docstring; a test caught it.

One honest limitation: a yes/no question reads as "holding" (score 3.1). Rise detection
is not claimed, because it could not be demonstrated. It costs a slightly longer pause
before a question is answered, and the words already handle questions — `looks_incomplete`
returns False on a trailing "?".

---

## 5.13 Keeping the model resident

Timing whisper.cpp against clip length said something odd:

    0.3s of speech → 644ms          10.5s of speech → 589ms

Flat. Almost none of that was the audio — it was spawning a process and reading a 490MB
model off disk, once per turn, for a decode the GPU finishes almost instantly.

`whisper-server` keeps the model loaded. Measured properly — all three backends warmed,
best of 5, in one process:

    clip     faster-whisper   whisper-cli   whisper-server   vs faster-whisper
    0.3s             1420ms         895ms           615ms          2.31x
    3.5s             1738ms         841ms           621ms          2.80x
   10.5s             1674ms         903ms           670ms          2.50x

**A correction worth keeping: a curl probe of the same server suggested 4x. Through the
shipping code path it is 2.3–2.8x.** Two reasons the floor does not vanish — `verbose_json`
costs ~120ms more to serialise than plain text, and Whisper's encoder runs over a padded
30-second window whatever the clip length, which is also why 0.3s is no cheaper than 3.5s.
Measure the thing that ships, not a probe of it.

The server mode is also the more ACCURATE of the two whisper.cpp modes, which matters
more than the 220ms: `verbose_json` carries per-segment `avg_logprob` and
`no_speech_prob`, so `judge()` gets real numbers and the confidence gate, the full
artifact list and language latching all work exactly as they do on faster-whisper. The
CLI has to pass None and lose half of each.

It stays local. The server is a CHILD of this process, on a free loopback port, killed
on shutdown — found the hard way, because `atexit` does not fire when uvicorn takes a
SIGTERM and the first version left a 490MB process listening after the app stopped.

`provider: auto` walks the ladder — server, then CLI, then faster-whisper — so a clone
runs at the best speed the machine can manage with nothing configured. `device: auto`
does the same for torch: CUDA when a card is present, otherwise CPU. Never `mps`:
SpeechBrain's ECAPA fails outright on it, and Apple's GPU is reached through
whisper.cpp's Metal backend instead, which is a separate process and needs no torch
device at all.

---

## 5.14 Three things the model was being trusted with, and should not have been

All three have the same shape, and it is the shape of every fix in section 5: the model
was being ASKED to get something right that a 4B model does not reliably get right, and
the answer is to decide it before the model is involved.

### The diary

"Is Dr Rao free tomorrow at four?" had exactly two possible answers before this, and both
were bad. Refuse — the grounding guard blocks any time the pack does not contain. Or
invent one, which is the documented failure here: asked for a neurologist the clinic does
not employ, the model quoted "₹500", a real fee belonging to a real doctor.

So the diary goes in the pack, the same way prices and timings already do — some slots
free, some `FULLY BOOKED`, because being told a day is full is a useful answer and being
told a booked day is free is worse than a refusal. In a real deployment this block is
what a calendar or EMR integration writes; as YAML it is the same shape, which is the
point — nothing above it changes later.

Two matching bugs found while building it, both in the direction that confirms
appointments nobody can keep:

- **"tomorrow" is a substring of "day after tomorrow."** A first-match day search books
  the caller for the wrong day and then confirms it, confidently. Longest match wins.
- **"Dr Rao" matched every doctor**, because the fallback matched on word "dr". A
  fully-booked dermatologist came back free on the strength of a cardiologist's slot.

`slot_is_free` returns None for "cannot judge" — no diary, no day named, no clock time —
and every caller treats that as permission. Most packs have no diary at all, and refusing
their bookings would break every other vertical in this repo.

**A clash is answered, not silently re-asked.** Dropping the slot alone asks "what day and
time would suit you?" of somebody who just answered exactly that, which reads as not
listening and is how this codebase once produced a twenty-turn call that stored one word.
Now: *"That one is already taken, I'm afraid. Dr Rao has 4:00 pm or 4:30 pm tomorrow."* —
for the day they asked about, since offering today's 5pm to someone asking about tomorrow
is a different question rather than an alternative.

And the trade-off this project keeps meeting: **every time in the diary widens
`allowed_numbers` and weakens the grounding guard.** "4" being a real slot means "4" can
no longer be caught as invented. Availability is worth that; a longer list would not be.

### Who the call goes to

One escalation was doing two jobs. A caller describing chest pain needs clinical staff and
the ambulance number now. A caller who is angry, wants a refund, or is asking for whoever
is in charge needs the **manager** — and hearing "let me connect you to a team member" is
precisely what makes that caller repeat themselves, louder.

Two keyword lists, manager checked first, so "I want your manager about this emergency" is
a manager call whatever else is in it. The clinic names its manager and gives her direct
number in the transfer line, because a transfer that drops must leave a way back. Every
other pack inherits the routing from `_base.yaml`.

### The domain guardrail

The model self-declares `kind: out_of_scope`, and that mostly works — "mostly" being the
same word that made every other rule in this file necessary. Asked to write a poem, a 4B
model writes a poem. Asked to ignore its instructions, it often obliges. Asked who the
prime minister is, it answers, in the voice of a clinic, sometimes wrongly. **The
grounding guard cannot catch any of it, because none of it contains an ungrounded
number** — the same blind spot as the invented-services bug, fixed the same way.

Two classes, treated differently on purpose:

- **Attempts to change what the agent is** — "ignore previous instructions", "you are
  ChatGPT", "pretend you are…" — are refused unconditionally. No caller phrases a real
  request that way.
- **Whole other subjects** — code, capitals, cricket scores, recipes — are refused only
  when the turn shows no sign of being about the business, checked against the pack's own
  vocabulary and knowledge tags rather than a list written in the engine.

That second condition is the whole design. `cricket` and `script` are both on the marker
lists, and *"do you do a cricket physio programme"* and *"can I get my script refilled"*
are both legitimate — in Indian English a prescription is routinely called a script. The
cost is asymmetric in the direction this codebase keeps re-learning: turning away a real
caller is far worse than answering one silly question.

It runs in `begin_user`, before the model, so a refusal costs no generation at all.

---

## 5.15 One ladder, two passes, and a dial

Three changes that are really one: turn-taking stopped being three implementations and
became a module, and once it was a module the other two became small.

### The ladder had drifted, silently

"Have they stopped talking" was being decided in three places that could not see each
other — the browser endpointer, the flags `server.py` pushed to it, and
`transport.Endpointer`. They had already diverged. The telephony endpointer was a flat
800/1200ms with **no learned pause, no per-slot extra and no prosody**, so the first
caller on a real phone line would have been endpointed by exactly the rules two rounds
of live testing tuned away from ("it breaks my long sentences").

`zensuvidha/turn.py` holds the numbers once. The browser is *sent* them on connect and
evaluates them; the phone endpointer imports them. A test runs the browser's fallback
copy in Node and fails if it disagrees with Python — proven by changing one constant.

The order of evidence is the design, and each step is a bug that was fixed:

    the WORDS    looks_incomplete / looks_complete   ← outranks everything
    a FILLER     "um", "मतलब", "यानी"                ← new
    the VOICE    the pitch contour
    the CALLER   how long THIS person pauses
    the CLOCK    how long they have been talking

**The filler signal is what silence cannot see.** A caller trailing off on "um" has not
finished — they are searching for the next word, and the gap looks identical to a
finished sentence. This is the signal ElevenLabs' turn-taking model reads and ours did
not. It is deliberately narrower than the existing `_HESITATIONS` list: "haan" and
"ठीक" belong there and *not* here, because a bare "yes" is a complete answer and
treating it as hesitation would make every confirmation in every booking wait longer.

It also earns *less* time than a dangling postposition — weaker evidence, smaller
extension — and the two never stack, or one hesitation is charged twice.

### Two-pass ASR

The speculative frame taken at ~450ms of silence exists to size the endpoint window and
show live text. Three turn-taking signals are computed from it and nothing else, and all
three were gated behind a full 621ms recognition — so the endpointer could not react
until the caller had already been silent for most of the window it was trying to size.

Wispr Flow runs a ~120M realtime model for partials and a large one for the commit. Same
idea here, and the machinery already existed; it was just running `small` twice.

    partial (tiny)   171 ms
    commit  (small)  722 ms      4.2x

The partial **never** reaches the LLM, the guard, or the transcript — `tiny` is
measurably worse and its errors must not reach a booking. `Session.transcribe(partial=True)`
also returns before the language-lock code, for the same reason a speculative frame
never enrols a voiceprint: a guess must not move call-wide state.

### The eagerness dial

The one knob an operator should have. A clinic taking bookings from elderly callers
wants Patient; a restaurant taking table numbers wants Eager.

    eagerness   short   long   filler   hold   settled
    eager        560     840    1240    1560     400
    normal       800    1200    1700    2100     400
    patient     1160    1740    2390    2910     400

It **scales** the ladder rather than replacing it, so every measured rule still holds —
and a test asserts the ordering of the signals is identical on all three settings. Note
`settled` is not scaled: "the words are finished and nothing can follow" is a fact about
the transcript, not a preference, and making a completed phone number wait longer is
just a slower call.

### Speculative reply, measured instead of configured

It was built, shipped, and switched off, because on one local Ollama it made turns 2.6x
SLOWER — requests serialise per model, so the guess *queued in front of* the real
generation. Whether that happens is a property of the machine, not of the config file.

`speculative_reply: auto` now runs one short generation and then two together at
startup. Under 1.5x the single means they genuinely overlap and it switches on; at ~2x
they serialise and it stays off, saying which in the log. On this laptop it stays off,
correctly, and on a GPU box with continuous batching it will not.

---

## 6. Why a call can never wedge

The client enters "thinking" the instant it ships audio, and only leaves on a reply. So
**every path that declines to answer must say so.**

```
  browser ──commit──▶ server
                        ├─ has words   ──▶ reply_start → audio → reply_end
                        └─ has nothing ──▶ commit_miss
                                             └─▶ browser sends the audio
                                                 it was HOLDING
  browser ──audio───▶ server
                        ├─ nothing recognisable, ≥2s ──▶ spoken "say that again?"
                        └─ nothing recognisable, short ─▶ turn_dropped
                                                          (mic handed back)

  caller silent:  30s ──▶ "are you still there?"    75s ──▶ goodbye, close
```

**Silence is often the right *reply*. It is never the right *protocol*.**

This came from a real bug: `elif text:` had no `else`. A commit with an empty speculative
transcript dropped the turn in total silence — and the browser had already discarded its
recording on the strength of having sent `commit`. Both sides then waited for the other
forever. No error, no reply, UI frozen on "thinking". That, not the LLM and not the noise
filtering, is what made calls look "stuck".

---

## 7. Operating envelope and latency

Caller-relative SNR at which the turn still works — words intact **and** caller still
recognised:

```
  mains hum 50 Hz    ████████████████████████████████  −6 dB
  traffic rumble     ████████████████████████████████  −6 dB
  music, no vocals   ████████████████████████████████  −6 dB
  another speaker    ██████████████████████            0 dB
  pink / room tone   ██████████████████                +3 dB
  music with vocals  ██████████████████                +3 dB
  TV / babble        ██████████████████                +3 dB
  white noise        ██████████████                    +6 dB
  fan / AC           ████                              +12 dB  ← worst case
                     └── more tolerant ──────────────┘
```

Two patterns. **Low-frequency and tonal interference is nearly free**; broadband is
expensive. And **anything containing a human voice takes out the speaker gate before it
takes out the transcript** — the words usually survive, and recognising them as *yours* is
what fails.

```
  isolate    ██                          66 ms   (~6% of the path)
  gate       █                           34 ms
  STT        ████████████████████████  1070 ms   ◀── dominant
  LLM        ██████████               400-600 ms
  TTS        ████████                   385 ms
```

The audio path is **recognition-bound**. Four attempts to speed isolation further were
measured and rejected: ECAPA on Metal fails outright (SpeechBrain retains internal CPU
tensors); batching the thirds gate is a real 1.5× on a 60 ms stage; fast-decode mode is not
reliably faster; and `cpu_threads` is *worse* with more threads on Apple silicon —
auto 1691 ms, 8 threads 2212 ms, 10 threads 3798 ms.

---

## 8. What makes it different

Not the model choices — anyone can wire Whisper to Ollama. Four things, each of which only
came from measuring:

1. **Denoise the decision, not the recognition** — then, on more data, *do not denoise at
   all* on the recognition path. Both were beliefs held and then overturned by numbers.
2. **"One label" is not "one person"** — a clusterer's output is an opinion, so a single
   surviving cluster is re-checked for being two people merged.
3. **A gate must earn the right to refuse** — silencing the real caller is a far worse
   failure than answering a stranger once.
4. **Thresholds calibrated on synthetic audio are guesses about real audio** — 0.55 against
   a real range of 0.27–0.41 is not a tuning error, it is a category error.

401 tests, and every measured claim in these documents is set in monospace so a reader can
tell which ones carry a number behind them.

---

## 9. What it cannot do

- **Fan and air-conditioner noise needs +12 dB** — the caller must be four times louder than
  a ceiling fan, the commonest condition in an Indian small business, and denoising cannot
  rescue it (measured −67%).
- **Five languages have no voice** on macOS: Malayalam, Gujarati, Punjabi, Odia, Urdu. Those
  callers see the reply and hear nothing. Warned at startup; fixed only by the GPU preset
  with Indic-Parler.
- **Loud audio at the microphone destroys the voiceprint** (0.07 against the caller's own
  voice). Mitigated by the latch guard, not solved.
- **Echo cancellation comes from the browser.** A telephony transport has none. A
  level-based guard is in place, but real server-side AEC is required before a phone line.
- **Every remaining threshold is calibrated on synthetic speech.** A real microphone has
  already proved one of them wrong by more than two-fold. Real recordings are the single
  highest-value missing input to this project.
- **No phone number yet.** Two routes exist at the transport seam and neither has been run
  against a live carrier: Pipecat (Exotel and Plivo built in) and LiveKit (SIP, and a
  self-hostable Apache-2.0 server). Neither brings diarization or voice isolation — that
  work stays on this side either way.
- **The second recogniser has only been measured on synthetic speech.** whisper.cpp is
  1.7-1.9x faster with slightly better WER on `say` clips; whether that holds on a real
  microphone is unmeasured, and this project has been wrong about exactly that before.

---

*See it running: [`demo/`](../demo/). The measurements: [ARCHITECTURE.md](../ARCHITECTURE.md).
The same system drawn: [DIAGRAMS.md](DIAGRAMS.md) · [ARCHITECTURE.html](../ARCHITECTURE.html).*
