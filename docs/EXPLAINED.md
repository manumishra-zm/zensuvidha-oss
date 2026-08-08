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
- **No phone number yet.** Pipecat is the route — Exotel and Plivo are built in — but it has
  no diarization or voice isolation at all, so adopting it means re-homing this work into
  their processor model.

---

*See it running: [`demo/`](../demo/). The measurements: [ARCHITECTURE.md](../ARCHITECTURE.md).
The same system drawn: [DIAGRAMS.md](DIAGRAMS.md) · [ARCHITECTURE.html](../ARCHITECTURE.html).*
