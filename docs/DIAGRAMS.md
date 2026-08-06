# ZenSuvidha OSS — Diagrams

Every diagram here renders on GitHub. They are the same system described in
[ARCHITECTURE.md](../ARCHITECTURE.md); this file is the visual index. For the whole
thing laid out as one page — including these nine, pre-rendered so they need no
network — open [../ARCHITECTURE.html](../ARCHITECTURE.html); to move
the boxes around yourself, open [../ARCHITECTURE.excalidraw](../ARCHITECTURE.excalidraw)
and [../ARCHITECTURE-pipeline.excalidraw](../ARCHITECTURE-pipeline.excalidraw) on
excalidraw.com.

Numbers shown were measured on an M1 Pro — `qwen3:4b` via Ollama (100% Metal),
`whisper-small` int8.

---

## 1. What talks to what

```mermaid
flowchart LR
  subgraph B["🖥️ Browser · web/index.html"]
    direction TB
    MIC["Microphone<br/><small>AGC off</small>"]
    WK["AudioWorklet<br/><small>biquad 300–3400 Hz</small>"]
    VAD["Silero VAD v5<br/><small>ONNX-WASM, 2.3 MB</small>"]
    EP["Endpointer<br/><small>800 / 1200 / →2000 ms</small>"]
    INS["Audio inspector<br/><small>spectrogram + per-turn rows</small>"]
  end

  subgraph S["⚙️ Server · FastAPI"]
    direction TB
    PIPE["pipeline.prepare<br/><small>isolate → denoise</small>"]
    STT["faster-whisper<br/><small>CTranslate2</small>"]
    GATE["Speaker gate<br/><small>ECAPA-TDNN</small>"]
    LLM["Qwen3-4B<br/><small>Ollama, forced JSON</small>"]
    GRD["Guard<br/><small>verify every sentence</small>"]
    TTS["TTS<br/><small>script-routed</small>"]
    DB[("SQLite WAL")]
  end

  MIC --> WK --> VAD --> EP
  EP -->|"WAV frames over WS"| PIPE
  PIPE --> STT --> GATE --> LLM --> GRD --> TTS
  GRD --> DB
  TTS -->|"audio + JSON"| INS

  style B fill:#F0EAF9,stroke:#7A4FB5
  style S fill:#E2F0F2,stroke:#0A6E7C
```

---

## 2. One turn, in order

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant BR as Browser
    participant SV as Server
    participant M as Models

    C->>BR: speaks
    BR->>BR: biquad → Silero VAD → endpointer
    Note over BR: latch guard: 7 s with no pause<br/>= not a person → discard, ask again

    BR->>SV: WAV frame (~3 s)
    SV->>M: pyannote segmentation + ERes2Net clustering
    M-->>SV: who spoke, when
    Note over SV: isolate on RAW audio —<br/>identity is worse on denoised (0.675 vs 0.596)

    SV->>M: faster-whisper
    M-->>SV: transcript (or dropped: no_speech / logprob / degeneracy)
    SV->>M: ECAPA voiceprint
    M-->>SV: is this the caller?

    SV->>M: Qwen3 (streamed, forced JSON)
    loop each finished sentence
        M-->>SV: tokens
        SV->>SV: guard — kind, echo, repetition,<br/>degeneracy, numbers, language
        SV->>M: TTS for this sentence
        SV-->>BR: audio + text
        BR-->>C: plays while the rest generates
    end
    SV-->>BR: reply_end
```

---

## 3. Voice isolation — the decision tree

The wide path is the common one. Everything else only fires when its condition holds.

```mermaid
flowchart TD
  A["turn arrives"] --> B{"voiceprint<br/>exists?"}
  B -->|"no · turn 1"| SKIP["skip isolation<br/><b>0 ms</b>"]
  B -->|yes| SEG["pyannote-segmentation-3.0<br/>+ ERes2Net clustering"]

  SEG --> C{"how many<br/>clusters?"}
  C -->|"1"| D{"clip ≥ 4 s?"}
  C -->|"2 or more"| SCORE["score EACH cluster<br/>against the caller's print"]

  D -->|no| DONE1["done<br/><b>~49 ms</b>"]
  D -->|yes| THIRDS["thirds gate<br/>3 ECAPA passes"]

  THIRDS --> E{"is every third<br/>the caller?"}
  E -->|yes| DONE2["done<br/><b>~124 ms</b>"]
  E -->|no| WIN["window rescan<br/>≤ 8 passes"]

  WIN --> F{"a RUN of low<br/>windows?"}
  F -->|"yes · relative AND absolute"| SPLIT["re-split — the clusterer<br/>merged two people"]
  F -->|"no · one dip"| DONE3["a pause, not a person<br/>leave it alone"]

  SCORE --> G{"keep which?"}
  G -->|"≥ 0.55, or within 0.40<br/>of best AND ≥ 0.40"| KEEP["trim to the caller"]
  KEEP --> H{"only ONE<br/>cluster kept?"}
  H -->|yes| MERGE["re-check IT for a merge<br/><small>a cluster can be two people</small>"]
  H -->|no| DONE4["done"]

  style SKIP fill:#E6F1EB,stroke:#2A6B50
  style DONE1 fill:#E6F1EB,stroke:#2A6B50
  style DONE2 fill:#E6F1EB,stroke:#2A6B50
  style DONE3 fill:#E6F1EB,stroke:#2A6B50
  style DONE4 fill:#E6F1EB,stroke:#2A6B50
  style WIN fill:#F6EFDC,stroke:#87621A
```

**Why the run test exists.** One caller speaking two sentences dips at her own pause;
a stranger's turn is a *run*. Splitting on the dip deleted 0.8 s of her own words:

```
one speaker, two sentences   0.77 0.78 0.74 0.71 [0.50] 0.66 0.67 0.73 0.66
caller then a stranger       0.77 0.78 0.74 0.71 [0.45  0.52  0.35  0.23]
```

---

## 4. The speaker gate — when it may refuse

```mermaid
stateDiagram-v2
    [*] --> NoPrint

    NoPrint --> Provisional: first turn ≥1 s of voice<br/>→ enrol
    note right of NoPrint
      Turn 1 has no print, so
      isolation cannot run on it —
      music here gets enrolled.
    end note

    Provisional --> Provisional: near miss (≥0.40)<br/>→ widen, clock does NOT advance
    Provisional --> Rival: mismatch<br/>→ ANSWER it, remember the voice
    Rival --> Provisional: a different voice<br/>→ reset
    Rival --> NoPrint: the SAME voice twice<br/>→ it is the caller now

    Provisional --> Proven: matched the caller once
    Proven --> Enforcing: print corroborated<br/>(≥3 utterances)
    Enforcing --> Enforcing: stranger → REFUSED
    Enforcing --> NoPrint: 3 refusals, same voice<br/>→ the print was wrong

    note right of Enforcing
      Both conditions must hold.
      A gate that has never matched
      this caller has not earned the
      right to refuse them.
    end note
```

**The measurement behind it:**

| | same speaker | closest impostor | verdict at 0.55 |
|---|---|---|---|
| macOS `say` voices | 0.867 | 0.429 | works |
| **a real microphone** | **0.27 – 0.41** | — | **refuses the caller** |

---

## 5. Noise: what runs, and when

```mermaid
flowchart TD
  A["audio in"] --> B["measure floor-to-voice gap<br/><small>numpy percentiles, free</small>"]
  B --> C{"auto_denoise<br/>enabled?"}
  C -->|"no · DEFAULT"| SKIP["skip<br/><b>0 ms</b>"]
  C -->|"yes"| D{"gap < 10 dB?"}
  D -->|no| SKIP2["clean enough"]
  D -->|yes| DF["DeepFilterNet<br/><b>226–475 ms</b>"]

  SKIP --> STT["Whisper"]
  SKIP2 --> STT
  DF --> STT

  style SKIP fill:#E6F1EB,stroke:#2A6B50
  style DF fill:#F7E7E4,stroke:#9E3A2E
```

**Why the default is off.** Swept across 6 interference types × 5 SNRs — 30 conditions:

```
DeepFilter WON 1 cell, LOST 8, tied 21     net −400% word recall
white hiss −50%      fan/AC −67%      music+vocals −100%
```

It is also the most expensive stage. On the recognition path it costs time *and*
accuracy — the same verdict `noisereduce` got.

---

## 6. The guard chain

```mermaid
flowchart LR
  T["token stream"] --> K{kind}
  K -->|"unknown /<br/>out_of_scope"| SAFE["pre-written refusal<br/>12 languages"]
  K -->|answer| E{echo?}
  E -->|"answering AS<br/>the caller"| SAFE
  E -->|no| R{repeats<br/>last turn?}
  R -->|yes| SAFE
  R -->|no| D{degenerate?}
  D -->|"a clause<br/>looping"| RETRY["close the stream,<br/>then retry"]
  D -->|no| N{ungrounded<br/>number?}
  N -->|"a price in neither<br/>the pack nor the call"| SAFE
  N -->|no| L{wrong<br/>script?}
  L -->|yes| SAFE
  L -->|no| OK["speak it"]

  style OK fill:#E6F1EB,stroke:#2A6B50
  style SAFE fill:#F7E7E4,stroke:#9E3A2E
  style RETRY fill:#F6EFDC,stroke:#87621A
```

It runs **on the stream** — a bad sentence is intercepted before synthesis, because on
a call you cannot un-say a price.

---

## 7. TTS routing — by script, not preference

```mermaid
flowchart TD
  A["reply text"] --> B{"dominant<br/>script?"}
  B -->|"Latin"| K["Kokoro · af_heart<br/><b>385 ms</b>"]
  B -->|"Devanagari"| KH["Kokoro · hf_alpha<br/><b>583 ms</b>"]
  B -->|"Telugu, Tamil,<br/>Kannada, Bengali…"| SYS["system voice<br/><small>macOS say · 1065 ms</small>"]
  SYS --> C{"a voice for<br/>this script?"}
  C -->|yes| PLAY["speak"]
  C -->|"no · 5 languages"| MUTE["silent + mute_reason<br/><small>the UI explains it</small>"]

  K --> PLAY
  KH --> PLAY

  style PLAY fill:#E6F1EB,stroke:#2A6B50
  style MUTE fill:#F7E7E4,stroke:#9E3A2E
```

Fed Telugu, an English Kokoro pipeline produced **2.1 MB and 6.5 seconds** of confident
nonsense — so the script is checked *before* anything is synthesised. Malayalam,
Gujarati, Punjabi, Odia and Urdu have no voice at all on macOS; that is warned at boot.

---

## 8. Why a call can never wedge

```mermaid
sequenceDiagram
    participant BR as Browser
    participant SV as Server

    Note over BR: enters "thinking" the moment it ships audio,<br/>and only leaves on a reply

    BR->>SV: commit (using a speculative transcript)
    alt server has words
        SV-->>BR: reply_start → audio → reply_end
    else server has nothing
        SV-->>BR: commit_miss
        BR->>SV: the audio it was holding
        Note over BR: it never discards the recording<br/>until a reply actually arrives
    end

    BR->>SV: audio
    alt nothing recognisable, ≥ 2 s
        SV-->>BR: spoken "could you say that again?"
    else nothing recognisable, short
        SV-->>BR: turn_dropped
        Note over BR: microphone handed back —<br/>silence is the right REPLY,<br/>never the right PROTOCOL
    end

    Note over SV: caller says nothing at all:<br/>30 s → "are you still there?"<br/>75 s → goodbye, close
```

---

## 9. Where the time goes

```mermaid
flowchart LR
  A["isolate<br/>66 ms"] --> B["speaker gate<br/>34 ms"] --> C["STT<br/><b>1070 ms</b>"] --> D["LLM first token<br/>400–600 ms"] --> E["TTS<br/>385 ms"]
  style C fill:#F7E7E4,stroke:#9E3A2E
```

The audio path is **STT-bound**; isolation is ~6% of it. Four attempts to speed it
further were measured and rejected — ECAPA on MPS (fails outright), batching the thirds
gate (1.5× on a 60 ms stage), `cpu_threads` (*more is worse* on Apple silicon: auto
1691 ms, 8 threads 2212 ms, 10 threads 3798 ms), and STT fast mode (not reliably
faster).

Indic latency is **tokenizer inflation, not runtime**: at 38.7 tok/s a 40-token reply is
1.03 s in English, 3.62 s in Hindi, **6.41 s in Telugu**. MLX was benchmarked at
39.6 tok/s against Ollama's 38.7 and rejected. The GPU preset is the answer.
