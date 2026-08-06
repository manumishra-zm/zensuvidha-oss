# Demo recordings

Two unedited runs of a clinic booking, recorded on an M1 Pro with everything running
locally — no cloud, no API key. Both have audio.

| | Length | Shows |
|---|---|---|
| **[demo.mp4](demo.mp4)** | 4 min 49 s | A full booking with **music playing in the background** — the hard case. Voice isolation is on, and the audio inspector at the bottom left reports what was kept and what was removed on every turn. |
| **[demo2.mp4](demo2.mp4)** | 4 min 07 s | The same booking recorded closer to the screen, so the conversation panel and the per-turn inspector rows are legible. Easier to follow if you want to read what the guard is doing. |

## What to watch for

- **The orb states** — `LISTENING → THINKING → SPEAKING`. Speech starts playing while the
  rest of the reply is still being generated, which is why the agent begins talking well
  before it has finished writing.
- **The audio inspector** (bottom left) — one row per turn: how many voices were found,
  how much was removed, the match score against the caller's voiceprint, and the verdict.
  In `demo.mp4` you can see it discarding the background audio.
- **The agent refusing to invent** — asked to list every slot in the week, it says it
  cannot rather than making one up. That is the grounding guard, not the model behaving.
- **Recovery** — at one point it says *"there's continuous audio in the background — I
  couldn't pick out your voice, could you say that again?"* rather than answering noise.
  A turn it cannot hear is dropped and re-asked, never silently swallowed.

See [ARCHITECTURE.html](../ARCHITECTURE.html) for what each of those stages is doing.
