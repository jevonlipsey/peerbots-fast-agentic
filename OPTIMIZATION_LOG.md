# Optimization Log

## Why this file was reset

The previous log recorded 16 iterations that moved "TTFA" from 887.12ms to
886.69ms — a net 0.43ms across sixteen commits of regex pre-compilation, cache-key
tuples and zero-allocation buffer reuse.

That number was not a measurement. `tests/test_harness.py` models the three
expensive stages as constants:

```python
DEFAULT_TTFT_DELAY_S  = 0.480   # awaited with asyncio.sleep
DEFAULT_TTS_DELAY_S   = 0.340   # awaited with asyncio.sleep
DEFAULT_DROID_DELAY_S = 0.030   # awaited with asyncio.sleep
```

480 + 340 + 30 = **850ms of the 887ms were sleeps.** The remaining ~37ms was
per-delta scheduling overhead in the event loop. Every one of those optimizations
was real work applied to a number that could not move, and the 886–894ms spread
across the log is `asyncio.sleep` jitter rather than signal.

The live console metric was misleading in a different way. `turn_start` was taken
*after* STT returned ([main.py](main.py)) and `first_audio_time` was set *before*
the audio was uploaded, so the figure it printed excluded the endpointing wait,
transcription, the upload and playback start — between them most of what a person
actually waits through.

The archived log is kept at `OPTIMIZATION_LOG.archive.md`.

## What replaced it

`scripts/turn_metrics.py` records the span that matters — **last speech frame to
audible sound** — broken into stages, one JSON object per turn in
`state/turns.jsonl`. Nothing is simulated. The old figure is still printed beside
it, labelled as legacy, so the two stay comparable.

## Measured baseline (2026-10-27)

macOS, Apple Silicon. `gpt-5.4-nano`, warm connection pool, phone mic + handset
speaker. Medians over small samples — directional, not a benchmark.

| Stage | Before | After | How |
| --- | --- | --- | --- |
| Endpoint wait | 1000ms fixed, **+1500ms** when the last word hit a continuation cue | **~250ms** | Smart Turn v3.2 decides; the 1.0s window survives only as a hard cap |
| STT (Parakeet TDT v3, CoreML) | unmeasured | **~90ms** | flat across 1.5s/3s/6s of audio |
| LLM TTFT | unmeasured, cache missed every turn | **~626ms** | stable prompt prefix now caches 1792 of 2249 tokens |
| First token → first clause | unmeasured | ~250–350ms | face fields stream first, then the first speakable clause |
| TTS first clause | ~300ms | **0ms on script** | 31 dialogue-tree clauses prewarmed at their live speeds |
| Dispatch + playback start | unmeasured | ~100ms | handset now reports when sound actually starts |
| **On-script total** | **~2.5–3.5s** (reconstructed) | **~1.3s** | |

## What the measurements overturned

**The endpointing timer was the whole ballgame.** At 1000ms it outweighed every
micro-optimization in the archived log by roughly 1000×. Semantic endpointing took
it to ~250ms, and that one change is worth more than the entire previous log.

**`reasoning_effort='none'` is already earning its keep.** Removing it costs
+400ms (626ms → 1027ms). It was the single most valuable line in the request
construction and it predates this pass. Keep it.

**Structured output is not a latency problem.** Dropping `response_format`
entirely changed TTFT by less than the noise (626ms → 698ms). The strict JSON
schema is free; what was expensive was the *field order* — with `speech` first,
emotion, colour and gaze only arrived after the whole utterance, so the face
changed after Lulo stopped talking and `EMOTION_SPEED_MAP` was dead code.

**Prompt size is not a latency problem either, now.** A 34-character system
prompt is no faster than the full 8660-character one (723ms vs 626ms), because the
prefix caches. Before this pass it could not: `_build_session_prefix()` prepended
`[session: turn N]` to the system message, so the leading bytes changed every turn
and ~3k tokens missed cache every time.

**A smarter model is nearly free.** `gpt-5.4-mini` costs +176ms over nano
(802ms vs 626ms); `gpt-5.4` costs +655ms (1281ms). For the half of the session
that is open Q&A rather than scripted tree, mini is the better trade — set
`OPENAI_MODEL` in `.env`.

## Where the remaining time is

Of the ~1.3s on-script total, TTFT (~626ms) and first-token-to-first-clause
(~250–350ms) are now **two thirds of the budget**. Nothing in the Python is worth
tuning against that.

The levers that remain are structural:

1. **Speech-to-speech realtime** (`gpt-realtime-2.1`) collapses STT + LLM + TTS
   into one socket with native barge-in and server-side turn detection. Reported
   250–600ms end to end. This is the only change that attacks TTFT directly, and
   it is a rewrite of the turn loop rather than an optimization.
2. **Speculative generation** — start the completion at the probe point and cancel
   if the user resumes. Hides endpointing almost entirely; worth ~250ms.
3. **Shorter first clause.** `MIN_FIRST_CHUNK_WORDS = 3` holds the first chunk
   back; 2 would start audio sooner at the cost of choppier delivery.

## Rule for future entries

Record the real span from `state/turns.jsonl`, say what sample it came from, and
state what the change was *supposed* to move. An optimization that cannot be
observed in that file did not happen.
