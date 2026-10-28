# next.md — pick up here

Handoff from the demo-hardening pass, 2026-10-27. Demo is Friday 2026-10-30.

Full reasoning and the original plan: `~/.claude/plans/we-present-lulo-on-drifting-sedgewick.md`

---

## Where things stand

Branch **`demo-hardening`**, 9 commits, **not merged, not pushed**, working tree clean.
25 files changed, +2932 / −265. Tests 9 → 55, all passing. `--preflight` all green.

```bash
git checkout demo-hardening
python -m pytest tests/ -q          # expect 55 passed
python main.py --preflight --phone-mic --no-arduino
```

Nothing is merged to `main`, so `main` is exactly what you demoed before. Merging is
your call — see **Rollback** at the bottom.

### Measured, not estimated

| Stage | Before | After |
| --- | --- | --- |
| Endpoint wait | 1000ms, **+1500ms** on common phrasings | **~250ms** |
| STT (Parakeet CoreML) | unmeasured | ~90ms, flat across 1.5–6s of audio |
| LLM TTFT | unmeasured, cache missed every turn | ~626ms, **1792/2249 tokens cached** |
| First token → first clause | unmeasured | ~250–350ms |
| TTS first clause | ~300ms | **0ms on script** (42/42 clauses cached) |
| **On-script total** | **~2.5–3.5s** reconstructed | **~1.3s** |

Every turn now appends a record to `state/turns.jsonl` (gitignored). `OPTIMIZATION_LOG.md`
has the full baseline; the old one is archived at `OPTIMIZATION_LOG.archive.md`.

**If a latency claim can't be seen in `turns.jsonl`, treat it as unproven.** The previous
harness simulated its three most expensive stages with `asyncio.sleep` — 850 of its
"887ms" were sleeps, which is why 16 logged optimization iterations moved it 0.43ms.

---

## ⚠️ Unverified — test these before Friday

I could not test any of this without the real room, the real phone, and a real person.
**This is the most important section in the file.** Everything here is reasoned and
compiles and passes tests, but has only been exercised against synthetic audio.

### 1. Smart Turn against real speech (highest risk)
I verified the model loads and runs (27ms median, 206ms one-time load), but I tested it
with **synthetic noise, not speech** — the probabilities it returned were not meaningful
for discrimination. The endpointing quality is genuinely unknown until a human talks to it.

```bash
python main.py --phone-mic --phone-audio
```
Watch for the `[[smart-turn]]: turn complete (p=…)` lines. What you want to see:
- Finishing a sentence → endpoints in ~250ms, `p` well above 0.5
- Pausing mid-sentence ("my knee is… uh…") → **no** endpoint, falls through to the 1.0s cap
- If it cuts you off mid-thought: raise `SMART_TURN_THRESHOLD` (0.5 → 0.6/0.7)
- If it feels sluggish: it's probably hitting the cap, so *lower* the threshold

Escape hatch if it misbehaves: `--no-smart-turn` restores the old fixed window exactly.

### 2. Echo cancellation on the actual handset
`peerbots_phone_mic/face.html` now asks for `echoCancellation: true` (noise suppression and
auto gain stay **off** — your original comment about Chrome's phone-call tuning was right
about those two). This is the single change that makes barge-in possible.

**The risk:** Android sometimes couples AEC to voice-communication mode, which can engage
some gain/gating anyway and clip word onsets. If transcription gets noticeably worse than
before, that's what happened. Fall back by setting it to `false` and accept no barge-in,
or build server-side AEC (we generate the TTS, so we have the exact reference signal — a
real but multi-day job).

### 3. Barge-in
Talk over Lulo mid-sentence. Expect audio to cut within ~250ms and
`-> [barge-in] user spoke, stopping audio`.
- False triggers (she cuts off when nobody spoke) → raise `BARGE_IN_THRESHOLD` (0.80 → 0.88)
- Won't trigger → lower it, or reduce `BARGE_IN_FRAMES` (7 ≈ 224ms sustained)
- On barge-in the mic flush is deliberately **skipped**, so your interrupting words carry
  into the next turn. If the next transcript looks doubled, that's the thing to look at.

### 4. Actuator travel — I guessed at your hardware
Gestures were 3.6mm and invisible; they're now 1.4–2.8cm. **But** that scaling assumes
`STROKE_MS = 5000` genuinely equals your full stroke (~10cm, so ~2cm/s). If your actuator
is faster or slower, the new durations are wrong in proportion.

Also new and untested on hardware:
- `REST_POS_MS = 1750` parks it ~35% extended at startup via a new `HOME` serial command.
  **This is why `slump` works at all now** — it only drives down, `drive(DOWN)` refuses from
  position 0, and 0 was the idle position, so Sad and Sleepy previously did nothing.
- All gestures are net-zero now (`startle` used to drift +6mm every time with no homing).

Flash `arduino/lulo_arm/lulo_arm.ino` and check each gesture is legible from ~3m, and that
`HOME` doesn't drive into the end stop.

### 5. Crowded-room noise floor
`listen_audio_silero` now measures an ambient VAD-probability floor and floats both
thresholds above it, because a fixed silence threshold of 0.45 can sit *below* a crowded
room's ambient level — which is how a turn could previously never end at all. Test with a
second speaker playing cafe noise at conversational volume.

### 6. Does the droid daemon implement `/stop`?
`DroidPlaybackQueue.interrupt()` POSTs to `{DROID_URL}/stop`. I never confirmed the daemon
supports it; failures are swallowed. Queue draining always works, so worst case the current
chunk plays out and the rest of the turn is dropped. **Irrelevant if you demo on the phone**
(the browser owns the audio nodes and cuts mid-word reliably).

---

## Demo-day runbook

### Already done on *this* machine (will not survive a fresh clone)
Both are gitignored build artifacts:
- `models/smart-turn-v3.2-int8.onnx` (11MB) — `python scripts/smart_turn.py --download`
- `stt-coreml/stt_worker/.build/.../stt_worker` (17MB) — `cd stt-coreml/stt_worker && swift build -c release` (~2 min)

**If you demo from a different laptop, do both first.** Without the Swift binary the first
launch runs `swift run -c release` and can sit there up to 300s compiling. `--preflight`
catches both.

### Launch
```bash
python main.py --preflight --phone-mic          # fix any FAIL; WARN is a quiet fallback
python main.py --phone-mic --phone-audio        # the demo config
```

### On stage
- **Sessions start fresh.** Previous conversation is archived to
  `state/history-<timestamp>.txt`, not carried over. `--keep-history` opts out.
  (Before this, Lulo inherited whatever the mic overheard in the lab — your
  `state/history.txt` had bench talk about 12V supplies and lunch plans in it.)
- **Phone page has a Listening/Muted toggle, bottom right.** Tap it if Lulo starts
  answering the audience instead of the patient. The level meter keeps running while
  muted so you can tell a muted mic from a broken one.
- You can interrupt her.
- Afterwards: read `state/turns.jsonl` rather than trusting a stopwatch.

### If something goes wrong mid-demo
| Symptom | Flag |
| --- | --- |
| Endpointing feels wrong | `--no-smart-turn` |
| Phone audio misbehaving | `--local-audio` (Mac speakers) or `--no-droid` (tablet TTS) |
| Body misbehaving | `--no-arduino` |
| Want continuity across runs | `--keep-history` |

---

## What changed, briefly

Eight commits, each independently revertable. `git log --oneline main..HEAD`.

**Phone acoustics** (`face.html`, `phone_mic_server.py`, `phone_playback.py`)
Echo cancellation back on; the page now reports `playback_started` / `playback_done`
instead of the server estimating from send time + clip duration (decode and wifi drift,
so the mic could reopen while she was still talking); mic queue raised from ~4.2s to ~20s
because the listen thread blocks on a synchronous STT call and a slow transcribe was
silently dropping the opening words of the next utterance; 48k→16k resample now
box-averaged instead of linearly interpolated with no low-pass.

**The demo-killers** (`whisper_stt.py`, `main.py`, `openai_response.py`, `lulo_arm.ino`)
- `okay`/`ok`/`yeah`/`yep`/`sure` were on the hallucination list, so four scripted user
  turns produced silence. Verified fixed: `"okay, call them"` → `"Notifying your PT right now."`
- Face ran 4–5s behind the voice — the only face update in dual-audio mode was the one
  carrying full speech text at end of turn, through a slow cloud path. Early update now
  fires in both modes. Verified: face precedes speech by 106–357ms every turn.
- `speech` was first in the reply schema, and strict mode emits schema order — so emotion,
  colour and gaze all arrived *after* the whole utterance. That's why the face lagged, the
  body gestured late, and `EMOTION_SPEED_MAP` was dead code.
- `[session: turn N]` was prepended to the system prompt, so ~3k tokens missed cache every
  turn. Now 1792/2249 cached.
- Phrase-limit check broke the inner loop before touching the silence counter while the
  outer exit only tested that counter → hard hang past the limit.
- Prewarm synthesized at `DEFAULT_SPEED` while live turns asked for the per-emotion speed,
  so nothing but the greeting ever hit the cache.
- `chunk_idx` restarted inside the tool loop, so a reply that both spoke and gestured
  replayed audio out of order. Your prompt actively encourages gestures.

**Endpointing + measurement** (`smart_turn.py`, `turn_metrics.py`) — new files.

**Barge-in** (`whisper_stt.watch_for_barge_in`, `main._play_or_be_interrupted`, `interrupt()`
on all three sinks).

**Preflight** (`preflight.py`) — 13 checks, blocking failures separated from quiet
fallbacks, because this pipeline degrades silently in half a dozen places.

**Incidental:** every lowercase `[[label]]` in the startup diagnostics was invisible —
`rich` only parses tags starting with a lowercase letter, so `[[stt]]` was consumed as a
style tag and printed as `[]`. 34 of them. `[[LULO]]` survived only because it's uppercase.

---

## Next moves, in the order I'd do them

### Before Friday
1. **Work the Unverified list above.** That's the whole job. Nothing else matters more.
2. **Decide the model.** `gpt-5.4-mini` is only **+176ms** over nano (802ms vs 626ms
   TTFT) and is meaningfully smarter. Half your session is open Q&A, and the live
   transcript shows nano drifting off-script and answering overheard chatter
   enthusiastically. One line: `OPENAI_MODEL` in `.env`. Full `gpt-5.4` is +655ms — too slow.
   Measure after changing: `state/turns.jsonl`.
3. **Rehearse the whole tree twice** with the real phone, tablet face and Arduino attached,
   with room noise present.

### Didn't get to (both were plan stretch items)
- **`turn_action` is still only `console.print`ed** — `main.py:556`. It's fully plumbed
  through schema, parser, dataclass and tests, and then nothing moves. Six lines of HRI
  gaze theory in your prompt the audience cannot perceive. One hobby servo maps the four
  values to real head orientation; your `Urdf_Animation.effective-motion.json` already
  authors `cap_yaw_joint` (0 → 2π) and `pitch_cap_joint` — the motion is designed, just
  never built. For an HRI audience this lands harder than latency does.
- **Vision rep counting.** `mcp-servers/vision_reader/` is not in `mcp_config.json`, and
  **couldn't work if it were**: `openai_response._run_tool` keeps only `.text` from tool
  results and silently discards `ImageContent`. Fix that first. Your prompt already says
  *"squatting only to 90 degrees"* and counts `"one… two… three…"` for reps it cannot see.

### After the demo
- **Speech-to-speech realtime** (`gpt-realtime-2.1`): collapses STT+LLM+TTS into one socket
  with native barge-in and server-side turn detection, reported 250–600ms end to end. It is
  the only change that attacks TTFT directly, which is now **two thirds of the remaining
  budget**. It's also a rewrite of the turn loop — 1–2 weeks, which is exactly why it isn't
  in this branch.
- **Speculative generation** — fire the completion at the probe point, cancel if the user
  resumes. Worth ~250ms.
- **`memory.jsonl` is 0 bytes.** Nothing has ever been persisted, the prompt never mentions
  the memory tools, and the patient's name survives only in a 20-message rolling history.
  Lulo never uses the name she just asked for.
- **"Notifying your PT right now" has no backend.** No notification, no log, no flag. For a
  PT study that's worth making real.
- **Idle presence.** A robot that's never frozen reads as alive: breathing on the actuator,
  blinks, a ~150ms non-verbal acknowledgement the instant speech ends. Perceptually that
  buys more than 500ms of real latency reduction, because it removes dead air rather than
  shortening it.

---

## Tunables

All env vars, all with working defaults — nothing needs setting.

| Var | Default | What it does |
| --- | --- | --- |
| `OPENAI_MODEL` | `gpt-5.4-nano` | `gpt-5.4-mini` is +176ms and smarter |
| `SMART_TURN_THRESHOLD` | `0.5` | ↑ = more patient, ↓ = faster to answer |
| `SMART_TURN_PROBE_S` | `0.20` | silence before asking "are they done?" |
| `SMART_TURN_REPROBE_S` | `0.20` | gap between repeat asks |
| `VAD_SILENCE_DURATION_S` | `1.0` | **hard cap** now, not the decision (`--pause-threshold`) |
| `VAD_SPEECH_THRESHOLD` | `0.60` | onset floor; floats up over measured room noise |
| `VAD_MIN_SPEECH_FRAMES` | `4` | ~128ms sustained before a turn opens (was 1 frame = 32ms, so a cough did it) |
| `BARGE_IN_THRESHOLD` | `0.80` | deliberately above onset — a false cut is worse than none |
| `BARGE_IN_FRAMES` | `7` | ~224ms sustained |
| `LULO_ANTI_REPETITION` | off | opener-avoidance hints; **off on purpose**, they push the model off the verbatim tree lines |
| `SMART_TURN_MODEL` | `models/…onnx` | override the weights path |

New flags: `--preflight`, `--no-smart-turn`, `--keep-history`.

---

## Rollback

```bash
git checkout main                      # untouched, exactly as demoed before
git revert <sha>                       # or drop one commit; they're independent
python main.py --no-smart-turn         # keep everything, restore old endpointing
```

To merge when you're happy:
```bash
git checkout main && git merge --no-ff demo-hardening
```

Worth running `/code-review high` over the branch first — it's a big diff, and the barge-in
concurrency (a VAD thread reading the stream while playback runs) is the part I'd most want
a second pass on.
