"""
automated benchmark and regression test harness for lulo voice agent.
runs synthetic latency turns, yap monologue stress tests, and naturalness checks.
outputs TTFA_MS: <float> on final line for overnight optimization loops.
"""

import asyncio
import os
import re
import struct
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

### path setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from scripts.droid_client import DroidPlaybackQueue, get_audio_duration_s
from scripts.openai_response import (
    StreamingReplyParser,
    _build_session_prefix,
    _record_turn_metadata,
    get_session_context,
    parse_reply,
    reset_session_context,
)
from scripts.peerbots_client import DEFAULT_COLOR, asend_peerbots_message
from scripts.whisper_stt import _ends_with_continuation
from main import EMOTION_SPEED_MAP, get_voice_speed_for_emotion, make_nod_callback

### magic constants for synthetic benchmark
DEFAULT_STT_DELAY_S = 0.150  # 150ms silero-vad endpointing + inference
DEFAULT_TTFT_DELAY_S = 0.480  # 480ms gpt-4o-mini ttft on warm keepalive
DEFAULT_TTS_DELAY_S = 0.340  # 340ms kokoro-cpu wav synthesis
DEFAULT_DROID_DELAY_S = 0.030  # 30ms tailscale http post acknowledgment
FAST_MODE_SCALE = 1.0  # timing scale (1.0 = realistic benchmark, 0.1 = sub-second quick test)


def make_dummy_wav(duration_s=1.0, sample_rate=24000):
    """
    generates in-memory wav bytes for mocking kokoro synthesis.

    inputs:
    duration_s: float seconds
    sample_rate: sample rate in hz
    outputs:
    raw bytes with valid wav header
    """
    num_samples = int(sample_rate * duration_s)
    byte_rate = sample_rate * 2  # 16-bit mono = 2 bytes/sample
    data_size = num_samples * 2
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,  # pcm
        1,  # mono
        sample_rate,
        byte_rate,
        2,  # block align
        16,  # bits per sample
        b'data',
        data_size,
    )
    return header + b'\x00' * data_size


def count_speech_sentences(speech_text):
    """
    counts distinct conversational sentences in assistant speech.

    inputs:
    speech_text: raw assistant speech string
    outputs:
    integer sentence count
    """
    if not speech_text:
        return 0
    parts = [s.strip() for s in re.split(r'[.!?]+', speech_text) if s.strip()]
    return len(parts)


### turn runner and timing instrumentation
async def run_mocked_turn(user_text, assistant_payload, sim_params=None):
    """
    simulates one end-to-end conversational turn through the pipeline.
    measures all micro-spans from speech end to audio dispatch.

    inputs:
    user_text: string transcribed user input
    assistant_payload: dict containing speech, emotion, color
    sim_params: dict of simulated latencies in seconds (stt_s, ttft_s, tts_s, droid_s)
    outputs:
    metrics dict with individual span timings in ms and ttfa
    """
    sim = sim_params or {}
    scale = sim.get('scale', FAST_MODE_SCALE)
    stt_delay = sim.get('stt_s', DEFAULT_STT_DELAY_S) * scale
    ttft_delay = sim.get('ttft_s', DEFAULT_TTFT_DELAY_S) * scale
    tts_delay = sim.get('tts_s', DEFAULT_TTS_DELAY_S) * scale
    droid_delay = sim.get('droid_s', DEFAULT_DROID_DELAY_S) * scale

    # 1. user stops speaking -> stt completion
    t_speech_end = time.perf_counter()
    if stt_delay > 0:
        await asyncio.sleep(stt_delay)
    t_stt_done = time.perf_counter()
    stt_ms = (t_stt_done - t_speech_end) * 1000.0

    # 2. turn start (matches main.py handle_turn start)
    t_turn_start = t_stt_done

    # speculative face update: fires unawaited neutral/white at turn start
    face_dispatches = []

    async def mock_face_sender(text, emotion, color, silent=True):
        face_dispatches.append({'time': time.perf_counter(), 'emotion': emotion, 'color': color, 'silent': silent})
        return True

    asyncio.create_task(mock_face_sender('', 'Neutral', DEFAULT_COLOR, silent=True))

    # 3. token stream start -> first clause emitted
    t_api_start = time.perf_counter()
    if ttft_delay > 0:
        await asyncio.sleep(ttft_delay)
    t_first_token = time.perf_counter()
    ttft_ms = (t_first_token - t_api_start) * 1000.0

    parser = StreamingReplyParser()
    chunk_order_events = {0: asyncio.Event()}
    chunk_order_events[0].set()

    tts_tasks = []
    spoken_chunks = []
    first_clause_time = None
    first_audio_time = None
    tts1_ms = 0.0
    face_time = None
    face_updated = False
    droid_dispatch_ms = 0.0

    # mock droid queue
    mock_queue = DroidPlaybackQueue()
    mock_queue.reset_turn()

    # prepare token deltas matching assistant_payload json
    speech = assistant_payload.get('speech', '')
    emotion = assistant_payload.get('emotion', 'Neutral')
    color = assistant_payload.get('color', DEFAULT_COLOR)

    # break into small token chunks for streaming simulation
    token_deltas = [
        '{"speech": "',
    ]
    # split speech into words / punctuation chunks
    words = speech.split(' ')
    chunk_accum = ''
    for i, w in enumerate(words):
        chunk_accum += w + (' ' if i < len(words) - 1 else '')
        if len(chunk_accum) >= 8 or i == len(words) - 1:
            token_deltas.append(chunk_accum)
            chunk_accum = ''
    if chunk_accum:
        token_deltas.append(chunk_accum)

    token_deltas.append(f'", "emotion": "{emotion}", ')
    token_deltas.append(f'"color": "{color}"}}')

    # feed token stream into parser
    chunk_counter = 0
    for delta in token_deltas:
        # simulate small inter-token interval
        if scale > 0:
            await asyncio.sleep(0.005 * scale)

        sentences = parser.feed(delta)

        # early face detection
        if not face_updated and parser.emotion and parser.color:
            face_updated = True
            face_time = time.perf_counter()
            asyncio.create_task(mock_face_sender('', parser.emotion, parser.color, silent=True))

        for sent in sentences:
            current_idx = chunk_counter
            chunk_counter += 1
            spoken_chunks.append(sent)

            if current_idx == 0:
                first_clause_time = time.perf_counter()

            async def _synth_and_enqueue(sent_text, idx):
                nonlocal first_audio_time, tts1_ms, droid_dispatch_ms
                s_start = time.perf_counter()
                if tts_delay > 0:
                    await asyncio.sleep(tts_delay)
                wav_bytes = make_dummy_wav(duration_s=max(0.6, len(sent_text.split()) * 0.25))
                s_done = time.perf_counter()

                if idx == 0:
                    tts1_ms = (s_done - s_start) * 1000.0

                if idx not in chunk_order_events:
                    chunk_order_events[idx] = asyncio.Event()

                try:
                    await chunk_order_events[idx].wait()
                    d_start = time.perf_counter()
                    if droid_delay > 0:
                        await asyncio.sleep(droid_delay)
                    d_done = time.perf_counter()

                    if idx == 0:
                        first_audio_time = d_done
                        droid_dispatch_ms = (d_done - d_start) * 1000.0

                    await mock_queue.enqueue(wav_bytes, chunk_idx=idx)
                finally:
                    next_idx = idx + 1
                    if next_idx not in chunk_order_events:
                        chunk_order_events[next_idx] = asyncio.Event()
                    chunk_order_events[next_idx].set()

            task = asyncio.create_task(_synth_and_enqueue(sent, current_idx))
            tts_tasks.append(task)

    # finalize leftover clauses
    leftovers = parser.finish()
    for sent in leftovers:
        current_idx = chunk_counter
        chunk_counter += 1
        spoken_chunks.append(sent)
        if current_idx == 0:
            first_clause_time = time.perf_counter()

        async def _synth_leftover(sent_text, idx):
            nonlocal first_audio_time, tts1_ms, droid_dispatch_ms
            s_start = time.perf_counter()
            if tts_delay > 0:
                await asyncio.sleep(tts_delay)
            wav_bytes = make_dummy_wav(duration_s=max(0.6, len(sent_text.split()) * 0.25))
            s_done = time.perf_counter()
            if idx == 0:
                tts1_ms = (s_done - s_start) * 1000.0
            if idx not in chunk_order_events:
                chunk_order_events[idx] = asyncio.Event()
            try:
                await chunk_order_events[idx].wait()
                d_start = time.perf_counter()
                if droid_delay > 0:
                    await asyncio.sleep(droid_delay)
                d_done = time.perf_counter()
                if idx == 0:
                    first_audio_time = d_done
                    droid_dispatch_ms = (d_done - d_start) * 1000.0
                await mock_queue.enqueue(wav_bytes, chunk_idx=idx)
            finally:
                next_idx = idx + 1
                if next_idx not in chunk_order_events:
                    chunk_order_events[next_idx] = asyncio.Event()
                chunk_order_events[next_idx].set()

        task = asyncio.create_task(_synth_leftover(sent, current_idx))
        tts_tasks.append(task)

    if tts_tasks:
        await asyncio.gather(*tts_tasks)

    # record turn metadata for opener anti-repetition & affect tracking
    _record_turn_metadata(speech, emotion)

    # span calculations
    clause1_ms = ((first_clause_time - t_turn_start) * 1000.0) if first_clause_time else 0.0
    face_ms = ((face_time - t_api_start) * 1000.0) if face_time else 0.0
    ttfa_turn_start_ms = ((first_audio_time - t_turn_start) * 1000.0) if first_audio_time else 0.0
    ttfa_speech_end_ms = ((first_audio_time - t_speech_end) * 1000.0) if first_audio_time else 0.0
    total_turn_ms = (time.perf_counter() - t_turn_start) * 1000.0

    return {
        'stt_ms': round(stt_ms, 2),
        'ttft_ms': round(ttft_ms, 2),
        'clause1_ms': round(clause1_ms, 2),
        'tts1_ms': round(tts1_ms, 2),
        'face_ms': round(face_ms, 2),
        'droid_ms': round(droid_dispatch_ms, 2),
        'ttfa_turn_start_ms': round(ttfa_turn_start_ms, 2),
        'ttfa_speech_end_ms': round(ttfa_speech_end_ms, 2),
        'total_turn_ms': round(total_turn_ms, 2),
        'emotion': emotion,
        'color': color,
        'spoken_chunks': spoken_chunks,
        'face_dispatches': face_dispatches,
    }


### benchmark 1: synthetic latency turns
async def benchmark_synthetic_latency(sim_params=None):
    """
    runs 3 end-to-end mocked turns simulating:
    user stops speaking -> stt completion
    token stream start -> first clause emitted
    kokoro synthesis -> audio bytes dispatched
    logs individual spans and returns average ttfa.

    inputs:
    sim_params: optional dict overriding simulation timing
    outputs:
    list of metrics dicts, average ttfa in ms
    """
    reset_session_context()

    turns_data = [
        (
            "I'm ready.",
            {
                'speech': "Great! Let's get started and have some fun—you got this!",
                'emotion': 'Happy',
                'color': 'Green',
            },
        ),
        (
            'Not feeling so good though.',
            {
                'speech': "I'm sorry to hear that, take it easy and rest if you need to. I'm here to help whenever you're ready!",
                'emotion': 'Concerned',
                'color': 'Orange',
            },
        ),
        (
            "Awesome, let's keep going.",
            {
                'speech': "Awesome! Glad you're feeling better. Let's keep going whenever you're ready!",
                'emotion': 'Happy',
                'color': 'Green',
            },
        ),
    ]

    print('\n--- 1. Synthetic Latency Test (3 Mocked Turns) ---')
    results = []

    for idx, (user_text, assistant_payload) in enumerate(turns_data, 1):
        m = await run_mocked_turn(user_text, assistant_payload, sim_params=sim_params)
        results.append(m)

        print(f'[Turn {idx}: "{user_text}"]')
        print(
            f'  -> Spans: STT: {m["stt_ms"]}ms | TTFT: {m["ttft_ms"]}ms | '
            f'Clause1: {m["clause1_ms"]}ms | TTS1: {m["tts1_ms"]}ms | '
            f'Face: {m["face_ms"]}ms | Droid: {m["droid_ms"]}ms'
        )
        print(
            f'  -> TTFA (from speech end): {m["ttfa_speech_end_ms"]}ms | '
            f'TTFA (turn start): {m["ttfa_turn_start_ms"]}ms | '
            f'Emotion: {m["emotion"]} / {m["color"]}'
        )

        # assertions per turn
        assert m['clause1_ms'] > 0, f'Turn {idx}: clause 1 span must be > 0'
        assert m['tts1_ms'] > 0, f'Turn {idx}: tts 1 span must be > 0'
        assert m['ttfa_turn_start_ms'] > 0, f'Turn {idx}: ttfa must be > 0'
        assert len(m['spoken_chunks']) >= 1, f'Turn {idx}: expected spoken chunks'

    avg_ttfa = sum(m['ttfa_turn_start_ms'] for m in results) / len(results)
    avg_speech_end_ttfa = sum(m['ttfa_speech_end_ms'] for m in results) / len(results)

    print(f'  Average TTFA (turn start): {avg_ttfa:.2f}ms')
    print(f'  Average TTFA (speech end): {avg_speech_end_ttfa:.2f}ms')

    return results, avg_ttfa


### benchmark 2: yap monologue & flow stress test
async def benchmark_yap_and_flow_stress():
    """
    simulates 45-second user transcription with trailing pauses.
    asserts silent face nods fire without blocking.
    asserts reply length remains between 2 and 4 sentences.

    inputs:
    none
    outputs:
    boolean success
    """
    print('\n--- 2. Yap & Flow Stress Test (45s Monologue) ---')

    # 1. 45-second user monologue text with trailing hesitation pause
    yap_text = (
        "I wanna get into my backstory. It's uh Pretty intense. Like You know how "
        "when you're doing physical therapy and everything hurts so much that you just "
        "want to quit? But then my trainer told me that pushing through the gentle stretches "
        "is what actually builds strength back up, so I was thinking maybe we should try..."
    )

    # assert continuation cue detection on trailing hesitation
    assert _ends_with_continuation(yap_text), 'yap monologue trailing ellipsis must trigger continuation'
    assert _ends_with_continuation('Well, I was thinking that maybe uh'), 'trailing uh must trigger continuation'
    assert not _ends_with_continuation('I am completely ready.'), 'clean terminal period must not trigger continuation'
    print('  -> Verified continuation cue detection for trailing pauses: PASS')

    # 2. silent face nod triggers at 10s, 25s, 45s without blocking stream
    loop = asyncio.get_running_loop()
    nod_cb = make_nod_callback(loop)

    mock_face_calls = []

    async def mock_nod_sender(speech, emotion, color, silent=True):
        mock_face_calls.append({'emotion': emotion, 'color': color, 'silent': silent})
        return True

    # patch asend_peerbots_message to intercept nod dispatches
    with patch('main.asend_peerbots_message', side_effect=mock_nod_sender):
        t0 = time.perf_counter()
        nod_cb(10)
        nod_cb(25)
        nod_cb(45)
        cb_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # callback must return immediately (<5ms) without blocking
        assert cb_elapsed_ms < 15.0, f'nod_callback blocked for {cb_elapsed_ms:.2f}ms'

        # allow event loop to dispatch scheduled tasks
        await asyncio.sleep(0.05)

        # verify all 3 nod marks fired with silent=True and white skin
        assert len(mock_face_calls) == 3, f'expected 3 nod face updates, got {len(mock_face_calls)}'
        assert mock_face_calls[0] == {'emotion': 'Happy', 'color': 'White', 'silent': True}
        assert mock_face_calls[1] == {'emotion': 'Neutral', 'color': 'White', 'silent': True}
        assert mock_face_calls[2] == {'emotion': 'Surprised', 'color': 'White', 'silent': True}

    print('  -> Testing silent listening face nods (10s, 25s, 45s)...')
    print('     [Nod 10s]: Happy / White (silent=True) - non-blocking: PASS')
    print('     [Nod 25s]: Neutral / White (silent=True) - non-blocking: PASS')
    print('     [Nod 45s]: Surprised / White (silent=True) - non-blocking: PASS')

    # 3. response sentence count constraint (2 to 4 sentences for monologues)
    simulated_yap_replies = [
        # standard warm 3-sentence reply
        (
            'Wow, that sounds really powerful. I\'m here to listen and support you—feel '
            'free to share whenever you\'re ready. We can take all the time you need.',
            3,
        ),
        # 2-sentence empathetic reply
        (
            'That makes total sense, and thank you for sharing that with me. '
            'Whenever you\'re ready, we can take it one gentle step at a time.',
            2,
        ),
        # 4-sentence structured reply
        (
            'Thank you for opening up to me about your journey. Dealing with that kind of pain '
            'takes serious emotional strength. We will customize every single stretch today to '
            'keep you comfortable. Just let me know whenever you want to begin.',
            4,
        ),
    ]

    for reply_speech, expected_count in simulated_yap_replies:
        s_count = count_speech_sentences(reply_speech)
        assert s_count == expected_count, f'sentence count mismatch: expected {expected_count}, got {s_count}'
        assert 2 <= s_count <= 4, f'yap reply must have 2-4 sentences, got {s_count}'

    # negative checks: 1-sentence and 5-sentence replies
    assert not (2 <= count_speech_sentences('Okay, let us start.') <= 4), '1 sentence should fail yap length bounds'
    long_yap = 'Sentence one. Sentence two. Sentence three. Sentence four. Sentence five.'
    assert not (2 <= count_speech_sentences(long_yap) <= 4), '5 sentences should fail yap length bounds'

    print('  -> Testing response sentence length for extended monologue...')
    print(f'     Sample Yap Reply: "{simulated_yap_replies[0][0]}"')
    print('     Sentence length checks (2-4 sentences): PASS')
    return True


### benchmark 3: naturalness & flow checks
async def benchmark_naturalness_and_flow():
    """
    tests anti-repetition, dynamic voice speed scaling, and starvation prevention.

    inputs:
    none
    outputs:
    boolean success
    """
    print('\n--- 3. Naturalness & Conversational Flow Assertions ---')

    # 1. session context and opener anti-repetition
    reset_session_context()
    _record_turn_metadata("Great! Let's get started", 'Happy')
    _record_turn_metadata('Great! You got this', 'Happy')
    _record_turn_metadata('Great! Keep going', 'Happy')

    prefix = _build_session_prefix()
    assert '[session: turn 1]' in prefix
    assert "Avoid recent openers: 'Great! Let's get', 'Great! You got', 'Great! Keep going'" in prefix
    assert "Emotion dampening: you used 'Happy' for 3 turns in a row" in prefix
    print('  -> Session context opener anti-repetition & dampening: PASS')

    # 2. emotion voice speed scaling
    assert get_voice_speed_for_emotion('Happy', 1.0) == 1.12
    assert get_voice_speed_for_emotion('Surprised', 1.0) == 1.10
    assert get_voice_speed_for_emotion('Neutral', 1.0) == 1.05
    assert get_voice_speed_for_emotion('Concerned', 1.0) == 0.92
    assert get_voice_speed_for_emotion('Sad', 1.0) == 0.90
    assert get_voice_speed_for_emotion('Sleepy', 1.0) == 0.88
    print('  -> Emotion-based dynamic Kokoro voice speed: PASS')

    # 3. buffer starvation prevention
    parser = StreamingReplyParser()
    # feeding 1-word opener followed by clause
    c1 = parser.feed('{"speech": "Awesome! ')
    assert len(c1) == 0, 'parser must not isolate 1-word Awesome! to prevent audio starvation'
    c2 = parser.feed("Let's jump into it—")
    assert len(c2) == 1, 'parser must emit chunk 0 once accumulated speech buffer >= 3 words'
    assert c2[0] == "Awesome! Let's jump into it—"
    print('  -> Buffer starvation prevention (no isolated 1-word chunks): PASS')

    # 4. wav byte duration parsing
    wav_1s = make_dummy_wav(1.0)
    dur = get_audio_duration_s(wav_1s)
    assert abs(dur - 1.0) < 0.05, f'wav duration parse error: expected ~1.0s, got {dur}'
    print('  -> In-memory WAV byte duration parsing: PASS')

    # 5. dual audio lip-sync mode
    from main import PEERBOTS_DUAL_AUDIO
    assert PEERBOTS_DUAL_AUDIO is True, 'dual audio should be enabled by default for tablet lip-sync'
    print('  -> Dual audio tablet lip-sync enabled by default: PASS')

    return True


### benchmark 4: dialogue tree and extended pause assertions
async def benchmark_dialogue_tree_and_pausing():
    '''
    validates dialogue tree prompt configuration and extended speech pausing.

    inputs:
    none
    outputs:
    boolean success
    '''
    print('\n--- 4. Dialogue Tree & Extended Pausing Assertions ---')
    from main import PAUSE_THRESHOLD, GREETING_SPEECH
    from scripts.whisper_stt import DEFAULT_VAD_SILENCE_S, _ends_with_continuation, listen_once
    from scripts.openai_response import load_system_prompt
    import inspect

    # 1. pause threshold extended to 2.0s
    assert PAUSE_THRESHOLD >= 2.0, f'pause threshold should be >= 2.0s, got {PAUSE_THRESHOLD}'
    assert DEFAULT_VAD_SILENCE_S >= 2.0, f'vad silence should be >= 2.0s, got {DEFAULT_VAD_SILENCE_S}'
    print(f'  -> Extended pause threshold ({PAUSE_THRESHOLD}s default): PASS')

    # 2. listen_once signature accepts pause_threshold
    sig = inspect.signature(listen_once)
    assert 'pause_threshold' in sig.parameters, 'listen_once must accept pause_threshold'
    print('  -> listen_once pause_threshold parameter pass-through: PASS')

    # 3. extended continuation cues for dialogue tree reading
    assert _ends_with_continuation('Yeah, how low should I...'), 'trailing ellipsis must trigger continuation'
    assert _ends_with_continuation('take a break and'), 'trailing conjunction must trigger continuation'
    assert _ends_with_continuation('wait'), 'hesitation word wait must trigger continuation'
    assert _ends_with_continuation('well,'), 'trailing comma hesitation must trigger continuation'
    assert not _ends_with_continuation('thanks lulo!'), 'completed exclamation must not trigger continuation'
    print('  -> Dialogue reading continuation cues: PASS')

    # 4. greeting matches study dialogue tree
    assert GREETING_SPEECH == "Hi! I'm Lulo. Nice to meet you! What's your name?", f'unexpected greeting: {GREETING_SPEECH}'
    print('  -> Startup greeting matches study dialogue tree: PASS')

    # 5. system prompt contains dialogue tree nodes
    prompt = load_system_prompt()
    assert 'squats' in prompt.lower(), 'prompt must specify squats exercise'
    assert 'curious deer' in prompt.lower(), 'prompt must contain deer backstory'
    assert '90 degrees' in prompt.lower(), 'prompt must contain 90 degrees squat info'
    assert 'throbbing' in prompt.lower(), 'prompt must contain throbbing knee escalation'
    assert 'flag me when' in prompt.lower(), 'prompt must contain break recovery phrasing'
    print('  -> System prompt study dialogue tree coverage: PASS')

    return True


### pytest hooks
def test_synthetic_latency():
    asyncio.run(benchmark_synthetic_latency({'scale': 0.1}))


def test_yap_and_flow_stress():
    asyncio.run(benchmark_yap_and_flow_stress())


def test_naturalness_and_flow():
    asyncio.run(benchmark_naturalness_and_flow())


def test_dialogue_tree_and_pausing():
    asyncio.run(benchmark_dialogue_tree_and_pausing())


### cli execution
async def amain():
    """
    main harness execution routine.
    runs benchmarks, logs spans, and prints TTFA_MS: <float> on final line.
    """
    print('======================================================================')
    print('LULO PIPELINE BENCHMARK & HARNESS')
    print('======================================================================')

    # parse optional cli flags
    scale = FAST_MODE_SCALE
    if '--fast' in sys.argv:
        scale = 0.1

    # run benchmarks
    _, avg_ttfa = await benchmark_synthetic_latency({'scale': scale})
    await benchmark_yap_and_flow_stress()
    await benchmark_naturalness_and_flow()
    await benchmark_dialogue_tree_and_pausing()

    print('\n======================================================================')
    print('ALL HARNESS ASSERTIONS PASSED (EXIT 0)')
    print('======================================================================')
    # print final line required by overnight optimization loops
    print(f'TTFA_MS: {avg_ttfa:.2f}')
    return 0


def main():
    try:
        code = asyncio.run(amain())
        sys.exit(code)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'\nHARNESS FAILURE: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
