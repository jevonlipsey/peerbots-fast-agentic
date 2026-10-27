'''
regression tests for the demo-hardening pass.

each test pins a failure that was live in the pipeline, so the comments say what
broke rather than what the code does.
'''

import asyncio
import json
import os
import sys

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
for _p in (ROOT_DIR, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import turn_metrics
from scripts.whisper_stt import _ends_with_continuation, is_hallucination


# ---------------------------------------------------------------- stt filtering

# every one of these is a user turn the study's dialogue tree scripts. they were
# all on the hallucination list, so lulo received '' and said nothing.
SCRIPTED_USER_TURNS = [
    'okay', 'ok', 'yeah', 'yep', 'sure', 'yes', 'no',
    'cool lets do it', 'thanks lulo!', 'okay, call them',
    "ok, i guess i'm ready", "i'm doing it!", 'i did it! thanks lulo',
]

GENUINE_JUNK = ['', '   ', '...', 'mm-hmm', 'mhm', 'uh', 'um', 'thanks for watching']


@pytest.mark.parametrize('text', SCRIPTED_USER_TURNS)
def test_scripted_user_turns_survive_the_filter(text):
    assert not is_hallucination(text), f'{text!r} is a scripted answer, not a hallucination'


@pytest.mark.parametrize('text', GENUINE_JUNK)
def test_caption_artefacts_still_filtered(text):
    assert is_hallucination(text), f'{text!r} should still be dropped'


# these all ended in a continuation cue, so each one bought a 1.5s dead wait
# before lulo would answer - and they are verbatim dialogue-tree lines.
@pytest.mark.parametrize('text', [
    'thank you', 'nice to meet you', 'my knee is throbbing',
    "i'm ready", 'i did it, thanks lulo', 'that was the last one',
])
def test_no_dead_wait_on_complete_sentences(text):
    assert not _ends_with_continuation(text)


@pytest.mark.parametrize('text', [
    'i was going to say and', 'hold on because', 'i think but', 'wait',
])
def test_real_mid_thought_still_waits(text):
    assert _ends_with_continuation(text)


# ------------------------------------------------------------------- the schema

def test_face_fields_stream_before_speech():
    '''
    speech used to be first in the schema. strict mode emits properties in schema
    order, so emotion, colour and gaze arrived only after the whole utterance had
    streamed: the face changed after lulo stopped talking, the body gestured late,
    and EMOTION_SPEED_MAP never applied anything but its initial Neutral.
    '''
    from scripts.openai_response import REPLY_SCHEMA

    order = list(REPLY_SCHEMA['schema']['properties'].keys())
    assert order.index('speech') == len(order) - 1, 'speech must stream last'
    for field in ('emotion', 'color', 'turn_action'):
        assert order.index(field) < order.index('speech')


def test_session_hint_cannot_break_prompt_caching():
    '''
    the hint used to lead with '[session: turn N]' and was prepended to the system
    prompt, so the leading bytes of every request differed and ~3k tokens of
    prompt plus history missed the cache on every turn.
    '''
    import scripts.openai_response as oresp

    oresp.reset_session_context()
    assert 'session: turn' not in oresp._build_session_prefix()
    assert oresp.get_session_context()['turn_count'] == 1


# -------------------------------------------------------- scripted tts prewarm

def test_scripted_clauses_match_the_live_parser():
    '''
    the tts cache is keyed on the clause text the streaming parser emits, so
    prewarming whole dialogue-tree lines would never be hit. the prewarm pushes
    each line through that same parser to recover its real boundaries.
    '''
    import main

    clauses = main._scripted_clauses()
    assert len(clauses) >= 19, 'every scripted line should contribute at least one clause'
    assert all(text.strip() for text, _ in clauses)
    # the speeds asked for must be the ones live turns will ask for
    speeds = {main.get_voice_speed_for_emotion(e, main.KOKORO_SPEED) for _, e in clauses}
    assert speeds, 'clauses must carry an emotion so the cached speed matches'
    assert main.get_voice_speed_for_emotion('Happy', main.KOKORO_SPEED) != main.KOKORO_SPEED, (
        'prewarm at DEFAULT_SPEED is exactly why the cache never hit'
    )


# ------------------------------------------------------------------- the metric

def test_turn_metrics_span_starts_at_speech_end(tmp_path, monkeypatch):
    '''
    the old metric started its clock after stt had returned and stopped before the
    upload, so it excluded the endpointing wait, transcription, the upload and
    playback start - most of what a person actually waits through.
    '''
    monkeypatch.setattr(turn_metrics, 'TURNS_FILE', str(tmp_path / 'turns.jsonl'))
    monkeypatch.setattr(turn_metrics, 'STATE_DIR', str(tmp_path))

    turn_metrics.note_capture(
        speech_end=100.0, endpoint_at=100.25, silence_window_s=0.25, noise_floor=0.11
    )
    turn_metrics.note_stt_done()
    cap = turn_metrics.take_capture()
    cap['stt_done'] = 100.40

    timer = turn_metrics.TurnTimer(capture=cap, model='test', audio_sink='phone')
    timer.turn_start = 100.40
    for stage, at in (
        ('llm_start', 100.41), ('first_token', 100.66), ('first_clause', 100.72),
        ('first_synth', 100.88), ('first_audio_sent', 100.91),
        ('playback_started', 100.99), ('turn_done', 103.0),
    ):
        timer.stamps[stage] = at

    record = timer.write()
    stages = record['stages_ms']
    assert stages['endpoint_wait'] == 250.0
    assert stages['stt'] == 150.0
    assert stages['llm_ttft'] == 250.0

    # the honest total must include the wait and the stt the legacy figure skipped
    assert record['totals_ms']['speech_end_to_audible'] == pytest.approx(990.0, abs=1)
    assert record['totals_ms']['legacy_ttfa'] == pytest.approx(510.0, abs=1)
    assert (
        record['totals_ms']['speech_end_to_audible']
        > record['totals_ms']['legacy_ttfa']
    ), 'the real span is strictly longer than what used to be reported'

    written = [json.loads(l) for l in open(record and str(tmp_path / 'turns.jsonl'))]
    assert len(written) == 1 and written[0]['audio_sink'] == 'phone'
    assert 'to audible' in turn_metrics.format_console(record)


def test_take_capture_is_single_use():
    '''a turn must not inherit the previous turn's endpoint stamps'''
    turn_metrics.note_capture(speech_end=1.0, endpoint_at=1.2, silence_window_s=0.2)
    assert turn_metrics.take_capture()
    assert turn_metrics.take_capture() == {}


# ------------------------------------------------------- handle_turn wiring

def test_handle_turn_records_a_metric_and_fires_the_face_early(monkeypatch, tmp_path):
    '''
    smoke test for the wiring: a streamed reply must update the face before any
    audio is enqueued, and must leave exactly one metrics record behind.
    '''
    import main

    events = [
        {'type': 'face', 'emotion': 'Concerned', 'color': 'Red'},
        {'type': 'sentence', 'text': 'Noting this for your therapist.', 'idx': 0},
        {'type': 'final', 'reply': {
            'speech': 'Noting this for your therapist.', 'emotion': 'Concerned',
            'color': 'Red', 'turn_action': 'lulo glances at therapist'}, 'metrics': {'ttft_ms': 210}},
    ]

    async def fake_stream(user_text, tools_list, tool_router):
        for ev in events:
            yield ev

    order = []

    class FakeQueue:
        def reset_turn(self):
            order.append('reset')

        async def enqueue(self, audio, chunk_idx=None):
            order.append(f'audio{chunk_idx}')

        async def wait_complete(self):
            order.append('wait')

    async def fake_synth(text, voice=None, speed=None):
        return b'RIFF....WAVEfake'

    monkeypatch.setattr(main, 'stream_peerbots_reply', fake_stream)
    monkeypatch.setattr(main, 'synthesize_speech', fake_synth)
    monkeypatch.setattr(main, 'fire_peerbots_update', lambda *a, **k: order.append('face'))
    monkeypatch.setattr(main, '_PLAYBACK_QUEUE', FakeQueue())
    monkeypatch.setattr(main, '_USE_DROID', True)
    monkeypatch.setattr(main, '_ARM', None)
    # main.py has scripts/ on sys.path and imports this bare, so 'scripts.turn_metrics'
    # is a different module object. patch the instance main actually calls into.
    monkeypatch.setattr(main.turn_metrics, 'TURNS_FILE', str(tmp_path / 'turns.jsonl'))
    monkeypatch.setattr(main.turn_metrics, 'STATE_DIR', str(tmp_path))

    asyncio.run(main.handle_turn('my knee is throbbing', [], {}))

    assert 'face' in order, 'the face must be updated during the stream'
    assert order.index('face') < order.index('audio0'), (
        'the face update has to precede audio, or it lands 4-5s behind the voice'
    )
    written = [json.loads(l) for l in open(tmp_path / 'turns.jsonl')]
    assert len(written) == 1
    assert written[0]['emotion'] == 'Concerned'
    assert written[0]['turn_action'] == 'lulo glances at therapist'


# ---------------------------------------------------------------- phone acoustics

def test_browser_asks_for_echo_cancellation():
    '''
    the handset is speaker and microphone at once. with echoCancellation off it
    hears its own tts, which is also what forced the half-duplex lockout that
    makes barge-in impossible. the noise gate and auto gain stay off on purpose.
    '''
    page = open(os.path.join(ROOT_DIR, 'peerbots_phone_mic', 'face.html'), encoding='utf-8').read()
    assert 'echoCancellation:true' in page.replace(' ', '')
    assert 'noiseSuppression:false' in page.replace(' ', '')
    assert 'autoGainControl:false' in page.replace(' ', '')


def test_browser_reports_playback_instead_of_being_guessed():
    '''playback end used to be estimated from send time plus clip duration'''
    page = open(os.path.join(ROOT_DIR, 'peerbots_phone_mic', 'face.html'), encoding='utf-8').read()
    assert "notify('playback_started')" in page
    assert "notify('playback_done')" in page
    assert 'stopSpeaking' in page, 'barge-in needs the page to cut its own audio'


def test_rich_labels_are_escaped():
    '''
    rich only parses tags that start with a lowercase letter, so '[[stt]]' was
    swallowed as a style tag and printed as '[]'. every lowercase subsystem label
    in the startup diagnostics was invisible.
    '''
    from rich.console import Console
    from io import StringIO

    buf = StringIO()
    Console(file=buf, force_terminal=False, width=200).print(
        '[dim white]\\[\\[stt]]: visible[/]'
    )
    assert '[[stt]]: visible' in buf.getvalue()


# -------------------------------------------------------------------- barge-in

def test_barge_in_cuts_playback_and_keeps_the_audio(monkeypatch, tmp_path):
    '''
    the pipeline awaited the playback queue with nothing reading the microphone,
    so the patient could not interrupt at all, and the one-shot flush afterwards
    discarded anything that arrived late. a barge-in must stop the audio and must
    NOT flush, or it throws away the words that caused it.
    '''
    import main

    cut = {'interrupted': False}

    class FakeQueue:
        def reset_turn(self):
            pass

        async def enqueue(self, audio, chunk_idx=None):
            pass

        async def wait_complete(self):
            # stands in for a long reply still playing out
            await asyncio.sleep(5)

        async def interrupt(self):
            cut['interrupted'] = True

    async def fake_stream(user_text, tools_list, tool_router):
        yield {'type': 'face', 'emotion': 'Happy', 'color': 'Blue'}
        yield {'type': 'sentence', 'text': 'One, two, three, keep that back straight!', 'idx': 0}
        yield {'type': 'final', 'reply': {
            'speech': 'One, two, three, keep that back straight!', 'emotion': 'Happy',
            'color': 'Blue', 'turn_action': 'lulo maintains gaze'}, 'metrics': {}}

    monkeypatch.setattr(main, 'stream_peerbots_reply', fake_stream)
    monkeypatch.setattr(main, 'synthesize_speech', lambda *a, **k: _done(b'RIFFfake'))
    monkeypatch.setattr(main, 'fire_peerbots_update', lambda *a, **k: None)
    monkeypatch.setattr(main, '_PLAYBACK_QUEUE', FakeQueue())
    monkeypatch.setattr(main, '_USE_DROID', True)
    monkeypatch.setattr(main, '_ARM', None)
    # the watcher reports the user talking over her
    monkeypatch.setattr(main, 'watch_for_barge_in', lambda source, stop, **k: True)
    monkeypatch.setattr(main.turn_metrics, 'TURNS_FILE', str(tmp_path / 'turns.jsonl'))
    monkeypatch.setattr(main.turn_metrics, 'STATE_DIR', str(tmp_path))

    was_interrupted = asyncio.run(
        main.handle_turn('wait stop', [], {}, mic_source=object())
    )

    assert was_interrupted is True, 'handle_turn must tell the loop not to flush'
    assert cut['interrupted'], 'playback must actually be cut'
    record = json.loads(open(tmp_path / 'turns.jsonl').read().strip())
    assert record['interrupted'] is True


def test_no_barge_in_without_a_source(monkeypatch, tmp_path):
    '''text mode and the tablet sink have no stream to watch, and must still work'''
    import main

    class FakeQueue:
        def reset_turn(self):
            pass

        async def enqueue(self, audio, chunk_idx=None):
            pass

        async def wait_complete(self):
            pass

        async def interrupt(self):
            raise AssertionError('must not interrupt when there is nothing listening')

    async def fake_stream(user_text, tools_list, tool_router):
        yield {'type': 'face', 'emotion': 'Happy', 'color': 'Green'}
        yield {'type': 'sentence', 'text': "Let's go!", 'idx': 0}
        yield {'type': 'final', 'reply': {
            'speech': "Let's go!", 'emotion': 'Happy', 'color': 'Green',
            'turn_action': 'lulo turns to user'}, 'metrics': {}}

    monkeypatch.setattr(main, 'stream_peerbots_reply', fake_stream)
    monkeypatch.setattr(main, 'synthesize_speech', lambda *a, **k: _done(b'RIFFfake'))
    monkeypatch.setattr(main, 'fire_peerbots_update', lambda *a, **k: None)
    monkeypatch.setattr(main, '_PLAYBACK_QUEUE', FakeQueue())
    monkeypatch.setattr(main, '_USE_DROID', True)
    monkeypatch.setattr(main, '_ARM', None)
    monkeypatch.setattr(main.turn_metrics, 'TURNS_FILE', str(tmp_path / 'turns.jsonl'))
    monkeypatch.setattr(main.turn_metrics, 'STATE_DIR', str(tmp_path))

    assert asyncio.run(main.handle_turn('hello', [], {}, mic_source=None)) is False


def _done(value):
    '''a already-completed awaitable, for stubbing async helpers'''
    fut = asyncio.Future()
    fut.set_result(value)
    return fut


def test_every_sink_can_be_interrupted():
    '''
    barge-in is only honest if the sink in use can actually stop. the phone sink
    is the demo path and is the strongest case, because the browser owns the audio
    nodes and can cut a clip mid-word.
    '''
    from scripts.droid_client import DroidPlaybackQueue
    from scripts.local_playback import LocalPlaybackQueue
    from scripts.phone_playback import PhoneSpeakerQueue

    for cls in (DroidPlaybackQueue, LocalPlaybackQueue, PhoneSpeakerQueue):
        assert hasattr(cls, 'interrupt'), f'{cls.__name__} cannot be interrupted'
        assert asyncio.iscoroutinefunction(cls.interrupt)


# ------------------------------------------------------------- smart endpointing

def test_probe_window_is_far_below_the_cap():
    '''
    the fixed 1.0s window was the largest single item in the latency budget. it
    survives only as a hard cap; the probe is what normally decides.
    '''
    from scripts import whisper_stt

    assert whisper_stt.SMART_TURN_PROBE_S <= 0.3
    assert whisper_stt.SMART_TURN_PROBE_S < whisper_stt.DEFAULT_VAD_SILENCE_S
    # one 32ms frame used to be enough to open a turn, so a cough did it
    assert whisper_stt.MIN_SPEECH_FRAMES >= 3


def test_barge_in_bar_is_higher_than_onset():
    '''a false barge-in cuts lulo off mid-word, which is worse than none at all'''
    from scripts import whisper_stt

    assert whisper_stt.BARGE_IN_THRESHOLD > whisper_stt.DEFAULT_VAD_SPEECH_THRESHOLD
    assert whisper_stt.BARGE_IN_FRAMES >= whisper_stt.MIN_SPEECH_FRAMES


def test_smart_turn_contract():
    '''8s window at 16kHz, speech flush to the end, >0.5 means complete'''
    from scripts import smart_turn

    assert smart_turn.WINDOW_SAMPLES == 128000
    assert smart_turn.SAMPLE_RATE == 16000
    assert smart_turn.TURN_COMPLETE_THRESHOLD == pytest.approx(0.5)
    # absent weights must degrade to the fixed window, never raise
    if not smart_turn.model_available():
        assert smart_turn.turn_complete_probability([0.0] * 1000) is None
