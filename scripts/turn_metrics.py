'''
real end-to-end turn timing, from the user's last word to audible speech.

this exists because the number the project had been optimizing was not a
measurement. tests/test_harness.py models the llm, kokoro and the network hop as
asyncio.sleep(0.480), sleep(0.340) and sleep(0.030), so ~850ms of its ~887ms
"TTFA" was hard-coded sleeps and the measured remainder was event-loop jitter.
sixteen logged optimization iterations moved that figure by 0.43ms in total.

the live console metric was misleading in a different way: its clock started
after stt had already returned and stopped before the audio was uploaded, so it
excluded the endpointing wait, transcription, the upload and playback start -
which between them are most of what a person actually waits through.

the span recorded here is the one that matters:

    last speech frame
      -> endpoint decision        (how long the vad waited to be sure)
      -> transcript ready         (stt)
      -> first llm token          (ttft)
      -> first clause parsed      (how much had to generate before anything speakable)
      -> first clause synthesized (tts)
      -> first audio sent         (dispatch)
      -> handset confirms audio   (playback actually starting)

one json object per turn is appended to state/turns.jsonl.
'''

import json
import os
import threading
import time

STATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'state'
)
TURNS_FILE = os.path.join(STATE_DIR, 'turns.jsonl')

# the capture thread records when speech stopped; the event loop reads it when the
# turn starts, so the handoff needs its own lock rather than a plain global
_capture_lock = threading.Lock()
_capture = {}


def note_capture(speech_end, endpoint_at, silence_window_s, noise_floor=None):
    '''
    called from the vad thread the moment an utterance is endpointed.

    inputs:
    speech_end: monotonic time of the last frame that still counted as speech
    endpoint_at: monotonic time the endpoint decision was actually made
    silence_window_s: silence the endpointer required before deciding
    noise_floor: measured ambient vad probability, when one was estimated
    outputs:
    none
    '''
    with _capture_lock:
        _capture.clear()
        _capture.update(
            {
                'speech_end': speech_end,
                'endpoint_at': endpoint_at,
                'silence_window_s': silence_window_s,
                'noise_floor': noise_floor,
            }
        )


def note_stt_done():
    '''record when the transcript came back'''
    with _capture_lock:
        if _capture:
            _capture['stt_done'] = time.monotonic()


def take_capture():
    '''
    hand the pending capture timings to the turn that is about to run.

    inputs:
    none
    outputs:
    dict of captured stamps, empty when the vad path did not record any
    '''
    with _capture_lock:
        snapshot = dict(_capture)
        _capture.clear()
    return snapshot


_playback_lock = threading.Lock()
_playback_started_at = None


def note_playback_started():
    '''
    called when the audio sink reports that sound actually began.

    inputs:
    none
    outputs:
    none
    '''
    global _playback_started_at
    with _playback_lock:
        _playback_started_at = time.monotonic()


def take_playback_started():
    '''pop the pending playback-start stamp, or None'''
    global _playback_started_at
    with _playback_lock:
        stamp = _playback_started_at
        _playback_started_at = None
    return stamp


def arm_playback():
    '''clear any stale playback-start stamp before a turn'''
    global _playback_started_at
    with _playback_lock:
        _playback_started_at = None


class TurnTimer:
    '''
    collects one turn's stage timings and writes a single jsonl record.

    all stamps are time.monotonic(). marks are idempotent: the first one for a
    given stage wins, so the "first" in first-clause and first-audio is honest
    even though the callers fire per chunk.
    '''

    def __init__(self, capture=None, backend=None, model=None, audio_sink=None):
        self.capture = capture or {}
        self.meta = {'backend': backend, 'model': model, 'audio_sink': audio_sink}
        self.turn_start = time.monotonic()
        self.stamps = {}
        self.extra = {}

    def mark(self, stage):
        '''record the first occurrence of a stage'''
        if stage not in self.stamps:
            self.stamps[stage] = time.monotonic()
        return self.stamps[stage]

    def set(self, key, value):
        self.extra[key] = value

    def _ms(self, a, b):
        if a is None or b is None:
            return None
        return round((b - a) * 1000.0, 1)

    def summary(self):
        '''
        build the per-stage breakdown.

        inputs:
        none
        outputs:
        dict with per-stage durations in ms plus the two totals that matter
        '''
        cap = self.capture
        speech_end = cap.get('speech_end')
        endpoint_at = cap.get('endpoint_at')
        stt_done = cap.get('stt_done')
        s = self.stamps
        first_audible = s.get('playback_started') or s.get('first_audio_sent')

        out = {
            'ts': round(time.time(), 3),
            'stages_ms': {
                # how long the endpointer held on after the user stopped talking
                'endpoint_wait': self._ms(speech_end, endpoint_at),
                'stt': self._ms(endpoint_at, stt_done),
                'llm_ttft': self._ms(s.get('llm_start'), s.get('first_token')),
                'llm_to_first_clause': self._ms(s.get('first_token'), s.get('first_clause')),
                'tts_first_clause': self._ms(s.get('first_clause'), s.get('first_synth')),
                'dispatch': self._ms(s.get('first_synth'), s.get('first_audio_sent')),
                'playback_start': self._ms(s.get('first_audio_sent'), s.get('playback_started')),
            },
            'totals_ms': {
                # the honest number: user stops talking -> lulo is audible
                'speech_end_to_audible': self._ms(speech_end, first_audible),
                # what the old console metric measured, for comparison
                'legacy_ttfa': self._ms(self.turn_start, s.get('first_audio_sent')),
                'turn_total': self._ms(self.turn_start, s.get('turn_done')),
            },
            'silence_window_s': cap.get('silence_window_s'),
            'noise_floor': cap.get('noise_floor'),
        }
        out.update({k: v for k, v in self.meta.items() if v is not None})
        out.update(self.extra)
        return out

    def write(self):
        '''append this turn's record to state/turns.jsonl, never raising'''
        record = self.summary()
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            with open(TURNS_FILE, 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(record) + '\n')
        except Exception:
            pass
        return record


def format_console(record):
    '''
    one-line human summary of a turn record.

    inputs:
    record: dict from TurnTimer.summary()
    outputs:
    display string
    '''
    st = record.get('stages_ms', {})
    to = record.get('totals_ms', {})
    parts = []
    for label, key in (
        ('wait', 'endpoint_wait'),
        ('stt', 'stt'),
        ('ttft', 'llm_ttft'),
        ('clause', 'llm_to_first_clause'),
        ('tts', 'tts_first_clause'),
        ('send', 'dispatch'),
        ('play', 'playback_start'),
    ):
        v = st.get(key)
        if v is not None:
            parts.append(f'{label} {v:.0f}')
    total = to.get('speech_end_to_audible')
    head = f'{total / 1000.0:.2f}s to audible' if total is not None else 'incomplete'
    return f'{head}  ({" | ".join(parts)})'
