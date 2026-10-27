'''
local speech-to-text. on mac uses the swift coreml worker,
else falls back to python whisper. exposes listen_once() for
main.py plus the original file-queue daemon mode.

layer 1: microphone -> whisper_stt -> openai_response
'''

import atexit
import collections
import os
import platform
import random
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time

import speech_recognition as sr

import smart_turn
import turn_metrics
from rich.console import Console

console = Console()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.file_utils import safe_read, safe_write


### config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(BASE_DIR, 'state')

LISTEN_FILE = os.path.join(STATE_DIR, 'listen.txt')
TRANSCRIPTION_FILE = os.path.join(STATE_DIR, 'transcription.txt')

### mic index now lives in main.py (--mic-index, see --list-mics)
### daemon mode defaults to the system microphone
MICROPHONE_INDEX = 7

DEFAULT_VAD_SILENCE_S = float(os.environ.get('VAD_SILENCE_DURATION_S', '1.0'))
DEFAULT_NON_SPEAKING_S = float(os.environ.get('VAD_NON_SPEAKING_S', '0.25'))
DEFAULT_VAD_SPEECH_THRESHOLD = float(os.environ.get('VAD_SPEECH_THRESHOLD', '0.60'))

# consecutive frames above threshold before an utterance is declared started. one
# 32ms frame was enough, so a cough, a clap or a chair scrape opened a whole turn
# and cost a silence window plus an stt round trip on junk.
MIN_SPEECH_FRAMES = int(os.environ.get('VAD_MIN_SPEECH_FRAMES', '4'))

# how many pre-speech frames to watch before trusting a measured noise floor, and
# the headroom to leave above it.
NOISE_FLOOR_MIN_FRAMES = 15
NOISE_FLOOR_SILENCE_MARGIN = 0.10
NOISE_FLOOR_SPEECH_MARGIN = 0.25

# with semantic endpointing the silence window stops being the decision and
# becomes the trigger to ask the question. a complete turn ends after this long;
# an incomplete one keeps waiting, up to the caller's window as a hard cap.
SMART_TURN_PROBE_S = float(os.environ.get('SMART_TURN_PROBE_S', '0.20'))
# don't re-run inference every frame while someone is mid-thought
SMART_TURN_REPROBE_S = float(os.environ.get('SMART_TURN_REPROBE_S', '0.20'))

IS_MAC = platform.system() == 'Darwin'
_SWIFT_TMP_WAV = os.path.join(tempfile.gettempdir(), f'lulo_stt_{os.getpid()}.wav')


def _cleanup_tmp_wav():
    try:
        if os.path.exists(_SWIFT_TMP_WAV):
            os.unlink(_SWIFT_TMP_WAV)
    except Exception:
        pass


atexit.register(_cleanup_tmp_wav)

### whisper caption junk and non-lexical mic breathing to ignore.
###
### this list used to also hold 'okay', 'ok', 'yeah', 'yep', 'sure', 'so', 'you',
### 'well', 'now', 'the', 'and', 'i', 'it', 'a', 'to'. those are not hallucinations,
### they are answers: the hri dialogue tree scripts 'cool lets do it',
### 'thanks lulo!', 'okay, call them' and 'ok, i guess i'm ready', so four scripted
### user turns were being discarded before the model ever saw them and lulo simply
### said nothing. only genuine caption artefacts and non-lexical murmurs stay.
HALLUCINATIONS = [
    '...', 'mm-hmm', 'mmhmm', 'mhm', 'hmm', 'uh', 'uhh', 'um', 'umm',
    'thanks for watching', 'thank you for watching', 'subtitles by',
    'sh hotel', 'la latker',
]
_HALLUCINATIONS_SET = {h.strip('.!?,') for h in HALLUCINATIONS}


### util
def is_hallucination(text):
    # filter mic breathing + whisper caption junk
    if not text:
        return True
    normalized = text.lower().strip().strip('.!?,')
    if not normalized:
        return True
    # a single character only counts as junk when it is not a word. 'a' and 'i'
    # are short but real, and dropping them silently loses a turn.
    if len(normalized) <= 1 and not normalized.isalpha():
        return True
    return normalized in _HALLUCINATIONS_SET


def _filter_stderr(proc, ready_event):
    for line in proc.stderr:
        line_lower = line.lower()
        if '[[stt_worker]]' in line_lower:
            console.print(f'[dim white]{line.strip()}[/]')
            if 'ready' in line_lower:
                ready_event.set()
        elif 'error' in line_lower or 'failed' in line_lower or 'exception' in line_lower:
            console.print(f'[bold red][[STT_WORKER ERROR]]: {line.strip()}[/]')


SWIFT_READY_TIMEOUT_S = 300


def start_swift_worker():
    # look for precompiled stt_worker binary first to skip swift build overhead
    candidate_bins = [
        os.path.join(
            BASE_DIR,
            'stt-coreml',
            'stt_worker',
            '.build',
            'arm64-apple-macosx',
            'release',
            'stt_worker',
        ),
        os.path.join(
            os.path.dirname(BASE_DIR),
            'nao-fast-agentic',
            'stt-coreml',
            'stt_worker',
            '.build',
            'arm64-apple-macosx',
            'release',
            'stt_worker',
        ),
    ]

    cmd = None
    work_dir = None
    for b in candidate_bins:
        if os.path.exists(b):
            cmd = [b]
            work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(b))))
            break

    if cmd is None:
        work_dir = os.path.join(BASE_DIR, 'stt-coreml', 'stt_worker')
        cmd = ['swift', 'run', '-c', 'release']

    proc = subprocess.Popen(
        cmd,
        cwd=work_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    ready_event = threading.Event()
    t = threading.Thread(target=_filter_stderr, args=(proc, ready_event), daemon=True)
    t.start()
    if not ready_event.wait(timeout=SWIFT_READY_TIMEOUT_S):
        try:
            proc.kill()
        except Exception:
            pass
        raise RuntimeError('stt worker never became ready, falling back to whisper')
    return proc


def _audio_to_wav_bytes(audio):
    # fast in-memory wav serialization for 16khz 16-bit mono without io.BytesIO overhead
    if audio.sample_rate == 16000 and audio.sample_width == 2:
        raw = audio.frame_data
        data_size = len(raw)
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF', data_size + 36, b'WAVE', b'fmt ',
            16, 1, 1, 16000, 32000, 2, 16, b'data', data_size
        )
        return header + raw
    return audio.get_wav_data(convert_rate=16000, convert_width=2)


### local whisper (non-mac stt). speech_recognition's recognize_whisper() calls
### whisper.load_model() on every single utterance, which reloads the weights from
### disk (and downloads ~140mb the very first time) for every turn. that makes
### linux transcription look like a dead stt worker. load the model once and reuse.
WHISPER_MODEL_NAME = os.environ.get('WHISPER_MODEL', 'base.en')
_WHISPER_MODEL = None
_WHISPER_LOCK = threading.Lock()

### auto = mac coreml, otherwise cached local whisper first then google.
### set STT_BACKEND=google to prefer the network, STT_BACKEND=whisper to stay offline.
STT_BACKEND = os.environ.get('STT_BACKEND', 'auto').lower()


def get_whisper_model():
    # load and cache the whisper model once for the life of the process
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        with _WHISPER_LOCK:
            if _WHISPER_MODEL is None:
                import whisper
                _WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_NAME)
    return _WHISPER_MODEL


def prewarm_whisper():
    # load (and first-run download) the model in the background so the first
    # utterance is not stalled for ~13s behind torch warmup
    if IS_MAC:
        return
    try:
        threading.Thread(target=get_whisper_model, daemon=True).start()
    except Exception:
        pass


def _audio_to_float32(audio):
    # whisper wants 16khz float32 mono in [-1, 1]
    import numpy as np
    raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _transcribe_whisper(audio):
    model = get_whisper_model()
    if model is None:
        raise RuntimeError('whisper model unavailable')
    result = model.transcribe(
        _audio_to_float32(audio),
        language='en',
        fp16=False,
        condition_on_previous_text=False,
    )
    return (result.get('text') or '').strip()


def transcribe_audio(recognizer, audio, swift_proc=None):
    '''
    multi-tier transcription:
    tier 1: coreml swift worker on mac
    tier 2: cached local whisper (reliable, offline) on non-mac
    tier 3: google speech recognition (network fallback)

    inputs:
    recognizer: speech_recognition recognizer
    audio: speech_recognition AudioData
    swift_proc: active swift subprocess
    outputs:
    text: transcribed string
    swift_proc: updated process handle
    '''
    if STT_BACKEND == 'google':
        use_whisper, use_google = False, True
    elif STT_BACKEND in ('whisper', 'local'):
        use_whisper, use_google = True, False
    else:  # auto
        use_whisper = not IS_MAC
        use_google = True

    # tier 1: coreml swift worker
    if IS_MAC and swift_proc is not None and swift_proc.poll() is None:
        try:
            wav_data = _audio_to_wav_bytes(audio)
            with open(_SWIFT_TMP_WAV, 'wb') as f:
                f.write(wav_data)
            text = None
            try:
                swift_proc.stdin.write(_SWIFT_TMP_WAV + '\n')
                swift_proc.stdin.flush()
                text = swift_proc.stdout.readline().strip()
            except BrokenPipeError:
                text = None
            if text is not None:
                # coreml processed the audio cleanly without crashing.
                # if text is empty, the audio was silence/room noise; do not fall through to cpu whisper
                return text, swift_proc
        except Exception as e:
            console.print(f'[dim yellow][[ coreml transcribe error: {e}, falling back ]][/]')

    # tier 2: cached local whisper. an empty result means silence, so we do not
    # fall through to google and burn a network round-trip on every pause.
    if use_whisper:
        try:
            return _transcribe_whisper(audio), swift_proc
        except sr.UnknownValueError:
            return '', swift_proc
        except Exception as e:
            console.print(f'[dim yellow][[ whisper error: {e}, falling back to google ]][/]')

    # tier 3: google speech recognition
    if use_google:
        try:
            return recognizer.recognize_google(audio).strip(), swift_proc
        except sr.UnknownValueError:
            return '', swift_proc
        except Exception as e:
            console.print(f'[dim yellow][[ google speech error: {e} ]][/]')

    return '', swift_proc


### words that really do mean "i am mid-thought, wait for me".
###
### 'you', 'the', 'to', 'my', 'i', "i'm", 'for', 'is', 'are', 'was' and 'just' used
### to be in here, which made 'thank you', 'nice to meet you', 'my knee is
### throbbing' and "i'm ready" each buy an extra 1.5s of dead air - and those are
### verbatim lines from the study's dialogue tree. only genuine hanging
### conjunctions and fillers are left.
CONTINUATION_CUES = {
    'and',
    'but',
    'or',
    'so',
    'because',
    'um',
    'uh',
    'uhh',
    'umm',
    'then',
    'if',
    'when',
    'wait',
    'also',
    'though',
}


def _ends_with_continuation(text):
    if not text:
        return False
    t = text.strip()
    if t.endswith(('...', '…', '—', '--', ',', '-', ':')):
        return True
    words = t.rsplit(maxsplit=1)
    if not words:
        return False
    last_word = words[-1].lower().strip('.,!?;:"\'-—–…')
    return last_word in CONTINUATION_CUES


### silero vad neural endpointing
_VAD_MODEL = None
_NP = None
_TORCH = None


def _get_stt_math():
    global _NP, _TORCH
    if _NP is None:
        import numpy as np
        _NP = np
    if _TORCH is None:
        import torch
        _TORCH = torch
    return _NP, _TORCH


def get_vad_model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        try:
            import silero_vad
            _VAD_MODEL = silero_vad.load_silero_vad(onnx=True)
        except Exception as e:
            console.print(f'[dim yellow][[ could not load silero vad: {e} ]][/]')
            _VAD_MODEL = None
    return _VAD_MODEL


def listen_audio_silero(
    source,
    timeout=None,
    phrase_time_limit=90.0,
    min_silence_duration_s=DEFAULT_VAD_SILENCE_S,
    non_speaking_duration_s=DEFAULT_NON_SPEAKING_S,
    speech_threshold=DEFAULT_VAD_SPEECH_THRESHOLD,
    nod_callback=None,
    use_smart_turn=True,
):
    '''
    captures one utterance using neural silero-vad for forgiving endpointing.

    inputs:
    source: open speech_recognition audio source (e.g. Microphone)
    timeout: max seconds to wait for speech to start
    phrase_time_limit: max speech duration (default 90s)
    min_silence_duration_s: silence duration to trigger endpoint (default 1.0s)
    non_speaking_duration_s: pre/post padding silence duration (default 0.25s)
    speech_threshold: vad onset probability floor (default 0.60); the effective
        onset and silence thresholds float above the measured room noise floor
    nod_callback: callable(seconds) triggered at 10s, 25s, 45s of continuous speech
    use_smart_turn: consult the semantic end-of-turn model, so min_silence_duration_s
        acts as a hard cap rather than the decision. falls back to the fixed window
        when the model is unavailable.
    outputs:
    AudioData instance
    '''
    vad_model = get_vad_model()
    if vad_model is None:
        raise RuntimeError('silero vad model unavailable')

    np, torch = _get_stt_math()
    vad_model.reset_states()

    sample_rate = getattr(source, 'SAMPLE_RATE', 16000)
    sample_width = getattr(source, 'SAMPLE_WIDTH', 2)
    frame_duration_s = 512.0 / 16000.0
    pre_speech_count = max(1, int(round(non_speaking_duration_s / frame_duration_s)))
    silence_frames_needed = max(2, int(round(min_silence_duration_s / frame_duration_s)))

    pre_speech_frames = collections.deque(maxlen=pre_speech_count)
    spoken_frames = []
    speaking_started = False
    silence_counter = 0
    onset_counter = 0
    speech_start_time = None
    start_time = time.time()
    nodded = {10: False, 25: False, 45: False}
    limit_reached = False

    # a crowded room scores well above zero on every frame, so a fixed silence
    # threshold of speech_threshold-0.15 can sit *below* the ambient level. the
    # counter then resets on room noise and the utterance never endpoints at all.
    # both thresholds therefore float on a measured floor instead.
    ambient_probs = collections.deque(maxlen=40)
    silence_threshold = speech_threshold - 0.15
    effective_speech_threshold = speech_threshold
    measured_floor = None

    # semantic endpointing: probe early, and only fall back to the full window
    # when the model says the speaker is not finished
    smart_enabled = use_smart_turn and smart_turn.get_session() is not None
    probe_frames = max(1, int(round(SMART_TURN_PROBE_S / frame_duration_s)))
    reprobe_frames = max(1, int(round(SMART_TURN_REPROBE_S / frame_duration_s)))
    next_probe_at = probe_frames
    smart_prob = None
    smart_frames = smart_turn.WINDOW_SAMPLES // 512

    float_buf = np.empty(512, dtype=np.float32)
    inv_scale = np.float32(1.0 / 32768.0)
    audio_float = torch.from_numpy(float_buf)

    raw_buffer = bytearray()
    stream = getattr(source, 'stream', None)
    if stream is None:
        raise RuntimeError('audio source stream is closed or missing')

    while True:
        chunk = stream.read(512)
        if not chunk:
            break
        raw_buffer.extend(chunk)

        while len(raw_buffer) >= 1024:
            if len(raw_buffer) == 1024:
                frame_bytes = bytes(raw_buffer)
                raw_buffer.clear()
            else:
                frame_bytes = bytes(raw_buffer[:1024])
                del raw_buffer[:1024]

            audio_int16 = np.frombuffer(frame_bytes, dtype=np.int16)
            np.multiply(audio_int16, inv_scale, out=float_buf)

            prob = vad_model(audio_float, 16000).item()

            if not speaking_started:
                if timeout is not None and (time.time() - start_time) > timeout:
                    raise sr.WaitTimeoutError('listening timed out waiting for speech')
                pre_speech_frames.append(frame_bytes)

                # only quiet frames describe the room, so a frame that looks like
                # speech must not be folded into the floor estimate
                if prob < speech_threshold:
                    ambient_probs.append(prob)
                    if len(ambient_probs) >= NOISE_FLOOR_MIN_FRAMES:
                        ranked = sorted(ambient_probs)
                        floor = ranked[int(0.9 * (len(ranked) - 1))]
                        measured_floor = floor
                        # keep silence strictly under the onset threshold, or
                        # nothing would ever count as the end of a phrase
                        silence_threshold = min(
                            speech_threshold - 0.02,
                            max(speech_threshold - 0.15, floor + NOISE_FLOOR_SILENCE_MARGIN),
                        )
                        effective_speech_threshold = max(
                            speech_threshold, floor + NOISE_FLOOR_SPEECH_MARGIN
                        )

                if prob >= effective_speech_threshold:
                    onset_counter += 1
                    if onset_counter >= MIN_SPEECH_FRAMES:
                        # speech onset is otherwise invisible, which makes a
                        # stalled vad indistinguishable from a dead microphone
                        console.print(
                            f'[bold dark_orange][[VAD]] speech detected '
                            f'(p={prob:.2f} onset>={effective_speech_threshold:.2f} '
                            f'silence<{silence_threshold:.2f})[/]'
                        )
                        speaking_started = True
                        speech_start_time = time.time()
                        spoken_frames.extend(pre_speech_frames)
                        silence_counter = 0
                else:
                    onset_counter = 0
            else:
                spoken_frames.append(frame_bytes)
                speech_elapsed = time.time() - speech_start_time

                if nod_callback:
                    for mark in (10, 25, 45):
                        if speech_elapsed >= mark and not nodded[mark]:
                            nodded[mark] = True
                            try:
                                nod_callback(mark)
                            except Exception:
                                pass

                if phrase_time_limit and speech_elapsed > phrase_time_limit:
                    # this used to break the inner loop before the silence counter
                    # was touched, while the outer exit only tested that counter.
                    # past the limit it stayed frozen below the threshold, so
                    # capture ran on until the stream closed - a hard hang.
                    limit_reached = True
                    endpoint_at = time.monotonic()
                    turn_metrics.note_capture(
                        speech_end=endpoint_at,
                        endpoint_at=endpoint_at,
                        silence_window_s=min_silence_duration_s,
                        noise_floor=measured_floor,
                    )
                    break

                if prob < silence_threshold:
                    silence_counter += 1

                    if (
                        smart_enabled
                        and silence_counter >= next_probe_at
                        and silence_counter < silence_frames_needed
                    ):
                        next_probe_at = silence_counter + reprobe_frames
                        # the model wants the whole current turn, most recent last
                        tail = b''.join(spoken_frames[-smart_frames:])
                        samples = np.frombuffer(tail, dtype=np.int16).astype(
                            np.float32
                        ) * inv_scale
                        complete, smart_prob = smart_turn.is_turn_complete(samples)
                        if complete:
                            endpoint_at = time.monotonic()
                            console.print(
                                f'[dim white]\\[\\[smart-turn]]: turn complete '
                                f'(p={smart_prob:.2f}) after '
                                f'{silence_counter * frame_duration_s:.2f}s silence[/]'
                            )
                            turn_metrics.note_capture(
                                speech_end=endpoint_at
                                - (silence_counter * frame_duration_s),
                                endpoint_at=endpoint_at,
                                silence_window_s=silence_counter * frame_duration_s,
                                noise_floor=measured_floor,
                            )
                            silence_counter = silence_frames_needed
                            break

                    if silence_counter >= silence_frames_needed:
                        # the span a person actually waits through starts at the
                        # last frame that was still speech, not here
                        endpoint_at = time.monotonic()
                        if smart_enabled and smart_prob is not None:
                            console.print(
                                f'[dim white]\\[\\[smart-turn]]: window cap reached '
                                f'(last p={smart_prob:.2f})[/]'
                            )
                        turn_metrics.note_capture(
                            speech_end=endpoint_at - (silence_counter * frame_duration_s),
                            endpoint_at=endpoint_at,
                            silence_window_s=min_silence_duration_s,
                            noise_floor=measured_floor,
                        )
                        break
                else:
                    silence_counter = 0
                    next_probe_at = probe_frames

        if limit_reached:
            break
        if speaking_started and silence_counter >= silence_frames_needed:
            break

    if not spoken_frames:
        return sr.AudioData(b'', sample_rate, sample_width)

    return sr.AudioData(b''.join(spoken_frames), sample_rate, sample_width)


# barge-in needs a higher bar than normal onset. echo cancellation on the handset
# is good but not perfect, and a false trigger cuts lulo off mid-word for nothing,
# which reads worse than not supporting interruption at all.
BARGE_IN_THRESHOLD = float(os.environ.get('BARGE_IN_THRESHOLD', '0.80'))
BARGE_IN_FRAMES = int(os.environ.get('BARGE_IN_FRAMES', '7'))  # ~224ms sustained


def watch_for_barge_in(source, stop_event, threshold=None, frames_needed=None):
    '''
    listen while lulo is speaking and report if the user starts talking over her.

    the pipeline was strictly half-duplex: nothing read the microphone during tts,
    and the only echo protection was flushing the input buffer afterwards. the
    patient physically could not interrupt, which is the single thing that most
    made the interaction feel like a kiosk rather than a conversation.

    this is safe to run concurrently with playback only because the main listen
    loop is parked while it runs, so there is exactly one reader on the stream.

    inputs:
    source: the same audio source the listen loop uses
    stop_event: threading.Event; set it to stand the watcher down
    threshold: vad probability a frame must clear (default BARGE_IN_THRESHOLD)
    frames_needed: consecutive frames required (default BARGE_IN_FRAMES)
    outputs:
    True when the user spoke over lulo, False when stopped or unavailable
    '''
    vad_model = get_vad_model()
    stream = getattr(source, 'stream', None)
    if vad_model is None or stream is None:
        return False

    threshold = BARGE_IN_THRESHOLD if threshold is None else threshold
    frames_needed = BARGE_IN_FRAMES if frames_needed is None else frames_needed

    np, torch = _get_stt_math()
    vad_model.reset_states()
    float_buf = np.empty(512, dtype=np.float32)
    inv_scale = np.float32(1.0 / 32768.0)
    audio_float = torch.from_numpy(float_buf)

    raw_buffer = bytearray()
    hits = 0
    try:
        while not stop_event.is_set():
            chunk = stream.read(512)
            if not chunk:
                # a short read here means the sink went quiet, not that the user
                # spoke, so stand down rather than guessing
                return False
            raw_buffer.extend(chunk)
            while len(raw_buffer) >= 1024:
                if stop_event.is_set():
                    return False
                frame_bytes = bytes(raw_buffer[:1024])
                del raw_buffer[:1024]
                audio_int16 = np.frombuffer(frame_bytes, dtype=np.int16)
                np.multiply(audio_int16, inv_scale, out=float_buf)
                if vad_model(audio_float, 16000).item() >= threshold:
                    hits += 1
                    if hits >= frames_needed:
                        return True
                else:
                    hits = 0
    except Exception:
        return False
    return False


def listen_once(
    recognizer,
    source,
    swift_proc=None,
    timeout=None,
    phrase_limit=90,
    use_silero=True,
    nod_callback=None,
    pause_threshold=None,
    use_smart_turn=True,
):
    '''
    block for one complete utterance with adaptive continuation endpointing.

    inputs:
    recognizer: active speech_recognition recognizer
    source: open microphone source
    swift_proc: coreml worker handle on mac, else none
    timeout: optional timeout in seconds to wait for speech start
    phrase_limit: max speech duration in seconds (default: 90)
    use_silero: whether to use neural silero vad (default: True)
    nod_callback: callable(seconds) for monologue nod feedback (10s, 25s, 45s)
    pause_threshold: silence seconds before ending speech (default: 2.0s)
    outputs:
    transcribed string, '' if silence or junk
    '''
    vad_model = get_vad_model() if use_silero else None
    silence_s = pause_threshold if pause_threshold is not None else DEFAULT_VAD_SILENCE_S
    try:
        if vad_model is not None:
            audio = listen_audio_silero(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_limit,
                min_silence_duration_s=silence_s,
                non_speaking_duration_s=DEFAULT_NON_SPEAKING_S,
                nod_callback=nod_callback,
                use_smart_turn=use_smart_turn,
            )
        else:
            recognizer.pause_threshold = silence_s
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
    except sr.WaitTimeoutError:
        return ''
    except Exception as e:
        console.print(f'[bold red][[ capture error: {e} ]][/]')
        return ''

    if not audio.get_raw_data():
        # no speech made it through the vad, so there is nothing to transcribe
        # and the coreml worker rejects empty audio with invalidAudioData
        return ''

    text, _ = transcribe_audio(recognizer, audio, swift_proc)
    turn_metrics.note_stt_done()
    if is_hallucination(text):
        return ''
    text = text.strip()

    # adaptive endpointing: if user paused on a continuation cue, give a bonus window
    extensions = 0
    continuation_silence_s = max(1.2, silence_s * 0.75)
    while text and _ends_with_continuation(text) and extensions < 3:
        try:
            if vad_model is not None:
                extra_audio = listen_audio_silero(
                    source,
                    timeout=1.5,
                    phrase_time_limit=20,
                    min_silence_duration_s=continuation_silence_s,
                use_smart_turn=use_smart_turn,
                    nod_callback=nod_callback,
                )
            else:
                extra_audio = recognizer.listen(source, timeout=1.5, phrase_time_limit=20)
            extra_text, _ = transcribe_audio(recognizer, extra_audio, swift_proc)
            if extra_text and not is_hallucination(extra_text):
                text = f'{text} {extra_text.strip()}'.strip()
                extensions += 1
            else:
                break
        except (sr.WaitTimeoutError, Exception):
            break

    return text


## daemon mode (original file-queue pipeline)
def main():
    swift_proc = None
    if IS_MAC:
        try:
            swift_proc = start_swift_worker()
        except Exception as e:
            console.print(f'[dim yellow][[ coreml worker failed, whisper fallback: {e} ]]')
    if swift_proc is None:
        console.print('[dim white][[STT_WORKER]]: python whisper fallback ready[/]')
        prewarm_whisper()

    r = sr.Recognizer()
    r.pause_threshold = DEFAULT_VAD_SILENCE_S
    r.non_speaking_duration = DEFAULT_NON_SPEAKING_S

    with sr.Microphone(device_index=MICROPHONE_INDEX) as source:
        if getattr(source, 'stream', None) is None:
            source.stream = type('DummyStream', (), {'close': lambda self: None})()
            console.print('[bold red][[ fatal: bad microphone index, check .env ]]')
            sys.exit(1)
        r.adjust_for_ambient_noise(source, duration=2.0)

        was_listening = False
        while True:
            if safe_read(LISTEN_FILE) == 'no':
                was_listening = False
                time.sleep(0.03)
                continue
            if not was_listening:
                try:
                    stream = getattr(getattr(source, 'stream', None), 'pyaudio_stream', None)
                    if stream and stream.is_active():
                        avail = stream.get_read_available()
                        if 0 < avail < 32768:
                            stream.read(avail, exception_on_overflow=False)
                except Exception:
                    pass
                console.print('[bold dark_orange][[LISTENING]][/]')
                was_listening = True

            if IS_MAC and (swift_proc is None or swift_proc.poll() is not None):
                console.print('[dim white][[ reviving crashed stt worker... ]][/]')
                try:
                    swift_proc = start_swift_worker()
                except Exception as e:
                    console.print(f'[dim yellow][[ stt worker failed, whisper fallback: {e} ]]')
                    swift_proc = None

            text = listen_once(r, source, swift_proc)
            if text:
                safe_write(TRANSCRIPTION_FILE, text)
                safe_write(LISTEN_FILE, 'no')
                was_listening = False


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
