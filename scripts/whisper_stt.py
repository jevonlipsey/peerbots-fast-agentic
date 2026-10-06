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
MICROPHONE_INDEX = None

IS_MAC = platform.system() == 'Darwin'
_SWIFT_TMP_WAV = os.path.join(tempfile.gettempdir(), f'lulo_stt_{os.getpid()}.wav')


def _cleanup_tmp_wav():
    try:
        if os.path.exists(_SWIFT_TMP_WAV):
            os.unlink(_SWIFT_TMP_WAV)
    except Exception:
        pass


atexit.register(_cleanup_tmp_wav)

### short murmurs + whisper hallucinations to ignore
HALLUCINATIONS = [
    '...', 'you', 'now', 'mm-hmm', 'mmhmm', 'mhm', 'uh', 'uhh', 'um', 'umm',
    'okay', 'ok', 'yeah', 'yep', 'sure', 'so',
    'the', 'and', 'on', 'in', 'well', 'i', 'it', 'a', 'to',
    'thanks for watching', 'thank you for watching', 'subtitles by',
    'sh hotel', 'la latker',
]


### util
def is_hallucination(text):
    # filter mic breathing + whisper caption junk
    if not text:
        return True
    normalized = text.lower().strip().strip('.!?,')
    if len(normalized) <= 1:
        return True
    stripped = [h.strip('.!?,') for h in HALLUCINATIONS]
    if normalized in stripped:
        return True
    # filter out single isolated filler words
    if normalized in ('the', 'and', 'on', 'in', 'well', 'a', 'an', 'to'):
        return True
    return False


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


def transcribe_audio(recognizer, audio, swift_proc=None):
    '''
    multi-tier transcription:
    tier 1: coreml swift worker on mac
    tier 2: google speech recognition
    tier 3: local whisper base model

    inputs:
    recognizer: speech_recognition recognizer
    audio: speech_recognition AudioData
    swift_proc: active swift subprocess
    outputs:
    text: transcribed string
    swift_proc: updated process handle
    '''
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
            console.print(f'[dim yellow][[ coreml transcribe error: {e}, falling back to google ]][/]')

    # tier 2: google speech recognition fallback
    try:
        text = recognizer.recognize_google(audio).strip()
        return text, swift_proc
    except sr.UnknownValueError:
        return '', swift_proc
    except Exception as e:
        console.print(f'[dim yellow][[ google speech error: {e}, falling back to whisper ]][/]')

    # tier 3: local whisper fallback
    try:
        return recognizer.recognize_whisper(audio, model='base.en').strip(), swift_proc
    except sr.UnknownValueError:
        return '', swift_proc
    except Exception as e:
        console.print(f'[bold red][[ whisper fallback error: {e} ]][/]')
        return '', swift_proc


CONTINUATION_CUES = {
    'and',
    'but',
    'or',
    'so',
    'because',
    'like',
    'um',
    'uh',
    'uhh',
    'umm',
    'then',
    'if',
    'with',
    'when',
}


def _ends_with_continuation(text):
    if not text:
        return False
    t = text.strip()
    if t.endswith(('...', '…', '—', '--', ',')):
        return True
    words = t.split()
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
    min_silence_duration_s=0.15,
    non_speaking_duration_s=0.15,
    speech_threshold=0.45,
    nod_callback=None,
):
    '''
    captures one utterance using neural silero-vad for instant ~150ms speech-offset detection.

    inputs:
    source: open speech_recognition audio source (e.g. Microphone)
    timeout: max seconds to wait for speech to start
    phrase_time_limit: max speech duration (default 90s)
    min_silence_duration_s: silence duration to trigger endpoint (default 0.15s)
    non_speaking_duration_s: pre/post padding silence duration (default 0.15s)
    speech_threshold: vad speech probability threshold (default 0.45)
    nod_callback: callable(seconds) triggered at 10s, 25s, 45s of continuous speech
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
    speech_start_time = None
    start_time = time.time()
    nodded = {10: False, 25: False, 45: False}

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
            frame_bytes = bytes(raw_buffer[:1024])
            del raw_buffer[:1024]

            audio_int16 = np.frombuffer(frame_bytes, dtype=np.int16)
            audio_float = torch.from_numpy(audio_int16.astype(np.float32) * (1.0 / 32768.0))

            prob = vad_model(audio_float, 16000).item()

            if not speaking_started:
                if timeout is not None and (time.time() - start_time) > timeout:
                    raise sr.WaitTimeoutError('listening timed out waiting for speech')
                pre_speech_frames.append(frame_bytes)
                if prob >= speech_threshold:
                    speaking_started = True
                    speech_start_time = time.time()
                    spoken_frames.extend(pre_speech_frames)
                    spoken_frames.append(frame_bytes)
                    silence_counter = 0
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
                    break

                if prob < (speech_threshold - 0.15):
                    silence_counter += 1
                    if silence_counter >= silence_frames_needed:
                        break
                else:
                    silence_counter = 0

        if speaking_started and silence_counter >= silence_frames_needed:
            break

    if not spoken_frames:
        return sr.AudioData(b'', sample_rate, sample_width)

    return sr.AudioData(b''.join(spoken_frames), sample_rate, sample_width)


def listen_once(
    recognizer,
    source,
    swift_proc=None,
    timeout=None,
    phrase_limit=90,
    use_silero=True,
    nod_callback=None,
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
    outputs:
    transcribed string, '' if silence or junk
    '''
    vad_model = get_vad_model() if use_silero else None
    try:
        if vad_model is not None:
            audio = listen_audio_silero(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_limit,
                min_silence_duration_s=0.15,
                non_speaking_duration_s=0.15,
                nod_callback=nod_callback,
            )
        else:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
    except sr.WaitTimeoutError:
        return ''
    except Exception as e:
        console.print(f'[bold red][[ capture error: {e} ]][/]')
        return ''

    text, _ = transcribe_audio(recognizer, audio, swift_proc)
    if is_hallucination(text):
        return ''
    text = text.strip()

    # adaptive endpointing: if user paused on a continuation cue, give a quick bonus window
    extensions = 0
    while text and _ends_with_continuation(text) and extensions < 3:
        try:
            if vad_model is not None:
                extra_audio = listen_audio_silero(
                    source,
                    timeout=1.2,
                    phrase_time_limit=20,
                    min_silence_duration_s=0.20,
                    nod_callback=nod_callback,
                )
            else:
                extra_audio = recognizer.listen(source, timeout=1.2, phrase_time_limit=20)
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

    r = sr.Recognizer()
    r.pause_threshold = 0.8
    r.non_speaking_duration = 0.3

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
