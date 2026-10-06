'''
local speech-to-text. on mac uses the swift coreml worker,
else falls back to python whisper. exposes listen_once() for
main.py plus the original file-queue daemon mode.

layer 1: microphone -> whisper_stt -> openai_response
'''

import atexit
import os
import platform
import random
import re
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
            wav_data = audio.get_wav_data(convert_rate=16000, convert_width=2)
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
    last_word = re.sub(r'^[^\w]+|[^\w]+$', '', words[-1].lower())
    return last_word in CONTINUATION_CUES


def listen_once(recognizer, source, swift_proc=None, timeout=None, phrase_limit=20):
    '''
    block for one complete utterance with adaptive continuation endpointing.

    inputs:
    recognizer: active speech_recognition recognizer
    source: open microphone source
    swift_proc: coreml worker handle on mac, else none
    outputs:
    transcribed string, '' if silence or junk
    '''
    try:
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
    while text and _ends_with_continuation(text) and extensions < 2:
        try:
            extra_audio = recognizer.listen(source, timeout=1.2, phrase_time_limit=10)
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
