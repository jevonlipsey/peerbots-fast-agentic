"""
kokoro tts manager and async client.
handles voice mixing syntax, process auto-start, and in-memory synthesis.
"""

import asyncio
import os
import shutil
import subprocess
import time
import httpx


### magic constants & paths
KOKORO_BASE_URL = os.environ.get('KOKORO_BASE_URL', 'http://localhost:8880')
KOKORO_SPEECH_URL = f'{KOKORO_BASE_URL.rstrip("/")}/v1/audio/speech'
KOKORO_DOCS_URL = f'{KOKORO_BASE_URL.rstrip("/")}/docs'
KOKORO_DIR = os.path.expanduser(os.environ.get('KOKORO_DIR', '~/Kokoro-FastAPI'))
DEFAULT_VOICE = 'af_heart:0.6+af_bella:0.4'
DEFAULT_SPEED = 1.05
AUDIO_FORMAT = os.environ.get('AUDIO_FORMAT', 'wav')
START_TIMEOUT_S = 18.0
MAX_CACHE_ENTRIES = 50

_KOKORO_PROC = None
_STARTED_BY_US = False
_ASYNC_CLIENT = None
_TTS_CACHE = {}
_VOICE_CACHE = {}


def get_async_client():
    """returns cached persistent httpx client for sub-millisecond connection reuse"""
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is None or _ASYNC_CLIENT.is_closed:
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=60.0)
        _ASYNC_CLIENT = httpx.AsyncClient(timeout=10.0, limits=limits)
    return _ASYNC_CLIENT


async def close_client():
    """closes persistent client on shutdown"""
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is not None and not _ASYNC_CLIENT.is_closed:
        try:
            await _ASYNC_CLIENT.aclose()
        except Exception:
            pass
        _ASYNC_CLIENT = None


### voice parsing
def normalize_kokoro_voice(voice_str):
    """
    converts af_heart:0.6+af_bella:0.4 syntax into kokoro native af_heart(0.6)+af_bella(0.4)

    inputs:
    voice_str: voice name or mixture string
    outputs:
    normalized: native kokoro formatted voice string
    """
    if not voice_str:
        return 'af_heart'
    cached = _VOICE_CACHE.get(voice_str)
    if cached is not None:
        return cached
    chunks = voice_str.split('+')
    parts = []
    for chunk in chunks:
        chunk = chunk.strip()
        if ':' in chunk and '(' not in chunk:
            name, weight = chunk.split(':', 1)
            parts.append(f'{name.strip()}({weight.strip()})')
        else:
            parts.append(chunk)
    result = '+'.join(parts)
    _VOICE_CACHE[voice_str] = result
    return result


### lifecycle
def is_kokoro_running(timeout=0.6):
    """
    quick sync ping to check if kokoro fastapi is responding

    inputs:
    timeout: connection timeout in seconds
    outputs:
    running: boolean true if server responds with 200
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(KOKORO_DOCS_URL)
            return resp.status_code == 200
    except Exception:
        return False


async def ais_kokoro_running(timeout=0.6):
    """
    async check if kokoro is reachable

    inputs:
    timeout: timeout in seconds
    outputs:
    running: boolean
    """
    try:
        client = get_async_client()
        resp = await client.get(KOKORO_BASE_URL, timeout=timeout)
        return resp.status_code in (200, 404, 405)
    except Exception:
        return False


def start_kokoro_process():
    """
    spawns start-cpu_mac.sh (or start-gpu_mac.sh fallback) in the background if kokoro dir exists

    inputs:
    none
    outputs:
    proc: subprocess.Popen or None
    """
    global _KOKORO_PROC, _STARTED_BY_US
    cpu_script = os.path.join(KOKORO_DIR, 'start-cpu_mac.sh')
    gpu_script = os.path.join(KOKORO_DIR, 'start-gpu_mac.sh')
    script_path = cpu_script if os.path.isfile(cpu_script) else gpu_script
    if not os.path.isfile(script_path):
        return None
    try:
        devnull = open(os.devnull, 'wb')
        proc = subprocess.Popen(
            ['bash', script_path],
            cwd=KOKORO_DIR,
            stdout=devnull,
            stderr=devnull,
            start_new_session=True,
        )
        _KOKORO_PROC = proc
        _STARTED_BY_US = True
        return proc
    except Exception:
        return None


PREWARM_PHRASES = [
    "Hi! I'm Lulo. Nice to meet you! What's your name?",
    'Hey! How are you doing today?',
    'hello there',
    'awesome,',
    'great,',
    'sounds good!',
    "let's do it!",
    "you got this!",
]


async def ensure_kokoro_ready(timeout=START_TIMEOUT_S):
    """
    verifies kokoro is up, auto-launching if needed, and warms up the model pipeline

    inputs:
    timeout: max seconds to wait for boot
    outputs:
    ready: boolean indicating if server is ready
    """
    ready = False
    if await ais_kokoro_running():
        ready = True
    else:
        # launch script
        proc = start_kokoro_process()
        if proc is None:
            return False

        start_t = time.time()
        while time.time() - start_t < timeout:
            if proc.poll() is not None:
                # crashed on boot
                return False
            if await ais_kokoro_running(timeout=0.5):
                ready = True
                break
            await asyncio.sleep(0.5)

    if ready:
        # warm up kokoro pipeline & seed cache with common conversational phrases concurrently
        await asyncio.gather(
            *(
                synthesize_speech(
                    phrase,
                    voice=DEFAULT_VOICE,
                    speed=DEFAULT_SPEED,
                    response_format=AUDIO_FORMAT,
                )
                for phrase in PREWARM_PHRASES
            ),
            return_exceptions=True,
        )
    return ready


def stop_kokoro_process():
    """
    terminates kokoro subprocess if launched by this run

    inputs:
    none
    outputs:
    none
    """
    global _KOKORO_PROC, _STARTED_BY_US
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(close_client())
    except Exception:
        pass
    if _STARTED_BY_US and _KOKORO_PROC is not None:
        try:
            _KOKORO_PROC.terminate()
            _KOKORO_PROC.wait(timeout=2.0)
        except Exception:
            try:
                _KOKORO_PROC.kill()
            except Exception:
                pass
        _KOKORO_PROC = None
        _STARTED_BY_US = False


### synthesis
async def synthesize_speech(text, voice=DEFAULT_VOICE, speed=DEFAULT_SPEED, response_format=AUDIO_FORMAT):
    """
    synthesizes speech in memory via kokoro fastapi /v1/audio/speech with exact-repeat caching

    inputs:
    text: sentence to speak
    voice: voice name or mixture string
    speed: playback speed multiplier
    response_format: audio container format (mp3 or wav)
    outputs:
    audio_bytes: raw in-memory audio bytes or None if synthesis failed
    """
    global _TTS_CACHE
    cleaned = text.strip()
    if not cleaned:
        return None

    norm_voice = normalize_kokoro_voice(voice)
    cache_key = (cleaned.lower(), norm_voice, speed, response_format)
    if cache_key in _TTS_CACHE:
        return _TTS_CACHE[cache_key]

    payload = {
        'model': 'kokoro',
        'input': cleaned,
        'voice': norm_voice,
        'response_format': response_format,
        'speed': speed,
    }

    try:
        client = get_async_client()
        resp = await client.post(KOKORO_SPEECH_URL, json=payload)
        if resp.status_code == 200:
            audio_bytes = resp.content
            if len(_TTS_CACHE) >= MAX_CACHE_ENTRIES:
                _TTS_CACHE.pop(next(iter(_TTS_CACHE)))
            _TTS_CACHE[cache_key] = audio_bytes
            return audio_bytes
        return None
    except Exception:
        return None
