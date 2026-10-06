"""
droid headless audio daemon client and sequential playback queue.
sends in-memory audio bytes to http://100.119.180.97:8765 over tailscale.
"""

import asyncio
import io
import os
import struct
import time
import httpx
import mutagen.mp3
from rich.console import Console

console = Console()

### config & endpoints
DROID_URL = os.environ.get('DROID_URL', 'http://100.119.180.97:8765')
DEFAULT_CONTENT_TYPE = 'audio/wav'
DEFAULT_TIMEOUT_S = 6.0
_HEADERS_WAV = {'Content-Type': 'audio/wav'}
_HEADERS_MP3 = {'Content-Type': 'audio/mpeg'}

_ASYNC_CLIENT = None


def get_async_client():
    """cached persistent client for low-latency reusable http connections"""
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is None or _ASYNC_CLIENT.is_closed:
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=60.0)
        _ASYNC_CLIENT = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S, limits=limits)
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


### duration estimation
def get_audio_duration_s(audio_bytes):
    """
    extracts exact playback duration from in-memory wav or mp3 bytes

    inputs:
    audio_bytes: raw wav or mp3 byte stream
    outputs:
    duration_s: float seconds or fallback estimate
    """
    if not audio_bytes:
        return 0.0

    # fast path for wav: direct struct header unpack without io.BytesIO or wave.open overhead
    if audio_bytes.startswith(b'RIFF') and len(audio_bytes) >= 36:
        try:
            ch, rate, byte_rate = struct.unpack_from('<HII', audio_bytes, 22)
            if byte_rate > 0:
                # streamed wavs (like kokoro-fastapi) set nframes to 2147483647
                if len(audio_bytes) >= 44 and audio_bytes[36:40] == b'data':
                    header_offset = 44
                else:
                    data_pos = audio_bytes.find(b'data')
                    header_offset = data_pos + 8 if data_pos != -1 else 44
                actual_data_len = max(0, len(audio_bytes) - header_offset)
                return max(0.1, actual_data_len / float(byte_rate))
        except Exception:
            pass
        # fallback for 24khz 16-bit mono: 48000 bytes/sec
        return max(0.1, (len(audio_bytes) - 44) / 48000.0)

    # mp3 path
    try:
        mp3 = mutagen.mp3.MP3(io.BytesIO(audio_bytes))
        return float(mp3.info.length)
    except Exception:
        # fallback rough estimate: ~16kbps or ~150 words per min
        return max(1.0, len(audio_bytes) / 16000.0)


### health
def is_droid_reachable(timeout=0.6):
    """
    checks if droid daemon is listening on port 8765

    inputs:
    timeout: socket timeout in seconds
    outputs:
    reachable: boolean
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            # server returns 501 on GET but proves port is open and answering
            resp = client.get(DROID_URL)
            return resp.status_code in (200, 501)
    except Exception:
        return False


async def ais_droid_reachable(timeout=0.6):
    """
    async reachability check for droid

    inputs:
    timeout: connection timeout
    outputs:
    reachable: boolean
    """
    try:
        client = get_async_client()
        resp = await client.get(DROID_URL, timeout=timeout)
        return resp.status_code in (200, 501)
    except Exception:
        return False


async def probe_droid_daemon():
    """
    probes droid daemon to inspect supported endpoints and responsiveness

    inputs:
    none
    outputs:
    results: dict of endpoint -> status code or error
    """
    endpoints = ['/', '/status', '/stop', '/clear', '/ping']
    results = {}
    client = get_async_client()
    for ep in endpoints:
        url = f'{DROID_URL.rstrip("/")}{ep}'
        try:
            r = await client.get(url, timeout=1.0)
            results[f'GET {ep}'] = r.status_code
        except Exception as e:
            results[f'GET {ep}'] = str(e)
    return results


### send chunk
async def asend_audio_chunk(audio_bytes, content_type=None, timeout=DEFAULT_TIMEOUT_S):
    """
    posts a single in-memory audio chunk to droid daemon

    inputs:
    audio_bytes: raw audio stream
    content_type: mime type (auto-detects audio/wav or audio/mpeg if None)
    timeout: request timeout
    outputs:
    success: boolean
    """
    if not audio_bytes:
        return False

    if content_type is None:
        headers = _HEADERS_WAV if audio_bytes.startswith(b'RIFF') else _HEADERS_MP3
    elif content_type == 'audio/wav':
        headers = _HEADERS_WAV
    elif content_type == 'audio/mpeg':
        headers = _HEADERS_MP3
    else:
        headers = {'Content-Type': content_type}
    try:
        client = get_async_client()
        resp = await client.post(DROID_URL, content=audio_bytes, headers=headers, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


### sequential queue
class DroidPlaybackQueue:
    """
    manages sequential audio streaming to droid so chunks do not overlap.
    plays chunk 1 immediately, then paces subsequent chunks by audio duration.
    """

    def __init__(self, droid_url=DROID_URL):
        self.droid_url = droid_url
        self.queue = asyncio.Queue()
        self.worker_task = None
        self.is_playing = False
        self.expected_idx = 0
        self.playback_end_time = 0.0
        self.last_upload_time = 0.12


    def reset_turn(self):
        """resets expected chunk index and playback timing for a new conversational turn"""
        self.expected_idx = 0
        self.playback_end_time = 0.0


    async def start(self):
        """starts background worker if not already running"""
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())


    async def stop(self):
        """cancels worker and flushes remaining queue items"""
        if self.worker_task is not None:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except (asyncio.CancelledError, Exception):
                pass
            self.worker_task = None
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break
        self.is_playing = False
        self.expected_idx = 0
        await close_client()


    async def enqueue(self, audio_bytes, duration_s=None, chunk_idx=None):
        """adds an audio chunk to the sequential playback stream"""
        if duration_s is None:
            duration_s = get_audio_duration_s(audio_bytes)
        if chunk_idx is not None:
            if chunk_idx != self.expected_idx:
                console.print(f'  [dim yellow][[ tts chunk inversion: got #{chunk_idx}, expected #{self.expected_idx} ]][/]')
            self.expected_idx = chunk_idx + 1
        await self.start()
        await self.queue.put((audio_bytes, duration_s, chunk_idx))


    async def wait_complete(self):
        """waits until all queued audio chunks have been sent and played completely"""
        await self.queue.join()
        # ensure last chunk's audio finishes playing before releasing caller
        remaining = self.playback_end_time - time.time()
        if remaining > 0:
            await asyncio.sleep(remaining)
        self.is_playing = False
        self.expected_idx = 0
        self.playback_end_time = 0.0


    async def _worker(self):
        """background loop sending chunks sequentially with duration pacing"""
        while True:
            audio_bytes, duration_s, chunk_idx = await self.queue.get()
            self.is_playing = True
            try:
                t0 = time.perf_counter()
                ok = await asend_audio_chunk(audio_bytes)
                t_upload = time.perf_counter() - t0
                if ok and t_upload > 0:
                    self.last_upload_time = 0.7 * self.last_upload_time + 0.3 * t_upload

                if ok and duration_s > 0:
                    now = time.time()
                    self.playback_end_time = max(now, self.playback_end_time) + duration_s
                    # lead time accounts for tailscale upload round-trip for the NEXT chunk
                    lead_time = min(duration_s * 0.5, self.last_upload_time + 0.02)
                    target_wake = self.playback_end_time - lead_time
                    wait_s = max(0.0, target_wake - time.time())
                    await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                self.is_playing = False
                raise
            except Exception:
                pass
            finally:
                self.is_playing = False
                self.queue.task_done()
