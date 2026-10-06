"""
droid headless audio daemon client and sequential playback queue.
sends in-memory audio bytes to http://100.119.180.97:8765 over tailscale.
"""

import asyncio
import io
import os
import time
import httpx
import mutagen.mp3


### config & endpoints
DROID_URL = os.environ.get('DROID_URL', 'http://100.119.180.97:8765')
DEFAULT_CONTENT_TYPE = 'audio/mpeg'
DEFAULT_TIMEOUT_S = 6.0

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
    extracts exact playback duration from in-memory mp3 bytes

    inputs:
    audio_bytes: raw mp3 byte stream
    outputs:
    duration_s: float seconds or fallback estimate
    """
    if not audio_bytes:
        return 0.0
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


### send chunk
async def asend_audio_chunk(audio_bytes, content_type=DEFAULT_CONTENT_TYPE, timeout=DEFAULT_TIMEOUT_S):
    """
    posts a single in-memory audio chunk to droid daemon

    inputs:
    audio_bytes: raw audio stream
    content_type: mime type (audio/mpeg or audio/wav)
    timeout: request timeout
    outputs:
    success: boolean
    """
    if not audio_bytes:
        return False
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
        await close_client()


    async def enqueue(self, audio_bytes, duration_s=None):
        """adds an audio chunk to the sequential playback stream"""
        if duration_s is None:
            duration_s = get_audio_duration_s(audio_bytes)
        await self.start()
        await self.queue.put((audio_bytes, duration_s))


    async def wait_complete(self):
        """waits until all queued audio chunks have been sent and played"""
        await self.queue.join()
        while self.is_playing:
            await asyncio.sleep(0.05)


    async def _worker(self):
        """background loop sending chunks sequentially with duration pacing"""
        while True:
            audio_bytes, duration_s = await self.queue.get()
            self.is_playing = True
            try:
                ok = await asend_audio_chunk(audio_bytes)
                if ok and duration_s > 0:
                    # leave a tiny 50ms overlap for gapless speech
                    sleep_time = max(0.0, duration_s - 0.05)
                    await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                self.is_playing = False
                self.queue.task_done()
                raise
            except Exception:
                pass
            finally:
                self.is_playing = False
                self.queue.task_done()
