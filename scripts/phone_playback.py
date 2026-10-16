'''
play kokoro tts through the android phone's own speaker.

the phone browser already keeps a websocket open for the microphone, so speech
rides back down the same socket and the page plays it. no droid daemon, no
tailscale hop, and the audio comes out of the handset the user is looking at.
'''

import asyncio
import time

from rich.console import Console

from droid_client import get_audio_duration_s

console = Console()

# extra slack on top of the clip length so a chunk is never cut off by the next
TAIL_SLACK_S = 0.15


class PhoneSpeakerQueue:
    '''
    same interface as DroidPlaybackQueue, but the sink is the phone browser.
    chunks are paced by their own duration so they cannot overlap on the handset.
    '''

    def __init__(self, server):
        self.server = server
        self.queue = asyncio.Queue()
        self.worker_task = None
        self.is_playing = False
        self.expected_idx = 0
        self.playback_end_time = 0.0

    def ensure_worker(self):
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())

    def reset_turn(self):
        self.expected_idx = 0
        self.playback_end_time = 0.0
        self.ensure_worker()

    async def start(self):
        self.ensure_worker()

    async def enqueue(self, audio_bytes, duration_s=None, chunk_idx=None):
        if duration_s is None:
            duration_s = get_audio_duration_s(audio_bytes)
        if chunk_idx is not None:
            if chunk_idx != self.expected_idx:
                console.print(
                    f'  [dim yellow][[ tts chunk inversion: got #{chunk_idx}, expected #{self.expected_idx} ]][/]'
                )
            self.expected_idx = chunk_idx + 1
        self.ensure_worker()
        self.queue.put_nowait((audio_bytes, duration_s, chunk_idx))

    async def wait_complete(self):
        await self.queue.join()
        # hold the caller until the last clip has actually finished on the phone
        remaining = self.playback_end_time - time.time()
        if remaining > 0:
            await asyncio.sleep(remaining)
        self.is_playing = False
        self.expected_idx = 0
        self.playback_end_time = 0.0

    async def _worker(self):
        while True:
            audio_bytes, duration_s, chunk_idx = await self.queue.get()
            self.is_playing = True
            try:
                sent = await self.server.send_audio(audio_bytes)
                if not sent:
                    console.print(
                        '[dim yellow][[ phone playback skipped: no browser attached ]][/]'
                    )
                    continue
                if duration_s > 0:
                    now = time.time()
                    self.playback_end_time = (
                        max(now, self.playback_end_time) + duration_s + TAIL_SLACK_S
                    )
                    await asyncio.sleep(max(0.0, self.playback_end_time - time.time()))
            except asyncio.CancelledError:
                self.is_playing = False
                raise
            except Exception:
                pass
            finally:
                self.is_playing = False
                self.queue.task_done()

    async def stop(self):
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
            self.queue.task_done()
        self.is_playing = False
        self.expected_idx = 0
