'''
local speaker playback for kokoro tts.

same interface as DroidPlaybackQueue, so main.py can swap it in when the droid
daemon is not around and still get sequential, non-overlapping speech. afplay
blocks for the length of the clip, which gives us the pacing for free.
'''

import asyncio
import os
import shutil
import sys
import tempfile

from rich.console import Console

from kokoro_manager import AUDIO_FORMAT

console = Console()

# a command that takes a file path as its last argument and blocks until the clip
# finishes. afplay is mac only; aplay is the alsa equivalent on linux. override
# with LOCAL_PLAY_CMD for anything else (ffplay, paplay, sox...).
_DEFAULT_PLAY_CMD = 'afplay' if sys.platform == 'darwin' else 'aplay'
LOCAL_PLAY_CMD = os.environ.get('LOCAL_PLAY_CMD', _DEFAULT_PLAY_CMD)


class LocalPlaybackQueue:
    '''
    plays kokoro chunks through this machine's speakers, one at a time.
    '''

    def __init__(self):
        self.queue = asyncio.Queue()
        self.worker_task = None
        self.is_playing = False
        self.expected_idx = 0
        self._tmpdir = tempfile.mkdtemp(prefix='lulo_tts_')
        self._clip_id = 0
        self.interrupted = False
        self._proc = None

    def ensure_worker(self):
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())

    def reset_turn(self):
        self.expected_idx = 0
        self.interrupted = False
        self.ensure_worker()

    async def interrupt(self):
        '''barge-in: drop queued clips and kill the one afplay is holding'''
        self.interrupted = True
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except Exception:
                break
        proc = getattr(self, '_proc', None)
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except Exception:
                pass
        self.is_playing = False

    async def start(self):
        self.ensure_worker()

    async def enqueue(self, audio_bytes, duration_s=None, chunk_idx=None):
        if self.interrupted:
            return
        if chunk_idx is not None:
            if chunk_idx != self.expected_idx:
                console.print(
                    f'  [dim yellow][[ tts chunk inversion: got #{chunk_idx}, expected #{self.expected_idx} ]][/]'
                )
            self.expected_idx = chunk_idx + 1
        self.ensure_worker()
        self.queue.put_nowait((audio_bytes, chunk_idx))

    async def wait_complete(self):
        # afplay holds the worker until the clip is done, so an empty queue means
        # the last sentence has finished playing
        await self.queue.join()
        self.is_playing = False
        self.expected_idx = 0

    async def _worker(self):
        while True:
            audio_bytes, chunk_idx = await self.queue.get()
            if self.interrupted:
                # the turn was cut while this clip waited in the queue
                self.queue.task_done()
                continue
            self.is_playing = True
            try:
                self._clip_id += 1
                path = os.path.join(self._tmpdir, f'clip_{self._clip_id:04d}.{AUDIO_FORMAT}')
                with open(path, 'wb') as f:
                    f.write(audio_bytes)
                proc = await asyncio.create_subprocess_exec(
                    LOCAL_PLAY_CMD,
                    path,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                # held so barge-in can kill the clip that is sounding right now
                self._proc = proc
                await proc.wait()
            except asyncio.CancelledError:
                self.is_playing = False
                raise
            except Exception as e:
                console.print(f'[dim yellow][[ local playback failed: {e} ]][/]')
            finally:
                self._proc = None
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
        shutil.rmtree(self._tmpdir, ignore_errors=True)
