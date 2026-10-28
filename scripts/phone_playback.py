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

# how long past the duration estimate to keep waiting for the page's ack before
# giving up on it. covers decode plus a slow wifi round trip.
ACK_GRACE_S = 1.5


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
        self.interrupted = False
        # barge-in has to be able to cut the worker's pacing sleep short. setting
        # playback_end_time alone cannot: asyncio.sleep() fixes its deadline when
        # it is called, so a cut chunk kept pacing and the next turn's first clip
        # sat in the queue until the dead chunk's full duration had elapsed.
        self._cut = asyncio.Event()

    def ensure_worker(self):
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())

    def reset_turn(self):
        self.expected_idx = 0
        self.playback_end_time = 0.0
        self.interrupted = False
        # a fresh event per turn. clearing a shared one could race the worker:
        # interrupt() sets it, reset_turn() clears it before the worker is ever
        # scheduled, and the cut chunk gets paced out in full after all. the
        # worker holds the event it started with, so a set can never be lost.
        self._cut = asyncio.Event()
        try:
            self.server.reset_playback_ack()
        except AttributeError:
            pass
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
        # the page reports when handset audio actually stops. decode and wifi both
        # drift, so sleeping out a send-time estimate could reopen the microphone
        # while lulo was still talking, and she would hear herself. the estimate is
        # kept only as the timeout, so a page too old to ack still works.
        remaining = self.playback_end_time - time.time()
        if remaining > 0 or self.is_playing:
            budget = max(0.0, remaining) + ACK_GRACE_S
            acked = False
            try:
                acked = await self.server.wait_playback_ack(budget)
            except AttributeError:
                await asyncio.sleep(max(0.0, remaining))
            if not acked and remaining > 0:
                console.print(
                    '  [dim yellow][[ phone playback ack timed out, fell back to estimate ]][/]'
                )
        self.is_playing = False
        self.expected_idx = 0
        self.playback_end_time = 0.0

    async def interrupt(self):
        '''
        barge-in: cut handset audio now and drop everything still queued.

        the page owns the audio nodes, so stopping is a control frame rather than
        anything this side can do. draining the queue here stops the worker from
        pushing the rest of the turn down the socket after the cut.
        '''
        self.interrupted = True
        self._cut.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except Exception:
                break
        try:
            await self.server.send_control('interrupt')
        except AttributeError:
            pass
        self.is_playing = False
        self.playback_end_time = 0.0

    async def _worker(self):
        while True:
            audio_bytes, duration_s, chunk_idx = await self.queue.get()
            cut = self._cut
            if self.interrupted:
                # the turn was cut while this chunk sat in the queue
                self.queue.task_done()
                continue
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
                    # pace the next chunk behind this one, but wake immediately
                    # if barge-in cuts the turn
                    remaining = max(0.0, self.playback_end_time - time.time())
                    if remaining > 0:
                        try:
                            await asyncio.wait_for(cut.wait(), timeout=remaining)
                        except asyncio.TimeoutError:
                            pass
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
