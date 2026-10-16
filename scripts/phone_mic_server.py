import asyncio
import os
import ssl
import time
from pathlib import Path
from queue import Empty, Queue

from aiohttp import WSMsgType, web
from rich.console import Console

try:
    import numpy
except ImportError:
    numpy = None

console = Console()
BASE_DIR = Path(__file__).resolve().parent.parent
PHONE_MIC_DIR = BASE_DIR / 'peerbots_phone_mic'
HTML_FILE = PHONE_MIC_DIR / 'face.html'


# how long the phone can send nothing before we give up on the current utterance.
# a browser reconnect takes ~1.5s, so this is generous slack rather than a sniff.
PHONE_MIC_IDLE_BAIL = 20.0

# the page ships a chunk roughly every 85ms (4096 samples at the audio context
# rate). anything much longer than that is the link stalling, not the speaker
# going quiet, and it is the difference between "walked out of wifi range" and
# "the microphone is bad".
FRAME_GAP_WARN_S = 0.25


class PhoneMicStream:
    '''
    mimics pyaudio stream read interface so whisper_stt silero vad can read directly
    '''

    def __init__(self, audio_queue):
        self.queue = audio_queue
        self.buffer = bytearray()
        self._closed = False

    def read(self, num_samples, exception_on_overflow=False):
        '''
        block until num_samples is available, matching the pyaudio contract.

        a momentary gap must not be mistaken for end-of-utterance, so wait out a
        generous idle window. but a bail must only end this one read: a phone that
        goes quiet and then starts talking again has to be picked up, so the next
        call keeps waiting instead of latching into a dead state.
        '''
        # 16-bit pcm = 2 bytes per sample
        bytes_needed = num_samples * 2
        idle = 0.0
        while len(self.buffer) < bytes_needed:
            if self._closed:
                return b''
            try:
                chunk = self.queue.get(timeout=0.5)
                self.buffer.extend(chunk)
                idle = 0.0
            except Empty:
                idle += 0.5
                if idle >= PHONE_MIC_IDLE_BAIL:
                    break
            except Exception:
                break
        if len(self.buffer) < bytes_needed:
            # short read: nothing more is arriving right now, so hand back the
            # partial audio so the vad can close the utterance. the next call
            # resumes as soon as the phone sends anything.
            chunk = bytes(self.buffer)
            self.buffer.clear()
            return chunk
        chunk = bytes(self.buffer[:bytes_needed])
        del self.buffer[:bytes_needed]
        return chunk

    def get_read_available(self):
        return len(self.buffer) // 2

    def close(self):
        self._closed = True


class PhoneMicSource:
    '''
    audio source compatible with speech_recognition and listen_audio_silero
    '''

    SAMPLE_RATE = 16000
    SAMPLE_WIDTH = 2

    def __init__(self, audio_queue):
        self.stream = PhoneMicStream(audio_queue)
        self.SAMPLE_RATE = 16000
        self.SAMPLE_WIDTH = 2

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream:
            self.stream.close()


class PhoneMicServer:
    def __init__(self, port=8443, token='', use_tunnel=False):
        self.port = port
        self.token = token or os.getenv('PEERBOTS_TOKEN', '')
        self.use_tunnel = use_tunnel
        # ~2s of 16k pcm. deeper than this and the vad is always chewing on stale
        # audio, so live speech queues up behind it and looks like a dead mic
        self.audio_queue = Queue(maxsize=50)
        self._rms_sum = 0.0
        self._rms_frames = 0
        self._last_frame_time = None
        self._frame_gaps = 0
        self.app = web.Application()
        self.app.router.add_get('/', self._handle_index)
        self.app.router.add_get('/audio', self._handle_audio)
        self.runner = None
        self.site = None
        self.tunnel_proc = None
        self.tunnel_url = None
        self._connected = False
        self._client_ws = None
        # set by get_source(); a phone that connects first must not blow up here
        self.stream = None
        # lets the greeting wait for a browser, since tts now plays on the handset
        self._client_connected = asyncio.Event()

    def _verify_token(self, request):
        if not self.token:
            return True
        return request.query.get('token') == self.token

    async def _handle_index(self, request):
        if not self._verify_token(request):
            raise web.HTTPForbidden(text='invalid access token')
        if not HTML_FILE.exists():
            raise web.HTTPNotFound(text='face.html not found')
        return web.FileResponse(HTML_FILE)

    async def _handle_audio(self, request):
        if not self._verify_token(request):
            raise web.HTTPForbidden(text='invalid access token')

        # only one browser may stream at a time. multiple tabs interleaving pcm
        # into one queue sounds like noise and never transcribes.
        if self._client_ws is not None and not self._client_ws.closed:
            console.print('[dim yellow][[phone-mic]]: another browser already streaming, rejecting this one (close extra tabs)[/]')
            return web.Response(status=409, text='another phone mic is already streaming')

        ws = web.WebSocketResponse(max_msg_size=1024 * 1024, heartbeat=30)
        await ws.prepare(request)
        self._client_ws = ws
        self._connected = True
        # stale audio from a previous session would be prepended to the next utterance
        self._drain_queue()
        # a reconnect is not a gap; only measure arrivals within one session
        self._last_frame_time = None
        self._client_connected.set()
        console.print('[bold green][[phone-mic]]: connected from browser! audio streaming started...[/]')
        received_chunks = 0
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    received_chunks += 1
                    if received_chunks == 1:
                        console.print('[bold green][[phone-mic]]: receiving first audio frames from phone mic[/]')
                    self._note_frame_gap(time.time())
                    self._track_level(msg.data)
                    # drop oldest on overflow so latency stays bounded and we keep
                    # the freshest audio instead of silently discarding new speech
                    try:
                        self.audio_queue.put_nowait(msg.data)
                    except Exception:
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait(msg.data)
                        except Exception:
                            pass
                elif msg.type == WSMsgType.ERROR:
                    console.print(f'[dim yellow][[phone-mic]]: ws error: {ws.exception()}[/]')
        finally:
            self._connected = False
            self._client_ws = None
            self._client_connected.clear()
            # deliberately not draining here: the browser reconnects every few
            # seconds, and wiping the queue mid-utterance truncates the phrase
            # the vad is currently collecting. stale audio is cleared on connect.
            err = None if ws.exception() is None else repr(ws.exception())
            console.print(
                f'[dim yellow][[phone-mic]]: disconnected '
                f'(code={ws.close_code} chunks={received_chunks} err={err})[/]'
            )
        return ws

    def _note_frame_gap(self, now):
        '''
        bursty arrival is invisible otherwise. a stalled link makes the vad end
        an utterance early, which is indistinguishable from the speaker pausing,
        so call out anything long enough to be the network rather than a
        scheduling hiccup.
        '''
        gap = 0.0 if self._last_frame_time is None else now - self._last_frame_time
        self._last_frame_time = now
        if gap >= FRAME_GAP_WARN_S:
            self._frame_gaps += 1
            console.print(
                f'[dim yellow][[phone-mic]] link gap {gap:.2f}s '
                f'({self._frame_gaps} total) - dropped audio, not silence[/]'
            )

    def _track_level(self, data):
        '''
        rolling input level readout, so a silent mic is distinguishable from
        audio arriving but the vad not firing.
        '''
        import array
        usable = data[: len(data) - (len(data) % 2)]
        if not usable:
            return
        # a per-sample python loop on the event loop starves the server and
        # shows up downstream as websocket disconnects
        if numpy is not None:
            vals = numpy.frombuffer(usable, dtype='<i2').astype(numpy.float32)
            vals *= 1.0 / 32768.0
            acc = float(numpy.dot(vals, vals))
            count = len(vals)
        else:
            samples = array.array('h')
            samples.frombytes(usable)
            acc = sum((s / 32768.0) ** 2 for s in samples)
            count = len(samples)
        self._rms_sum += acc
        self._rms_frames += count
        if self._rms_frames >= 16000:
            rms = (self._rms_sum / self._rms_frames) ** 0.5
            bar = '#' * min(30, int(rms * 300))
            console.print(f'[dim cyan][[phone-mic]] input rms {rms:.4f} |{bar}[/]')
            self._rms_sum = 0.0
            self._rms_frames = 0

    def _drain_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Exception:
                break

    def _check_port_free(self):
        '''
        a busy port otherwise surfaces as a bare [[bye!]] with no explanation,
        because the bind failure tears down the process via the signal handler.
        '''
        import socket
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('0.0.0.0', self.port))
        except OSError as e:
            console.print(f'[bold red][[phone-mic]]: port {self.port} is already in use ({e})[/]')
            console.print('[bold red]  another "main.py --phone-mic" is probably still running.[/]')
            console.print('[bold red]  stop it (ctrl-c, or: pkill -f "main.py --phone-mic") or pass --phone-port <other>[/]')
            raise SystemExit(1) from e
        finally:
            probe.close()

    async def start(self):
        self._check_port_free()
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        # when using adb reverse or tunnel, plain HTTP is strongly preferred because
        # browsers block WebSockets over self-signed HTTPS certificates.
        ssl_ctx = None
        cert = PHONE_MIC_DIR / 'cert.pem'
        key = PHONE_MIC_DIR / 'key.pem'
        # only use ssl_context if explicitly not using tunnel and not relying on adb loopback
        use_ssl = (not self.use_tunnel) and os.environ.get('PHONE_MIC_HTTPS', 'false').lower() in ('true', '1')
        if use_ssl and cert.exists() and key.exists():
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(str(cert), str(key))

        self.site = web.TCPSite(self.runner, host='0.0.0.0', port=self.port, ssl_context=ssl_ctx)
        await self.site.start()
        scheme = 'https' if ssl_ctx else 'http'
        token_str = f'?token={self.token}' if self.token else ''
        console.print(f'[bold green][[phone-mic]]: server listening on {scheme}://localhost:{self.port}/{token_str}[/]')

        if self.use_tunnel:
            await self._start_cloudflared_tunnel()
        else:
            console.print('  [bold cyan]For Android Chrome via USB:[/]')
            console.print(f'    1. In terminal run: [bold white]adb reverse tcp:{self.port} tcp:{self.port}[/]')
            console.print(f'    2. On phone Chrome open: [bold white]http://localhost:{self.port}/{token_str}[/]')
            console.print('    3. Tap "Enable microphone"')

    async def _start_cloudflared_tunnel(self):
        cloudflared_bin = PHONE_MIC_DIR / 'tools' / 'cloudflared'
        if not cloudflared_bin.exists():
            console.print(f'[dim yellow][[phone-mic]]: cloudflared binary not found at {cloudflared_bin}[/]')
            return
        token_str = f'?token={self.token}' if self.token else ''
        console.print('[dim white][[phone-mic]]: requesting Cloudflare tunnel...[/]')
        import re
        try:
            self.tunnel_proc = await asyncio.create_subprocess_exec(
                str(cloudflared_bin),
                'tunnel',
                '--no-autoupdate',
                '--url',
                f'http://127.0.0.1:{self.port}',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            # read stderr for tunnel url
            async def _find_url():
                while True:
                    line = await self.tunnel_proc.stderr.readline()
                    if not line:
                        break
                    line_str = line.decode('utf-8', errors='replace')
                    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line_str)
                    if match:
                        self.tunnel_url = match.group(0)
                        console.print(f'[bold green][[phone-mic]]: Public HTTPS URL: {self.tunnel_url}/{token_str}[/]')
                        console.print('  [dim white]Open this URL on any phone (no USB cable or ADB required)[/]')
                        break

            asyncio.create_task(_find_url())
        except Exception as e:
            console.print(f'[dim yellow][[phone-mic]]: could not start cloudflared tunnel: {e}[/]')

    async def stop(self):
        # release any read parked waiting for the phone to come back
        if self.stream is not None:
            self.stream.close()
        if self.tunnel_proc:
            try:
                self.tunnel_proc.terminate()
            except Exception:
                pass
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

    def get_source(self):
        source = PhoneMicSource(self.audio_queue)
        # keep the handle so shutdown can release a read that is blocked on a
        # phone which has stopped sending
        self.stream = source.stream
        return source

    async def send_audio(self, audio_bytes):
        '''
        push a tts chunk down the mic websocket so the phone browser plays it
        out of the handset speaker.

        inputs:
        audio_bytes: wav/mp3 bytes
        outputs:
        sent: boolean, false when no browser is attached
        '''
        ws = self._client_ws
        if ws is None or ws.closed:
            return False
        try:
            await ws.send_bytes(audio_bytes)
            return True
        except Exception:
            return False

    async def wait_for_client(self, timeout=None):
        '''
        block until a browser opens its audio websocket, so a greeting is not
        spoken into an empty room.
        '''
        try:
            await asyncio.wait_for(self._client_connected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
