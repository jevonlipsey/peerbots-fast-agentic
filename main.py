"""
lulo main loop. pure python 3.11, no naoqi, no conda env.

flow per turn:
a) listen for user audio via whisper_stt
b) send an immediate thinking face so the patient sees feedback
c) ask the llm for strict json (speech, emotion, color)
d) forward that json to the peerbots face via send_peerbots_message
"""

import argparse
import asyncio
import os
import signal
import sys
import time

import speech_recognition as sr
from dotenv import load_dotenv
from rich.console import Console

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from whisper_stt import (
    IS_MAC,
    listen_once,
    start_swift_worker,
    get_vad_model,
    prewarm_whisper,
)
from phone_mic_server import PhoneMicServer
from openai_response import (
    connect_mcp,
    start_fresh_session,
    stream_peerbots_reply,
    prewarm_openai,
    start_openai_keepalive,
    stop_openai_keepalive,
    close_client as close_openai_client,
)
from peerbots_client import (
    asend_peerbots_message,
    fire_peerbots_update,
    close_client as close_peerbots_client,
)
from kokoro_manager import (
    ensure_kokoro_ready,
    synthesize_speech,
    stop_kokoro_process,
    close_client as close_kokoro_client,
    DEFAULT_VOICE,
    DEFAULT_SPEED,
)
from droid_client import (
    ais_droid_reachable,
    get_audio_duration_s,
    DroidPlaybackQueue,
    probe_droid_daemon,
    close_client as close_droid_client,
)
from local_playback import LocalPlaybackQueue
from phone_playback import PhoneSpeakerQueue
from arm_controller import ArmController

load_dotenv()
console = Console()

### magic constants, tweak here when running from an ide
### cli flags override these when present
TEXT_MODE = False  # true = type instead of using the mic
MIC_INDEX = None  # microphone index; None = system default, auto-picks a working one
PAUSE_THRESHOLD = float(
    os.environ.get("PAUSE_THRESHOLD", "1.0")
)  # seconds of silence before considering speech finished
USE_DROID_AUDIO = True  # send audio chunks to droid headless daemon
PEERBOTS_DUAL_AUDIO = os.environ.get("PEERBOTS_DUAL_AUDIO", "true").lower() in (
    "true",
    "1",
    "yes",
)
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", DEFAULT_VOICE)
KOKORO_SPEED = float(os.environ.get("KOKORO_SPEED", str(DEFAULT_SPEED)))

### thinking mask, disabled by default to keep conversation snappy
ENABLE_THINKING_FILLER = False
THINK_DELAY_S = 2.0
THINKING_SPEECH = "Hmm, let me think about that..."
THINKING_EMOTION = "Neutral"
THINKING_COLOR = "Yellow"

### greeting spoken once at startup (matches hri dialogue tree)
GREETING_SPEECH = "Hi! I'm Lulo. Nice to meet you! What's your name?"
GREETING_EMOTION = "Happy"
GREETING_COLOR = "Green"

_CURRENT_SWIFT_PROC = None
_PLAYBACK_QUEUE = None
_USE_DROID = False
_ARM = None  # usb serial linear-actuator body, set when --no-arduino is absent

### dynamic voice speed per emotion
EMOTION_SPEED_MAP = {
    "Happy": 1.12,
    "Surprised": 1.10,
    "Neutral": 1.05,
    "Concerned": 0.92,
    "Sad": 0.90,
    "Sleepy": 0.88,
}


def get_voice_speed_for_emotion(emotion, base_speed=KOKORO_SPEED):
    """
    scales playback speed based on lulo emotional state.

    inputs:
    emotion: emotion string (e.g. 'Happy', 'Concerned')
    base_speed: base float speed from config
    outputs:
    adjusted float speed
    """
    if not emotion:
        return base_speed
    mult = EMOTION_SPEED_MAP.get(emotion)
    if mult is not None:
        return round(mult * (base_speed / 1.0), 2)
    return base_speed


def make_nod_callback(loop):
    """
    triggers silent face nod feedback during extended user monologues.

    inputs:
    loop: running asyncio event loop
    outputs:
    callable(seconds)
    """
    nod_faces = {
        10: ("Happy", "White"),
        25: ("Neutral", "White"),
        45: ("Surprised", "White"),
    }

    def _on_nod(seconds):
        face = nod_faces.get(seconds)
        if not face:
            return
        emotion, color = face
        try:
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    asend_peerbots_message("", emotion, color, silent=True)
                )
            )
            console.print(
                f"  [dim cyan]-> [listening nod {seconds}s] {emotion} / {color}[/]"
            )
        except Exception:
            pass

    return _on_nod


SCRIPTED_LINES_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "scripted_lines.json"
)


def _scripted_clauses():
    '''
    the exact tts cache keys an on-script reply will ask for.

    the cache is keyed on the clause text the streaming parser emits, not on the
    whole reply, so prewarming a full dialogue-tree line would never be hit. each
    line is therefore pushed through that same parser to recover its real clause
    boundaries.

    inputs:
    none
    outputs:
    list of (clause_text, emotion) pairs
    '''
    import json

    from openai_response import StreamingReplyParser

    try:
        with open(SCRIPTED_LINES_FILE, encoding="utf-8") as fh:
            lines = json.load(fh).get("lines", [])
    except Exception:
        return []

    clauses = []
    for entry in lines:
        speech = entry.get("speech", "")
        emotion = entry.get("emotion", "Neutral")
        if not speech:
            continue
        # shaped exactly like a real strict-schema reply so the parser splits it
        # the way it will at run time
        payload = json.dumps(
            {
                "emotion": emotion,
                "color": "White",
                "turn_action": "lulo turns to user",
                "speech": speech,
            }
        )
        parser = StreamingReplyParser()
        parts = []
        # feed in small slices so clause flushing behaves as it does on a token
        # stream rather than on one complete string
        for i in range(0, len(payload), 8):
            parts.extend(parser.feed(payload[i : i + 8]))
        parts.extend(parser.finish())
        for clause in parts:
            if clause.strip():
                clauses.append((clause, emotion))
    return clauses


async def prewarm_scripted_lines():
    '''
    synthesize every scripted clause ahead of time, at its live speed.

    the demo follows a known dialogue tree, so each of those replies can be served
    from the in-memory kokoro cache for 0ms instead of ~300ms of synthesis. the
    speed has to match exactly or the lookup misses: the generic prewarm used
    DEFAULT_SPEED while live turns ask for get_voice_speed_for_emotion(...), so
    before this nothing but the greeting ever hit.

    inputs:
    none
    outputs:
    count of clauses cached
    '''
    clauses = _scripted_clauses()
    if not clauses:
        return 0
    results = await asyncio.gather(
        *(
            synthesize_speech(
                clause,
                voice=KOKORO_VOICE,
                speed=get_voice_speed_for_emotion(emotion, KOKORO_SPEED),
            )
            for clause, emotion in clauses
        ),
        return_exceptions=True,
    )
    return sum(1 for r in results if isinstance(r, (bytes, bytearray)) and r)


async def send_greeting():
    # spoken greeting once the whole pipeline is ready
    global _USE_DROID, _PLAYBACK_QUEUE
    if _ARM is not None:
        # let the body greet too
        _ARM.fire_emotion(GREETING_EMOTION)
    try:
        if _USE_DROID and _PLAYBACK_QUEUE:
            # fire peerbots face/speech update and synthesize speech concurrently
            fire_peerbots_update(
                GREETING_SPEECH,
                GREETING_EMOTION,
                GREETING_COLOR,
                silent=not PEERBOTS_DUAL_AUDIO,
            )
            audio_bytes = await synthesize_speech(
                GREETING_SPEECH,
                voice=KOKORO_VOICE,
                # same speed the scripted prewarm cached it at, or the lookup misses
                speed=get_voice_speed_for_emotion(GREETING_EMOTION, KOKORO_SPEED),
            )
            if audio_bytes:
                await _PLAYBACK_QUEUE.enqueue(audio_bytes, chunk_idx=0)
                console.print(f"\n[bold green][[LULO]]:[/] {GREETING_SPEECH}")
                await _PLAYBACK_QUEUE.wait_complete()
                return

        # fallback: peerbots tablet tts
        await asend_peerbots_message(
            GREETING_SPEECH, GREETING_EMOTION, GREETING_COLOR, silent=False
        )
        console.print(f"\n[bold green][[LULO]]:[/] {GREETING_SPEECH}")
    except Exception as e:
        console.print(f"[bold red][[ greeting failed: {e} ]][/]")


async def _delayed_think(delay):
    # latency mask: stays silent unless the llm is slower than delay
    await asyncio.sleep(delay)
    try:
        await asend_peerbots_message(THINKING_SPEECH, THINKING_EMOTION, THINKING_COLOR)
    except Exception as e:
        console.print(f"[dim yellow][[ thinking mask failed: {e} ]][/]")


async def handle_turn(user_text, tools_list, tool_router):
    # streaming turn: early emotion -> peerbots face, sentence -> kokoro -> droid queue
    global _USE_DROID, _PLAYBACK_QUEUE
    console.print(f"\n[bold cyan][[USER]]:[/] {user_text}")

    think_task = (
        asyncio.create_task(_delayed_think(THINK_DELAY_S))
        if ENABLE_THINKING_FILLER
        else None
    )

    turn_start = time.time()
    if _USE_DROID and _PLAYBACK_QUEUE:
        _PLAYBACK_QUEUE.reset_turn()

    face_updated = False
    current_emotion = "Neutral"
    tts_tasks = []
    spoken_sentences = []
    first_audio_sent = False
    first_audio_time = None
    first_clause_time = None
    tts1_ms = 0
    final_reply = None
    metrics = {}
    chunk_order_events = {0: asyncio.Event()}
    chunk_order_events[0].set()

    try:
        async for event in stream_peerbots_reply(user_text, tools_list, tool_router):
            # cancel thinking filler as soon as the first stream event arrives
            if think_task and not think_task.done():
                think_task.cancel()

            ev_type = event["type"]
            if ev_type == "face" and not face_updated:
                face_updated = True
                # update peerbots face immediately without waiting for tts
                emotion = event["emotion"]
                color = event["color"]
                current_emotion = emotion
                if _ARM is not None:
                    # physically react to lulo's emotion the moment the face changes
                    _ARM.fire_emotion(emotion)
                if _USE_DROID:
                    # fire the face the moment the emotion token lands, in dual
                    # audio mode too. the only face update in that mode used to be
                    # the one carrying the full speech text at the very end of the
                    # turn, and the peerbots cloud can take 4-5s to deliver it, so
                    # lulo would speak a Concerned/Red escalation line while still
                    # wearing the previous face. this update carries no speech, so
                    # it cannot duplicate audio; the later send still drives the
                    # mouth for lip-sync.
                    fire_peerbots_update("", emotion, color, silent=True)
                console.print(f"  [dim white]-> [face] {emotion} / {color}[/]")

            elif ev_type == "sentence":
                sent = event["text"]
                chunk_idx = event.get("idx", len(spoken_sentences))
                spoken_sentences.append(sent)
                if chunk_idx == 0:
                    first_clause_time = time.time()

                if _USE_DROID and _PLAYBACK_QUEUE:
                    # concurrently synthesize this sentence with kokoro and stream into playback queue
                    async def _synthesize_and_enqueue(sentence_text, idx):
                        nonlocal first_audio_sent, first_audio_time, tts1_ms
                        s_start = time.time()
                        target_speed = get_voice_speed_for_emotion(
                            current_emotion, KOKORO_SPEED
                        )
                        audio_data = await synthesize_speech(
                            sentence_text, voice=KOKORO_VOICE, speed=target_speed
                        )
                        s_ms = int((time.time() - s_start) * 1000)
                        if idx == 0:
                            tts1_ms = s_ms
                        if idx not in chunk_order_events:
                            chunk_order_events[idx] = asyncio.Event()
                        try:
                            await chunk_order_events[idx].wait()
                            if audio_data:
                                if not first_audio_sent:
                                    first_audio_sent = True
                                    first_audio_time = time.time()
                                await _PLAYBACK_QUEUE.enqueue(audio_data, chunk_idx=idx)
                        finally:
                            next_idx = idx + 1
                            if next_idx not in chunk_order_events:
                                chunk_order_events[next_idx] = asyncio.Event()
                            chunk_order_events[next_idx].set()

                    task = asyncio.create_task(_synthesize_and_enqueue(sent, chunk_idx))
                    tts_tasks.append(task)

            elif ev_type == "final":
                final_reply = event["reply"]
                metrics = event.get("metrics", {})
                if _USE_DROID and PEERBOTS_DUAL_AUDIO:
                    full_text = final_reply.get("speech", "") or " ".join(
                        spoken_sentences
                    )
                    # send speech to peerbots to animate mouth; browser audio is muted on android
                    fire_peerbots_update(
                        full_text,
                        final_reply.get("emotion", current_emotion),
                        final_reply.get("color", "White"),
                        silent=False,
                    )

    finally:
        if think_task:
            think_task.cancel()
            try:
                await think_task
            except (asyncio.CancelledError, Exception):
                pass

    if tts_tasks:
        await asyncio.gather(*tts_tasks, return_exceptions=True)

    if _USE_DROID and _PLAYBACK_QUEUE and spoken_sentences:
        # wait for droid audio queue to finish playing completely before re-opening mic
        await _PLAYBACK_QUEUE.wait_complete()
    elif not _USE_DROID and final_reply:
        # fallback: tablet tts handles speech and face together
        try:
            await asend_peerbots_message(
                final_reply["speech"],
                final_reply["emotion"],
                final_reply["color"],
                silent=False,
            )
        except Exception as e:
            console.print(f"[bold red][[ peerbots send failed: {e} ]][/]")

    total_turn_s = time.time() - turn_start
    full_speech = (
        " ".join(spoken_sentences)
        if spoken_sentences
        else (final_reply.get("speech", "") if final_reply else "")
    )
    turn_action_disp = (
        final_reply.get("turn_action", "lulo turns to user")
        if final_reply
        else "lulo turns to user"
    )
    console.print(f"\n[bold magenta][[GAZE/TURN]]:[/] {turn_action_disp}")
    console.print(f"[bold green][[LULO]]:[/] {full_speech}")

    emotion_disp = final_reply.get("emotion", "Neutral") if final_reply else "Neutral"
    color_disp = final_reply.get("color", "White") if final_reply else "White"

    if first_audio_time:
        ttfa_s = first_audio_time - turn_start
        ttft_ms = metrics.get("ttft_ms", 0)
        clause1_ms = (
            int((first_clause_time - turn_start) * 1000) if first_clause_time else 0
        )
        face_ms = metrics.get("face_ms", 0)
        console.print(
            f"  [bright_yellow]-> [Metrics] TTFA: {ttfa_s:.2f}s "
            f"(TTFT: {ttft_ms}ms | Clause1: {clause1_ms}ms | TTS1: {tts1_ms}ms | Face: {face_ms}ms) "
            f"| Total: {total_turn_s:.2f}s | {emotion_disp} / {color_disp} | Gaze: {turn_action_disp}[/]"
        )
    else:
        console.print(
            f"  [bright_yellow]-> [Metrics] Total: {total_turn_s:.2f}s | {emotion_disp} / {color_disp} | Gaze: {turn_action_disp}[/]"
        )


def list_mics():
    '''
    list every audio device with its input channel count.

    pyaudio enumerates all alsa devices, including the many pseudo devices that
    have no capture channels at all. without the channel count an output-only
    device looks like a perfectly good microphone, and picking it fails with an
    opaque 'channelCount <= maxChans' error deep in alsa.
    '''
    try:
        import pyaudio
    except Exception as exc:
        console.print(f'[bold red][[ could not enumerate audio devices: {exc} ]][/]')
        return
    pa = pyaudio.PyAudio()
    try:
        try:
            default_index = pa.get_default_input_device_info()['index']
        except Exception:
            default_index = None
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            channels = int(info.get('maxInputChannels', 0))
            name = info.get('name', 'unknown')
            if channels > 0:
                mark = '->' if index == default_index else '  '
                # a device can advertise input channels and still refuse to open at
                # 16k mono (alsa 'default'/'pulse' aliases bound to a playback card),
                # so probe it rather than trusting the channel count
                ok, _ = _probe_mic(index)
                note = '' if ok else '  [cannot open at 16k mono]'
                line = f'{mark} [{index}] {name} ({channels} input channels){note}'
                # markup off, the bracketed index otherwise parses as a rich tag
                console.print(line, style=None if ok else 'dim', markup=False)
            else:
                line = f'   [{index}] {name} (output only)'
                console.print(line, style='dim', markup=False)
        console.print('  -> = system default. choose one with: python main.py --mic-index N', style='dim')
    finally:
        pa.terminate()


def _probe_mic(mic_index):
    '''
    open a throwaway 16k mono input stream to confirm the device can capture.

    speech_recognition's Microphone.__enter__ swallows the open error and leaves
    stream None, then __exit__ calls self.stream.close() with no None guard. the
    real error is therefore buried under a bogus 'NoneType has no attribute
    close'. checking here instead lets us report what actually went wrong.

    inputs:
    mic_index: pyaudio device index, or None for the system default
    outputs:
    ok: True when a stream opened
    detail: error text when ok is False, otherwise None
    '''
    try:
        import pyaudio
    except Exception as exc:
        return False, f'pyaudio unavailable ({exc})'
    pa = pyaudio.PyAudio()
    stream = None
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=mic_index,
            frames_per_buffer=1024,
        )
        return True, None
    except Exception as exc:
        return False, str(exc)
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        pa.terminate()


def _first_working_mic_index():
    '''
    probe every capture-capable device and return the first that really opens at
    16k mono.

    pyaudio happily lists alsa plumbing aliases ('default', 'pulse', 'sysdefault')
    as having many input channels, but opening them fails with 'invalid number of
    channels' when they resolve to a playback-only card. prefer real hardware, then
    pulse/pipewire, then the aliases, so a bad default does not look like a dead mic.

    inputs:
    none
    outputs:
    index: first device that opened, or None if none could
    '''
    try:
        import pyaudio
    except Exception:
        return None
    pa = pyaudio.PyAudio()
    try:
        candidates = []
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            if int(info.get('maxInputChannels', 0)) <= 0:
                continue
            candidates.append((index, str(info.get('name', ''))))

        def _rank(item):
            name = item[1].lower()
            if name in ('default', 'sysdefault'):
                return 3
            if 'pulse' in name or 'pipewire' in name or 'dmix' in name:
                return 2
            if 'hdmi' in name:
                return 1
            return 0

        for index, _name in sorted(candidates, key=_rank):
            ok, _detail = _probe_mic(index)
            if ok:
                return index
    finally:
        pa.terminate()
    return None


try:
    _ExceptionGroupTypes = (BaseExceptionGroup,)
except NameError:
    _ExceptionGroupTypes = ()

_CURRENT_SWIFT_PROC = None


def _signal_handler(signum, frame):
    # instant clean shutdown on ctrl-c or sigterm without threadpool deadlock
    global _CURRENT_SWIFT_PROC, _PLAYBACK_QUEUE
    if _CURRENT_SWIFT_PROC is not None:
        try:
            _CURRENT_SWIFT_PROC.kill()
        except Exception:
            pass
    stop_kokoro_process()
    sys.stdout.write("\n\033[2m[[bye!]]\033[0m\n")
    sys.stdout.flush()
    os._exit(0)


async def mic_loop(
    mic_index,
    tools_list,
    tool_router,
    pause_threshold=PAUSE_THRESHOLD,
    use_silero=True,
):
    # blocking mic loop, runs until ctrl-c
    global _CURRENT_SWIFT_PROC
    swift_proc = None
    if IS_MAC:
        try:
            swift_proc = start_swift_worker()
            _CURRENT_SWIFT_PROC = swift_proc
        except Exception as e:
            console.print(
                f"[dim yellow][[ coreml worker failed, whisper fallback: {e} ]]"
            )
    if swift_proc is None:
        console.print("[dim white][[stt]]: local whisper (base.en) active, warming model[/]")
        prewarm_whisper()

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = pause_threshold
    recognizer.non_speaking_duration = 0.30

    vad_available = use_silero and (get_vad_model() is not None)
    if vad_available:
        console.print(
            f"[dim white][[stt]]: neural silero-vad endpointing active (pause_threshold={pause_threshold}s)[/]"
        )
    else:
        console.print(
            f"[dim white][[stt]]: legacy vad active (pause_threshold={pause_threshold}s)[/]"
        )

    loop = asyncio.get_running_loop()
    nod_cb = make_nod_callback(loop) if _USE_DROID else None

    # probe before entering the context manager. a failed open leaves stream as
    # None and speech_recognition's __exit__ then raises AttributeError on top of
    # the real error, so the failure has to be caught out here instead
    ok, detail = _probe_mic(mic_index)
    if not ok:
        # the configured device (or the system default) cannot capture at 16k mono,
        # so try to find one that can. otherwise an output-only card or a stale
        # alsa alias looks exactly like a dead microphone.
        console.print(f"[dim yellow][[ mic {mic_index} unavailable: {detail} ]][/]")
        alt = _first_working_mic_index()
        if alt is not None and alt != mic_index:
            console.print(
                f"[dim white][[ auto-selected working microphone index {alt} ]][/]"
            )
            mic_index = alt
            ok, detail = _probe_mic(mic_index)
    if not ok:
        console.print(f"[bold red][[ fatal: could not open microphone: {detail} ]][/]")
        console.print("[dim white][[ audio devices on this machine ]][/]")
        list_mics()
        sys.exit(1)

    with sr.Microphone(device_index=mic_index, sample_rate=16000) as source:
        console.print("[dim white][[ calibrating for room noise... ]][/]")
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        await send_greeting()
        console.print(
            "\n[bold dark_orange][[LISTENING - speak to lulo (ctrl-c to quit)]][/]"
        )
        while True:
            text = await asyncio.to_thread(
                listen_once,
                recognizer,
                source,
                swift_proc,
                timeout=None,
                phrase_limit=90,
                use_silero=vad_available,
                nod_callback=nod_cb,
                pause_threshold=pause_threshold,
            )
            if IS_MAC and swift_proc is not None and swift_proc.poll() is not None:
                console.print("[dim white][[ reviving crashed stt worker... ]][/]")
                try:
                    swift_proc = start_swift_worker()
                    _CURRENT_SWIFT_PROC = swift_proc
                except Exception as e:
                    console.print(
                        f"[dim yellow][[ stt worker failed, whisper fallback: {e} ]]"
                    )
                    swift_proc = None
            if text:
                await handle_turn(text, tools_list, tool_router)
                # discard buffered audio frames recorded during robot speaking to avoid echo loops
                try:
                    # source.stream is a MicrophoneStream wrapper; the raw pyaudio
                    # stream lives one level down and is what exposes read-available
                    raw_stream = getattr(
                        getattr(source, "stream", None), "pyaudio_stream", None
                    )
                    if raw_stream is not None:
                        available = raw_stream.get_read_available()
                        if available > 0:
                            raw_stream.read(available, exception_on_overflow=False)
                except Exception:
                    pass
                console.print("\n[bold dark_orange][[LISTENING]][/]")


async def phone_mic_loop(
    tools_list,
    tool_router,
    port=8443,
    token="",
    use_tunnel=False,
    pause_threshold=PAUSE_THRESHOLD,
    use_silero=True,
    phone_audio=False,
):
    # stream audio from phone browser over websocket
    global _CURRENT_SWIFT_PROC, _PLAYBACK_QUEUE
    swift_proc = None
    if IS_MAC:
        try:
            swift_proc = start_swift_worker()
            _CURRENT_SWIFT_PROC = swift_proc
        except Exception as e:
            console.print(
                f"[dim yellow][[ coreml worker failed, whisper fallback: {e} ]]"
            )
    if swift_proc is None:
        console.print("[dim white][[stt]]: local whisper (base.en) active, warming model[/]")
        prewarm_whisper()

    server = PhoneMicServer(port=port, token=token, use_tunnel=use_tunnel)
    await server.start()

    if phone_audio:
        # tts rides back down the mic socket and plays on the handset speaker
        _PLAYBACK_QUEUE = PhoneSpeakerQueue(server)
        await _PLAYBACK_QUEUE.start()

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = pause_threshold
    recognizer.non_speaking_duration = 0.30

    vad_available = use_silero and (get_vad_model() is not None)
    if vad_available:
        console.print(
            f"[dim white][[stt]]: neural silero-vad endpointing active (pause_threshold={pause_threshold}s)[/]"
        )
    else:
        console.print(
            f"[dim white][[stt]]: legacy vad active (pause_threshold={pause_threshold}s)[/]"
        )

    loop = asyncio.get_running_loop()
    nod_cb = make_nod_callback(loop) if _USE_DROID else None

    source = server.get_source()
    if phone_audio:
        # speech comes out of the handset, so there is no point greeting an empty room
        console.print(
            "[dim white][[ waiting for the phone browser to open the page... ]][/]"
        )
        await server.wait_for_client(timeout=180)
    await send_greeting()
    # the phone starts streaming the moment it connects, so anything said during
    # the greeting sits in the queue and gets transcribed as if the robot asked it
    server._drain_queue()
    console.print(
        "\n[bold dark_orange][[PHONE MIC READY - open face URL and speak (ctrl-c to quit)]][/]"
    )

    try:
        while True:
            text = await asyncio.to_thread(
                listen_once,
                recognizer,
                source,
                swift_proc,
                timeout=None,
                phrase_limit=90,
                use_silero=vad_available,
                nod_callback=nod_cb,
                pause_threshold=pause_threshold,
            )
            if IS_MAC and swift_proc is not None and swift_proc.poll() is not None:
                console.print("[dim white][[ reviving crashed stt worker... ]][/]")
                try:
                    swift_proc = start_swift_worker()
                    _CURRENT_SWIFT_PROC = swift_proc
                except Exception as e:
                    console.print(
                        f"[dim yellow][[ stt worker failed, whisper fallback: {e} ]]"
                    )
                    swift_proc = None
            if text:
                await handle_turn(text, tools_list, tool_router)
                # clear audio queue of any audio recorded during robot speech
                while not server.audio_queue.empty():
                    try:
                        server.audio_queue.get_nowait()
                    except Exception:
                        break
                console.print("\n[bold dark_orange][[LISTENING (PHONE MIC)]][/]")
    finally:
        await server.stop()


async def text_loop(tools_list, tool_router):
    # typed fallback when there is no mic, or for quick lab testing
    await send_greeting()
    console.print("\n[bold dark_orange][[TEXT MODE - type to lulo (quit to exit)]][/]")
    while True:
        text = await asyncio.to_thread(input, "you: ")
        text = text.strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if text:
            await handle_turn(text, tools_list, tool_router)


async def amain(
    use_text,
    mic_index,
    pause_threshold=PAUSE_THRESHOLD,
    enable_droid=USE_DROID_AUDIO,
    use_silero=True,
    enable_dual_audio=PEERBOTS_DUAL_AUDIO,
    use_phone_mic=False,
    phone_mic_port=8443,
    phone_mic_token="",
    use_tunnel=False,
    local_audio=False,
    phone_audio=False,
    enable_arduino=True,
    fresh_history=True,
):
    global _USE_DROID, _PLAYBACK_QUEUE, PEERBOTS_DUAL_AUDIO, _ARM
    PEERBOTS_DUAL_AUDIO = enable_dual_audio

    if fresh_history:
        archived = start_fresh_session()
        if archived:
            console.print(
                f"[dim white][[ previous session archived to {os.path.basename(archived)}, "
                f"starting fresh ]][/]"
            )
    else:
        console.print("[dim yellow][[ keeping previous session history ]][/]")
    missing = [
        k
        for k in ("OPENAI_API_KEY", "PEERBOTS_API_KEY", "PEERBOTS_USERNAME")
        if not os.environ.get(k, "")
    ]
    if missing:
        console.print(f"[bold red][[ missing in .env: {', '.join(missing)} ]]")
        sys.exit(1)

    placeholders = [
        k
        for k in ("PEERBOTS_API_KEY", "PEERBOTS_USERNAME")
        if (os.environ.get(k, "") or "").lower().startswith("your-")
    ]
    if placeholders:
        console.print(
            f"[bold yellow][[ .env still has placeholder values for "
            f"{', '.join(placeholders)}: the peerbots face and tablet voice will "
            f"silently do nothing ]][/]"
        )

    # initialize kokoro tts & the audio sink if requested
    audio_sink = 'tablet'
    if enable_droid:
        console.print("[dim white][[ checking kokoro tts & droid audio daemon... ]][/]")
        koko_ok = await ensure_kokoro_ready()
        if koko_ok:
            cached = await prewarm_scripted_lines()
            if cached:
                console.print(
                    f"[dim white][[ cached {cached} scripted dialogue-tree clauses "
                    f"(on-script replies skip tts) ]][/]"
                )
            if phone_audio:
                if not use_phone_mic:
                    console.print("[bold red][[ --phone-audio needs --phone-mic ]][/]")
                else:
                    audio_sink = 'phone'
            elif local_audio:
                audio_sink = 'mac'
            elif await ais_droid_reachable():
                audio_sink = 'droid'

    if audio_sink == 'tablet':
        if enable_droid:
            reasons = []
            if not koko_ok:
                reasons.append("kokoro offline")
            elif not (phone_audio or local_audio):
                reasons.append("droid unreachable")
            console.print(
                f"[dim yellow][[ audio fallback ]]: {', '.join(reasons) or 'no sink available'}, using Peerbots tablet audio[/]"
            )
        _USE_DROID = False
    else:
        _USE_DROID = True
        mode_str = " + tablet lip-sync" if PEERBOTS_DUAL_AUDIO else " (silent face)"
        if audio_sink == 'droid':
            _PLAYBACK_QUEUE = DroidPlaybackQueue()
            await _PLAYBACK_QUEUE.start()
            console.print(
                f"[bold green][[audio]]: streaming Kokoro MPS -> Droid daemon (100.119.180.97){mode_str}[/]"
            )
            try:
                probe_res = await probe_droid_daemon()
                probe_summary = ", ".join(
                    f"{k}:{v}" for k, v in probe_res.items() if isinstance(v, int)
                )
                if probe_summary:
                    console.print(
                        f"  [dim white]-> [droid endpoints] {probe_summary}[/]"
                    )
            except Exception:
                pass
        elif audio_sink == 'mac':
            _PLAYBACK_QUEUE = LocalPlaybackQueue()
            await _PLAYBACK_QUEUE.start()
            console.print(
                f"[bold green][[audio]]: streaming Kokoro MPS -> local speakers{mode_str}[/]"
            )
        else:
            # the queue is built in phone_mic_loop once the server exists
            console.print(
                f"[bold green][[audio]]: streaming Kokoro MPS -> phone browser speakers{mode_str}[/]"
            )

    # bring up the physical body (linear actuator over usb serial) if requested
    if enable_arduino:
        _ARM = ArmController()
        await _ARM.start()

    # prewarm openai connection pool and start keepalive
    console.print("[dim white][[ pre-warming openai connection pool... ]][/]")
    await prewarm_openai()
    start_openai_keepalive()

    # boot mcp servers once, tools stay live for the whole session
    from contextlib import AsyncExitStack

    try:
        async with AsyncExitStack() as stack:
            tools_list, tool_router = await connect_mcp(stack)
            if use_text:
                await text_loop(tools_list, tool_router)
            elif use_phone_mic:
                await phone_mic_loop(
                    tools_list,
                    tool_router,
                    port=phone_mic_port,
                    token=phone_mic_token,
                    use_tunnel=use_tunnel,
                    pause_threshold=pause_threshold,
                    use_silero=use_silero,
                    phone_audio=phone_audio,
                )
            else:
                await mic_loop(
                    mic_index,
                    tools_list,
                    tool_router,
                    pause_threshold=pause_threshold,
                    use_silero=use_silero,
                )
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        stop_openai_keepalive()
        if _PLAYBACK_QUEUE:
            await _PLAYBACK_QUEUE.stop()
        if _ARM is not None:
            await _ARM.stop()
        stop_kokoro_process()
        await close_openai_client()
        await close_peerbots_client()
        await close_kokoro_client()
        await close_droid_client()


def main():
    global KOKORO_VOICE
    parser = argparse.ArgumentParser(description="lulo peerbots companion")
    parser.add_argument(
        "--text",
        action="store_true",
        default=TEXT_MODE,
        help="type instead of using the mic",
    )
    parser.add_argument(
        "--mic-index",
        type=int,
        default=MIC_INDEX,
        help="microphone index, see --list-mics",
    )
    parser.add_argument(
        "--pause-threshold",
        type=float,
        default=PAUSE_THRESHOLD,
        help="seconds of silence before speech is considered done (default: 1.0)",
    )
    parser.add_argument(
        "--legacy-vad",
        action="store_true",
        help="disable silero-vad neural endpointing and use legacy energy-based threshold",
    )
    parser.add_argument(
        "--no-droid",
        action="store_true",
        help="disable kokoro + droid audio and use peerbots tablet tts",
    )
    parser.add_argument(
        "--local-audio",
        action="store_true",
        help="play kokoro tts through this machine's speakers instead of the droid daemon",
    )
    parser.add_argument(
        "--phone-audio",
        action="store_true",
        help="stream kokoro tts to the phone browser and play it on the handset speaker (needs --phone-mic)",
    )
    parser.add_argument(
        "--no-dual-audio",
        action="store_true",
        help="disable tablet lip-syncing speech and send silent face-only updates to peerbots",
    )
    parser.add_argument(
        "--phone-mic",
        action="store_true",
        help="stream audio from phone browser microphone over websocket",
    )
    parser.add_argument(
        "--phone-port",
        type=int,
        default=int(os.environ.get("PHONE_MIC_PORT", "8443")),
        help="port for phone microphone server (default: 8443)",
    )
    parser.add_argument(
        "--phone-token",
        type=str,
        default=os.environ.get("PEERBOTS_TOKEN", ""),
        help="optional token query parameter for phone mic access",
    )
    parser.add_argument(
        "--phone-tunnel",
        action="store_true",
        help="automatically launch a public Cloudflare HTTPS tunnel for phone mic (no USB needed)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=KOKORO_VOICE,
        help="kokoro voice or mixture (default: af_heart:0.6+af_bella:0.4)",
    )
    parser.add_argument(
        "--list-mics", action="store_true", help="print all microphones and exit"
    )
    parser.add_argument(
        "--keep-history",
        action="store_true",
        help="carry the previous session's conversation over instead of starting fresh",
    )
    parser.add_argument(
        "--no-arduino",
        action="store_true",
        help="disable the usb serial linear-actuator body",
    )
    parser.add_argument(
        "--arm-port",
        type=str,
        default=os.environ.get("ARDUINO_PORT", ""),
        help="arduino serial port path or index (e.g. 0, /dev/cu.usbmodem1101)",
    )
    parser.add_argument(
        "--arduino-port",
        type=str,
        dest="arm_port",
        help="alias for --arm-port (path or index)",
    )
    parser.add_argument(
        "--list-arm-ports",
        action="store_true",
        help="print serial ports for the actuator body and exit",
    )
    parser.add_argument(
        "--list-arduino-ports",
        action="store_true",
        dest="list_arm_ports",
        help="alias for --list-arm-ports",
    )
    args = parser.parse_args()
    if args.list_mics:
        list_mics()
        return
    if args.list_arm_ports:
        from arm_controller import find_arduino_port, list_serial_ports

        list_serial_ports()
        console.print(f"autodetect would pick: {find_arduino_port()}", markup=False)
        return

    if args.arm_port:
        from arm_controller import resolve_arduino_port
        resolved = resolve_arduino_port(args.arm_port)
        if resolved:
            os.environ["ARDUINO_PORT"] = resolved

    if args.voice:
        KOKORO_VOICE = args.voice

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    try:
        asyncio.run(
            amain(
                use_text=args.text,
                mic_index=args.mic_index,
                pause_threshold=args.pause_threshold,
                enable_droid=not args.no_droid,
                use_silero=not args.legacy_vad,
                enable_dual_audio=not args.no_dual_audio,
                use_phone_mic=args.phone_mic,
                phone_mic_port=args.phone_port,
                phone_mic_token=args.phone_token,
                use_tunnel=args.phone_tunnel,
                local_audio=args.local_audio,
                phone_audio=args.phone_audio,
                enable_arduino=not args.no_arduino,
                fresh_history=not args.keep_history,
            )
        )
    except (KeyboardInterrupt, *_ExceptionGroupTypes):
        _signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
