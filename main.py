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

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from whisper_stt import IS_MAC, listen_once, start_swift_worker
from openai_response import connect_mcp, stream_peerbots_reply
from peerbots_client import asend_peerbots_message
from kokoro_manager import (
    ensure_kokoro_ready,
    synthesize_speech,
    stop_kokoro_process,
    DEFAULT_VOICE,
    DEFAULT_SPEED,
)
from droid_client import (
    ais_droid_reachable,
    get_audio_duration_s,
    DroidPlaybackQueue,
)

load_dotenv()
console = Console()

### magic constants, tweak here when running from an ide
### cli flags override these when present
TEXT_MODE = False  # true = type instead of using the mic
MIC_INDEX = 2  # microphone index, run with --list-mics to find yours
PAUSE_THRESHOLD = 1.2  # seconds of silence before considering speech finished (patient friendly)
USE_DROID_AUDIO = True  # send audio chunks to droid headless daemon
KOKORO_VOICE = os.environ.get('KOKORO_VOICE', DEFAULT_VOICE)
KOKORO_SPEED = float(os.environ.get('KOKORO_SPEED', str(DEFAULT_SPEED)))

### thinking mask, disabled by default to keep conversation snappy
ENABLE_THINKING_FILLER = False
THINK_DELAY_S = 2.0
THINKING_SPEECH = 'Hmm, let me think about that...'
THINKING_EMOTION = 'Neutral'
THINKING_COLOR = 'Yellow'

### greeting spoken once at startup
GREETING_SPEECH = "Hi! I'm Lulo, your physical therapy buddy! Are you ready to exercise with me?"
GREETING_EMOTION = 'Happy'
GREETING_COLOR = 'Green'

_CURRENT_SWIFT_PROC = None
_PLAYBACK_QUEUE = None
_USE_DROID = False


async def send_greeting():
    # spoken greeting once the whole pipeline is ready
    global _USE_DROID, _PLAYBACK_QUEUE
    try:
        if _USE_DROID and _PLAYBACK_QUEUE:
            # update peerbots face silently (no tablet audio) and speak via droid
            face_task = asend_peerbots_message(GREETING_SPEECH, GREETING_EMOTION, GREETING_COLOR, silent=True)
            audio_bytes = await synthesize_speech(GREETING_SPEECH, voice=KOKORO_VOICE, speed=KOKORO_SPEED)
            if audio_bytes:
                await face_task
                await _PLAYBACK_QUEUE.enqueue(audio_bytes)
                console.print(f'\n[bold green][[LULO]]:[/] {GREETING_SPEECH}')
                await _PLAYBACK_QUEUE.wait_complete()
                return

        # fallback: peerbots tablet tts
        await asend_peerbots_message(GREETING_SPEECH, GREETING_EMOTION, GREETING_COLOR, silent=False)
        console.print(f'\n[bold green][[LULO]]:[/] {GREETING_SPEECH}')
    except Exception as e:
        console.print(f'[bold red][[ greeting failed: {e} ]][/]')


async def _delayed_think(delay):
    # latency mask: stays silent unless the llm is slower than delay
    await asyncio.sleep(delay)
    try:
        await asend_peerbots_message(THINKING_SPEECH, THINKING_EMOTION, THINKING_COLOR)
    except Exception as e:
        console.print(f'[dim yellow][[ thinking mask failed: {e} ]][/]')


async def handle_turn(user_text, tools_list, tool_router):
    # streaming turn: early emotion -> peerbots face, sentence -> kokoro -> droid queue
    global _USE_DROID, _PLAYBACK_QUEUE
    console.print(f'\n[bold cyan][[USER]]:[/] {user_text}')

    think_task = (
        asyncio.create_task(_delayed_think(THINK_DELAY_S))
        if ENABLE_THINKING_FILLER
        else None
    )

    turn_start = time.time()
    face_updated = False
    tts_tasks = []
    spoken_sentences = []
    first_audio_sent = False
    first_audio_time = None
    final_reply = None

    try:
        async for event in stream_peerbots_reply(user_text, tools_list, tool_router):
            # cancel thinking filler as soon as the first stream event arrives
            if think_task and not think_task.done():
                think_task.cancel()

            ev_type = event['type']
            if ev_type == 'face' and not face_updated:
                face_updated = True
                # update peerbots face immediately without waiting for tts
                emotion = event['emotion']
                color = event['color']
                if _USE_DROID:
                    # silent update: sets face expression & halo glow with 0 volume
                    asyncio.create_task(asend_peerbots_message('', emotion, color, silent=True))
                console.print(f'  [dim white]-> [face] {emotion} / {color}[/]')

            elif ev_type == 'sentence':
                sent = event['text']
                spoken_sentences.append(sent)

                if _USE_DROID and _PLAYBACK_QUEUE:
                    # concurrently synthesize this sentence with kokoro and stream into playback queue
                    async def _synthesize_and_enqueue(sentence_text):
                        nonlocal first_audio_sent, first_audio_time
                        s_start = time.time()
                        audio_data = await synthesize_speech(sentence_text, voice=KOKORO_VOICE, speed=KOKORO_SPEED)
                        s_ms = int((time.time() - s_start) * 1000)
                        if audio_data:
                            if not first_audio_sent:
                                first_audio_sent = True
                                first_audio_time = time.time()
                            await _PLAYBACK_QUEUE.enqueue(audio_data)

                    task = asyncio.create_task(_synthesize_and_enqueue(sent))
                    tts_tasks.append(task)

            elif ev_type == 'final':
                final_reply = event['reply']
                metrics = event['metrics']

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
                final_reply['speech'], final_reply['emotion'], final_reply['color'], silent=False
            )
        except Exception as e:
            console.print(f'[bold red][[ peerbots send failed: {e} ]][/]')

    total_turn_s = time.time() - turn_start
    full_speech = ' '.join(spoken_sentences) if spoken_sentences else (final_reply.get('speech', '') if final_reply else '')
    console.print(f'\n[bold green][[LULO]]:[/] {full_speech}')

    latency_str = f'Total: {total_turn_s:.2f}s'
    if first_audio_time:
        ttfa_s = first_audio_time - turn_start
        latency_str = f'TTFA: {ttfa_s:.2f}s | {latency_str}'
    emotion_disp = final_reply.get('emotion', 'Neutral') if final_reply else 'Neutral'
    color_disp = final_reply.get('color', 'Light Blue') if final_reply else 'Light Blue'
    console.print(f'  [bright_yellow]-> [Metrics] {latency_str} | {emotion_disp} / {color_disp}[/]')


def list_mics():
    # print every input device so the user can pick --mic-index
    microphones = sr.Microphone.list_microphone_names()
    for index, name in enumerate(microphones):
        print(f'Microphone with index {index} and name "{name}" found')


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


async def mic_loop(mic_index, tools_list, tool_router, pause_threshold=PAUSE_THRESHOLD):
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
        console.print("[dim white][[stt]]: python whisper fallback ready[/]")

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = pause_threshold

    with sr.Microphone(device_index=mic_index) as source:
        if getattr(source, "stream", None) is None:
            console.print(
                "[bold red][[ fatal: could not open microphone, run --list-mics to find yours ]]"
            )
            sys.exit(1)
        console.print("[dim white][[ calibrating for room noise... ]][/]")
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        await send_greeting()
        console.print(
            "\n[bold dark_orange][[LISTENING - speak to lulo (ctrl-c to quit)]][/]"
        )
        while True:
            text = await asyncio.to_thread(listen_once, recognizer, source, swift_proc)
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
                    if hasattr(source, 'stream') and source.stream:
                        available = source.stream.get_read_available()
                        if available > 0:
                            source.stream.read(available, exception_on_overflow=False)
                except Exception:
                    pass
                console.print("\n[bold dark_orange][[LISTENING]][/]")


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


async def amain(use_text, mic_index, pause_threshold=PAUSE_THRESHOLD, enable_droid=USE_DROID_AUDIO):
    global _USE_DROID, _PLAYBACK_QUEUE
    missing = [
        k
        for k in ("OPENAI_API_KEY", "PEERBOTS_API_KEY", "PEERBOTS_USERNAME")
        if not os.environ.get(k, "")
    ]
    if missing:
        console.print(f"[bold red][[ missing in .env: {', '.join(missing)} ]]")
        sys.exit(1)

    # initialize kokoro tts & droid audio if requested
    if enable_droid:
        console.print("[dim white][[ checking kokoro tts & droid audio daemon... ]][/]")
        koko_ok = await ensure_kokoro_ready()
        droid_ok = await ais_droid_reachable()
        if koko_ok and droid_ok:
            _USE_DROID = True
            _PLAYBACK_QUEUE = DroidPlaybackQueue()
            await _PLAYBACK_QUEUE.start()
            console.print("[bold green][[audio]]: streaming Kokoro MPS -> Droid daemon (100.119.180.97)[/]")
        else:
            reasons = []
            if not koko_ok:
                reasons.append("kokoro offline")
            if not droid_ok:
                reasons.append("droid unreachable")
            console.print(f"[dim yellow][[audio fallback]]: {', '.join(reasons)}, using Peerbots tablet audio[/]")
            _USE_DROID = False
    else:
        _USE_DROID = False

    # boot mcp servers once, tools stay live for the whole session
    from contextlib import AsyncExitStack

    try:
        async with AsyncExitStack() as stack:
            tools_list, tool_router = await connect_mcp(stack)
            if use_text:
                await text_loop(tools_list, tool_router)
            else:
                await mic_loop(
                    mic_index,
                    tools_list,
                    tool_router,
                    pause_threshold=pause_threshold,
                )
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        if _PLAYBACK_QUEUE:
            await _PLAYBACK_QUEUE.stop()
        stop_kokoro_process()


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
        help="seconds of silence before speech is considered done (default: 0.85)",
    )
    parser.add_argument(
        "--no-droid",
        action="store_true",
        help="disable kokoro + droid audio and use peerbots tablet tts",
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
    args = parser.parse_args()
    if args.list_mics:
        list_mics()
        return

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
            )
        )
    except (KeyboardInterrupt, *_ExceptionGroupTypes):
        _signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
