'''
pre-demo check: confirm every subsystem is up before you walk on stage.

the pipeline spans three networks (openai, a local kokoro server, the peerbots
cloud), a websocket to the handset, a usb serial link to the actuator and two
local model files. any one of them failing degrades quietly - the audio falls back
to the tablet, the endpointer falls back to a fixed window, the body falls back to
simulation - so a broken setup looks like a working one until it is on camera.

each check returns (ok, detail) and never raises. exit code is 0 only when nothing
is in the FAIL column.
'''

import asyncio
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console

console = Console()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# a failed check blocks the demo; a warning only costs quality or latency
BLOCKING = 'fail'
WARNING = 'warn'


def _env_keys():
    required = ('OPENAI_API_KEY', 'PEERBOTS_API_KEY', 'PEERBOTS_USERNAME')
    missing = [k for k in required if not os.environ.get(k, '')]
    if missing:
        return False, f'missing in .env: {", ".join(missing)}'
    placeholder = [
        k for k in required
        if (os.environ.get(k, '') or '').lower().startswith(('your-', 'your_', 'sk-...'))
    ]
    if placeholder:
        return False, f'still placeholder: {", ".join(placeholder)}'
    return True, f'{len(required)} keys present'


def _model_name():
    model = os.environ.get('OPENAI_MODEL', '')
    if not model:
        return False, 'OPENAI_MODEL unset'
    return True, model


def _swift_worker():
    import whisper_stt

    if not whisper_stt.IS_MAC:
        return True, 'not macos, coreml worker not used'
    built = os.path.join(
        ROOT, 'stt-coreml', 'stt_worker', '.build', 'arm64-apple-macosx',
        'release', 'stt_worker',
    )
    if os.path.exists(built):
        return True, 'release binary present'
    # without it start_swift_worker falls back to `swift run -c release`, which
    # compiles on launch and waits up to 300s for the worker to say it is ready
    return False, (
        'no release binary: first launch will compile (up to 300s). build it with '
        'cd stt-coreml/stt_worker && swift build -c release'
    )


def _smart_turn_model():
    import smart_turn

    if smart_turn.model_available():
        size = os.path.getsize(smart_turn.MODEL_PATH) / (1024 * 1024)
        return True, f'{os.path.basename(smart_turn.MODEL_PATH)} ({size:.1f}MB)'
    return False, (
        'no semantic endpointer: endpointing falls back to a fixed 1.0s window. '
        'get it with python scripts/smart_turn.py --download'
    )


def _scripted_lines():
    import json

    path = os.path.join(ROOT, 'config', 'scripted_lines.json')
    try:
        lines = json.load(open(path, encoding='utf-8'))['lines']
    except Exception as exc:
        return False, f'unreadable: {exc}'
    if not lines:
        return False, 'no scripted lines, nothing will be cached'
    return True, f'{len(lines)} dialogue-tree lines to prewarm'


def _system_prompt():
    path = os.path.join(ROOT, 'config', 'system_prompt.md')
    if not os.path.exists(path):
        return False, 'config/system_prompt.md missing'
    size = os.path.getsize(path)
    return True, f'{size} bytes (~{size // 4} tokens, cached after turn 1)'


def _history_clean():
    import openai_response

    try:
        data = open(openai_response.HISTORY_FILE, encoding='utf-8').read().strip()
    except Exception:
        return True, 'no history file, will start fresh'
    if data in ('', '[]'):
        return True, 'empty, will start fresh'
    return True, f'{len(data)} bytes present (archived automatically unless --keep-history)'


def _arduino():
    try:
        from arm_controller import find_arduino_port
    except Exception as exc:
        return False, f'arm controller unavailable: {exc}'
    port = find_arduino_port()
    if port:
        return True, port
    return False, 'no serial port found, body will run in simulation (use --no-arduino)'


def _microphone():
    try:
        import pyaudio
    except Exception as exc:
        return False, f'pyaudio unavailable: {exc}'
    pa = pyaudio.PyAudio()
    try:
        inputs = [
            i for i in range(pa.get_device_count())
            if int(pa.get_device_info_by_index(i).get('maxInputChannels', 0)) > 0
        ]
        if not inputs:
            return False, 'no capture-capable devices'
        try:
            stream = pa.open(
                format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                frames_per_buffer=1024,
            )
            stream.stop_stream()
            stream.close()
        except Exception as exc:
            return False, f'default device will not open at 16k mono: {exc}'
        return True, f'{len(inputs)} input device(s), default opens at 16k mono'
    finally:
        pa.terminate()


def _port_free(port):
    sock = socket.socket()
    try:
        sock.bind(('0.0.0.0', port))
        return True, f'port {port} free for the phone mic server'
    except Exception as exc:
        return False, f'port {port} unavailable: {exc}'
    finally:
        sock.close()


async def _kokoro():
    import kokoro_manager

    if await kokoro_manager.ais_kokoro_running(timeout=2.0):
        return True, f'running at {kokoro_manager.KOKORO_SPEECH_URL}'
    return False, 'not running (it is auto-launched at startup, but that takes time)'


async def _openai():
    import openai_response

    try:
        await asyncio.wait_for(openai_response.prewarm_openai(), timeout=12.0)
        return True, 'api reachable, connection pool warm'
    except Exception as exc:
        return False, f'unreachable: {type(exc).__name__}: {exc}'


async def _peerbots():
    import httpx

    user = os.environ.get('PEERBOTS_USERNAME', '')
    if not user:
        return False, 'PEERBOTS_USERNAME unset'
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get('https://api.peerbots.org/')
        return True, f'api.peerbots.org reachable (http {resp.status_code})'
    except Exception as exc:
        return False, f'unreachable: {type(exc).__name__}'


async def run(phone_port=8443, check_arduino=True, check_mic=True):
    '''
    run every check and print a report.

    inputs:
    phone_port: port the phone mic server will bind
    check_arduino: include the usb serial body
    check_mic: include the local microphone (skip for phone-mic demos)
    outputs:
    True when nothing is in the FAIL column
    '''
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, '.env'))

    sync_checks = [
        ('credentials', _env_keys, BLOCKING),
        ('llm model', _model_name, BLOCKING),
        ('system prompt', _system_prompt, BLOCKING),
        ('scripted lines', _scripted_lines, WARNING),
        ('session history', _history_clean, WARNING),
        ('coreml stt worker', _swift_worker, WARNING),
        ('semantic endpointer', _smart_turn_model, WARNING),
        ('phone mic port', lambda: _port_free(phone_port), WARNING),
    ]
    if check_mic:
        sync_checks.append(('microphone', _microphone, WARNING))
    if check_arduino:
        sync_checks.append(('actuator body', _arduino, WARNING))

    async_checks = [
        ('openai', _openai, BLOCKING),
        ('kokoro tts', _kokoro, WARNING),
        ('peerbots cloud', _peerbots, BLOCKING),
    ]

    console.print('\n[bold]pre-demo check[/]\n')
    failures = 0
    warnings = 0

    results = []
    for name, fn, severity in sync_checks:
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f'check raised {type(exc).__name__}: {exc}'
        results.append((name, ok, detail, severity))

    for name, fn, severity in async_checks:
        try:
            ok, detail = await fn()
        except Exception as exc:
            ok, detail = False, f'check raised {type(exc).__name__}: {exc}'
        results.append((name, ok, detail, severity))

    for name, ok, detail, severity in results:
        if ok:
            console.print(f'  [bold green]PASS[/]  {name:22} [dim]{detail}[/]', markup=True)
        elif severity == BLOCKING:
            failures += 1
            console.print(f'  [bold red]FAIL[/]  {name:22} {detail}')
        else:
            warnings += 1
            console.print(f'  [bold yellow]WARN[/]  {name:22} {detail}')

    console.print('')
    if failures:
        console.print(
            f'[bold red]{failures} blocking problem(s)[/]'
            + (f', {warnings} warning(s)' if warnings else '')
        )
    elif warnings:
        console.print(
            f'[bold yellow]ready, with {warnings} warning(s)[/] '
            '[dim](each one is a quiet fallback, not a crash)[/]'
        )
    else:
        console.print('[bold green]all systems go[/]')
    return failures == 0


if __name__ == '__main__':
    ok = asyncio.run(
        run(
            check_arduino='--no-arduino' not in sys.argv,
            check_mic='--phone-mic' not in sys.argv,
        )
    )
    sys.exit(0 if ok else 1)
