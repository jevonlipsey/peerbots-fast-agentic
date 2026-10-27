'''
usb serial controller for lulo's linear-actuator body.

this module is the single owner of the arduino serial port. main.py imports it to
make lulo physically react to his emotion, and it also serves a tiny localhost
json socket so the lulo_actuator mcp tool server can drive the same hardware
without fighting main.py for the port.

when no arduino is detected it runs in simulation mode: every command is logged
and acknowledged, so the rest of the pipeline still runs on a bare laptop.
'''

import asyncio
import json
import os
import threading
import time
from collections import deque

from rich.console import Console

console = Console()

### config
BAUD = int(os.environ.get('ARDUINO_BAUD', '115200'))
CONTROL_HOST = os.environ.get('LULO_ARM_HOST', '127.0.0.1')
CONTROL_PORT = int(os.environ.get('LULO_ARM_PORT', '5007'))
ACK_TIMEOUT_S = 1.5
GESTURE_DEDUPE_S = float(os.environ.get('LULO_ARM_DEDUPE_S', '0.6'))


def _port_env():
    # read fresh so --arm-port / edited env apply without a restart
    return os.environ.get('ARDUINO_PORT', '').strip()

### usb-serial bridge vids used by arduino-compatible boards
_USB_VIDS = {
    0x2341,  # arduino
    0x2A03,  # arduino (older)
    0x1A86,  # wch ch340
    0x10C4,  # silabs cp210x
    0x0403,  # ftdi
    0x1B4F,  # sparkfun
    0x239A,  # adafruit
    0x2E8A,  # raspberry pi pico
}

### emotion -> onboard gesture (must match the names in lulo_arm.ino)
EMOTION_GESTURES = {
    'Happy': 'bounce',
    'Surprised': 'startle',
    'Neutral': 'rest',
    'Concerned': 'nod',
    'Sad': 'slump',
    'Sleepy': 'slump',
}
VALID_GESTURES = {
    'stop', 'rest', 'extend', 'retract', 'nod', 'bounce', 'startle', 'slump', 'wave',
}


def gesture_for_emotion(emotion):
    '''
    maps a face emotion onto the closest physical gesture.

    inputs:
    emotion: one of the peerbots emotions, or anything else
    outputs:
    gesture name understood by the firmware
    '''
    return EMOTION_GESTURES.get((emotion or '').strip(), 'rest')


def find_arduino_port():
    '''
    locate a likely arduino serial port.

    honors ARDUINO_PORT first, then scans for a known usb-serial bridge vid (or a
    /dev/ttyACM* / /dev/ttyUSB* device) so the user does not have to hardcode it.

    inputs:
    none
    outputs:
    device path string, or None when nothing looks like a board
    '''
    port_env = _port_env()
    if port_env:
        resolved = resolve_arduino_port(port_env)
        if resolved:
            return resolved
    try:
        from serial.tools import list_ports
    except Exception:
        return None
    fallback = None
    for p in list_ports.comports():
        dev = p.device or ''
        if p.vid in _USB_VIDS:
            return dev
        if fallback is None and (dev.startswith('/dev/ttyACM') or dev.startswith('/dev/ttyUSB') or 'usbmodem' in dev):
            fallback = dev
    return fallback


def get_all_serial_ports():
    try:
        from serial.tools import list_ports
        return list(list_ports.comports())
    except Exception:
        return []


def list_serial_ports():
    # human-readable port dump for the CLI with 0-based indices
    ports = get_all_serial_ports()
    if not ports:
        console.print(
            '[dim white][[ arm: no serial ports found (plug in the arduino) ]][/]'
        )
        return []
    for idx, p in enumerate(ports):
        vid = f'{p.vid:04x}' if p.vid is not None else '----'
        pid = f'{p.pid:04x}' if p.pid is not None else '----'
        mark = '->' if p.vid in _USB_VIDS else '  '
        console.print(f'{mark} [{idx}] {p.device}  (vid:pid {vid}:{pid}) {p.description}', markup=False)
    return ports


def resolve_arduino_port(val):
    '''
    resolves a port string or integer index (e.g. '0', 0, or '/dev/cu.usbmodem1101')
    to an actual serial port device path.
    '''
    if val is None or str(val).strip() == '':
        return None
    val_str = str(val).strip()
    if val_str.isdigit():
        idx = int(val_str)
        ports = get_all_serial_ports()
        if 0 <= idx < len(ports):
            return ports[idx].device
    return val_str


class _SerialWorker(threading.Thread):
    '''
    owns the blocking pyserial handle so the asyncio loop never stalls on io.
    opened lines are routed back into the controller for ack matching.
    '''

    def __init__(self, controller, port, baud):
        super().__init__(daemon=True)
        self.controller = controller
        self.port = port
        self.baud = baud
        self._ser = None
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self.ready = threading.Event()
        self.ok = False

    def run(self):
        try:
            import serial
            self._ser = serial.Serial(self.port, self.baud, timeout=0.2, write_timeout=1.0)
            # uno/nano reset when the port opens, so wait out the bootloader and
            # drop the banner bytes before we start matching acks
            time.sleep(2.0)
            try:
                self._ser.reset_input_buffer()
            except Exception:
                pass
            self.ok = True
        except Exception as e:
            self.controller._note_serial_failure(str(e))
            self._ser = None
        self.ready.set()

        if self._ser is None:
            return
        while not self._stop.is_set():
            try:
                line = self._ser.readline().decode('utf-8', 'replace').strip()
            except Exception as e:
                self.controller._note_serial_failure(str(e))
                break
            if line:
                self.controller._on_line(line)
        try:
            self._ser.close()
        except Exception:
            pass

    def write_line(self, text):
        if self._ser is None:
            return False
        try:
            with self._write_lock:
                self._ser.write((text + '\n').encode('ascii', 'ignore'))
            return True
        except Exception as e:
            self.controller._note_serial_failure(str(e))
            return False

    def close(self):
        self._stop.set()


class ArmController:
    '''
    high-level lulo body: emotion gestures, raw moves, and a localhost control
    socket for the mcp tool server.
    '''

    def __init__(self, port=None, simulate=None):
        self.port = port or _port_env() or None
        self.simulate = simulate
        self.worker = None
        self.control_server = None
        self.connected = False
        self.last_error = ''
        self._loop = None
        self._pending = deque()
        self._pending_lock = threading.Lock()
        self._last_status = {}
        self._last_gesture = None
        self._last_gesture_time = 0.0
        self._running = False
        self.log = deque(maxlen=80)

    ### lifecycle
    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._running = True

        if self.simulate is None:
            self.port = self.port or find_arduino_port()
            self.simulate = self.port is None

        if not self.simulate:
            self.worker = _SerialWorker(self, self.port, BAUD)
            self.worker.start()
            await asyncio.to_thread(self.worker.ready.wait, 6.0)
            self.connected = bool(self.worker.ok)
            if self.connected:
                # start every session from a known, centred pose
                try:
                    await self.home()
                except Exception:
                    pass

        if not self.connected:
            self.simulate = True
            if self.port:
                console.print(
                    f'[bold yellow][[arm]]: could not open {self.port}: '
                    f'{self.last_error or "unknown error"}[/]'
                )
            console.print(
                '[dim yellow][[arm]]: no arduino detected, lulo body running in '
                'simulation mode (commands are logged, not sent)[/]'
            )
            if 'permission denied' in (self.last_error or '').lower():
                console.print(
                    '[bold yellow][[arm]]: fix the port permission by joining '
                    'dialout: sudo usermod -aG dialout $USER  (then log out and '
                    'back in, or run `newgrp dialout`)[/]'
                )
        else:
            console.print(f'[bold green][[arm]]: lulo body online on {self.port} @ {BAUD}[/]')

        try:
            self.control_server = await asyncio.start_server(
                self._handle_client, CONTROL_HOST, CONTROL_PORT
            )
            console.print(
                f'[dim white][[arm]]: control socket on {CONTROL_HOST}:{CONTROL_PORT} for mcp tools[/]'
            )
        except Exception as e:
            console.print(f'[dim yellow][[arm]]: control socket unavailable: {e}[/]')

    async def stop(self):
        self._running = False
        if self.control_server is not None:
            try:
                self.control_server.close()
                await self.control_server.wait_closed()
            except Exception:
                pass
            self.control_server = None
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        self._fail_pending('controller stopped')

    ### serial plumbing
    def _note_serial_failure(self, detail):
        self.last_error = detail
        if self.connected:
            console.print(f'[bold red][[arm]]: serial link lost: {detail}[/]')
        self.connected = False
        self.simulate = True

    def _on_line(self, line):
        # runs on the serial worker thread
        self.log.append(line)
        if line.startswith('STATUS'):
            parsed = self._parse_status(line)
            if parsed is not None:
                self._last_status = parsed
        with self._pending_lock:
            fut = self._pending.popleft() if self._pending else None
        if fut is not None and not fut.done() and self._loop is not None:
            self._loop.call_soon_threadsafe(fut.set_result, line)

    def _fail_pending(self, reason):
        with self._pending_lock:
            pending = list(self._pending)
            self._pending.clear()
        for fut in pending:
            if not fut.done():
                fut.set_exception(RuntimeError(reason))

    @staticmethod
    def _parse_status(line):
        try:
            return json.loads(line.split(' ', 1)[1])
        except Exception:
            return None

    @staticmethod
    def _sim_response(cmd):
        # mirror what lulo_arm.ino actually replies, so tests and logs are faithful
        parts = cmd.split(' ', 1)
        verb = parts[0].upper()
        arg = parts[1] if len(parts) > 1 else ''
        if verb == 'STATUS':
            return 'STATUS {"pos":0,"dir":"STOPPED","gesture":"stop","busy":0}'
        if verb == 'PING':
            return 'OK PONG'
        if verb == 'GESTURE':
            return f'OK GESTURE {arg.upper()}'
        if verb in ('UP', 'DOWN'):
            return 'OK MOVE'
        return f'OK {verb}'

    async def send_command(self, cmd, wait=True, timeout=ACK_TIMEOUT_S):
        '''
        send one ascii command to the board.

        inputs:
        cmd: command line without the trailing newline
        wait: block until the firmware acknowledges
        timeout: ack timeout in seconds
        outputs:
        reply line, or None on timeout / no link
        '''
        if not cmd:
            return None
        if self.simulate or not (self.worker and self.worker.ok):
            self.log.append(cmd)
            return self._sim_response(cmd) if wait else ''

        fut = self._loop.create_future() if wait else None
        if fut is not None:
            with self._pending_lock:
                self._pending.append(fut)
        if not self.worker.write_line(cmd):
            if fut is not None:
                with self._pending_lock:
                    try:
                        self._pending.remove(fut)
                    except ValueError:
                        pass
            return None
        if fut is None:
            return ''
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            with self._pending_lock:
                try:
                    self._pending.remove(fut)
                except ValueError:
                    pass
            return None

    ### semantic commands
    async def gesture(self, name):
        name = (name or '').strip().lower()
        if name not in VALID_GESTURES:
            return None
        return await self.send_command(f'GESTURE {name.upper()}')

    async def move(self, direction, duration_ms=0):
        d = (direction or '').strip().lower()
        if d in ('up', 'extend', 'extended', 'out'):
            verb = 'UP'
        elif d in ('down', 'retract', 'retracted', 'in'):
            verb = 'DOWN'
        else:
            return None
        cmd = f'{verb} {int(duration_ms)}' if duration_ms else verb
        return await self.send_command(cmd)

    async def halt(self):
        return await self.send_command('STOP')

    async def reset(self):
        return await self.send_command('RESET')

    async def home(self):
        '''
        park the actuator mid-stroke so gestures have travel in both directions.

        the firmware assumes fully retracted at power-up and tracks position by
        integrating drive time. at position 0 a downward move is refused, so slump
        (Sad, Sleepy) did nothing at all. homing on connect also clears whatever
        estimate error the previous session accumulated.
        '''
        return await self.send_command('HOME')

    async def status(self):
        reply = await self.send_command('STATUS')
        parsed = self._parse_status(reply) if reply else None
        return parsed or dict(self._last_status)

    ### emotion bridge (sync, safe to call from asyncio code)
    def fire_emotion(self, emotion):
        '''
        make lulo physically react to a face emotion, without blocking the turn.
        identical emotions within the dedupe window are ignored so a repeated
        face update does not machine-gun the actuator.
        '''
        g = gesture_for_emotion(emotion)
        now = time.time()
        if g == self._last_gesture and (now - self._last_gesture_time) < GESTURE_DEDUPE_S:
            return
        self._last_gesture = g
        self._last_gesture_time = now
        self.fire(f'GESTURE {g.upper()}')

    def fire(self, cmd):
        # schedule a fire-and-forget command on the running loop
        try:
            loop = self._loop or asyncio.get_running_loop()
            return loop.create_task(self.send_command(cmd, wait=True))
        except RuntimeError:
            return None

    ### localhost control socket for the mcp tool server
    async def _dispatch(self, req):
        cmd = str(req.get('cmd', '')).strip().lower()
        if cmd == 'gesture':
            return await self.gesture(req.get('name') or req.get('gesture'))
        if cmd == 'emotion':
            return await self.gesture(gesture_for_emotion(req.get('name') or req.get('emotion')))
        if cmd == 'move':
            return await self.move(req.get('direction', ''), req.get('duration_ms', 0))
        if cmd == 'stop':
            return await self.halt()
        if cmd == 'reset':
            return await self.reset()
        if cmd == 'status':
            return json.dumps(await self.status())
        if cmd == 'ping':
            return await self.send_command('PING')
        if cmd == 'raw':
            return await self.send_command(str(req.get('line', '')))
        return None

    async def _handle_client(self, reader, writer):
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    req = json.loads(line.decode('utf-8').strip() or '{}')
                except Exception:
                    req = {}
                reply = await self._dispatch(req)
                ok = reply is not None
                payload = {'ok': ok, 'reply': reply, 'simulated': bool(self.simulate)}
                writer.write((json.dumps(payload) + '\n').encode('utf-8'))
                await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description='lulo arm controller')
    parser.add_argument('--list-ports', action='store_true', help='print serial ports and exit')
    parser.add_argument('--sim', action='store_true', help='force simulation mode')
    args = parser.parse_args()

    if args.list_ports:
        list_serial_ports()
        console.print(f'autodetect would pick: {find_arduino_port()}', markup=False)
        return

    async def demo():
        arm = ArmController(simulate=True if args.sim else None)
        await arm.start()
        try:
            for g in ('bounce', 'nod', 'slump'):
                console.print(f'sending GESTURE {g}')
                await arm.gesture(g)
                await asyncio.sleep(0.3)
        finally:
            await arm.stop()

    asyncio.run(demo())


if __name__ == '__main__':
    _cli()
