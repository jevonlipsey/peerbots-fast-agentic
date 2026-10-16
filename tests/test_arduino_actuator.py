'''
exercises the lulo actuator stack end to end without any hardware:
emotion mapping, the controller's simulation transport, the localhost control
socket, and the mcp tool server talking to a live controller.

run directly (python tests/test_arduino_actuator.py) or under pytest.
'''

import asyncio
import importlib.util
import json
import os
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, 'scripts')
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

# use a non-default control port so a real lulo session is never disturbed
TEST_PORT = 5099
os.environ['LULO_ARM_PORT'] = str(TEST_PORT)

from arm_controller import (  # noqa: E402
    EMOTION_GESTURES,
    VALID_GESTURES,
    ArmController,
    gesture_for_emotion,
)


def _load_tool_server():
    path = os.path.join(ROOT, 'mcp-servers', 'lulo_actuator', 'server.py')
    spec = importlib.util.spec_from_file_location('lulo_actuator_server', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _socket_call(payload):
    # blocking json round trip; callers run this in a worker thread
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(3.0)
        s.connect(('127.0.0.1', TEST_PORT))
        s.sendall((json.dumps(payload) + '\n').encode('utf-8'))
        data = b''
        while b'\n' not in data:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        return json.loads(data.decode('utf-8').split('\n', 1)[0])


def test_emotion_gesture_map():
    for emotion in ('Happy', 'Surprised', 'Neutral', 'Concerned', 'Sad', 'Sleepy'):
        gesture = gesture_for_emotion(emotion)
        assert gesture in VALID_GESTURES, f'{emotion} -> {gesture} not in firmware set'
    assert gesture_for_emotion('') == 'rest'
    assert gesture_for_emotion('nonsense') == 'rest'
    assert set(EMOTION_GESTURES.values()) <= VALID_GESTURES


def test_simulated_controller():
    async def run():
        arm = ArmController(simulate=True)
        await arm.start()
        try:
            assert await arm.gesture('bounce') == 'OK GESTURE BOUNCE'
            assert await arm.gesture('not-a-gesture') is None
            assert await arm.move('extend', 300) == 'OK MOVE'
            assert await arm.move('sideways') is None
            status = await arm.status()
            assert status.get('dir') == 'STOPPED'
            assert 'GESTURE BOUNCE' in arm.log
            assert 'UP 300' in arm.log
        finally:
            await arm.stop()

    asyncio.run(run())


def test_control_socket_roundtrip():
    async def run():
        arm = ArmController(simulate=True)
        await arm.start()
        try:
            reply = await asyncio.to_thread(_socket_call, {'cmd': 'gesture', 'name': 'nod'})
            assert reply['ok'] and 'GESTURE NOD' in reply['reply']
            reply = await asyncio.to_thread(_socket_call, {'cmd': 'emotion', 'name': 'Happy'})
            assert reply['ok'] and 'GESTURE BOUNCE' in reply['reply']
            reply = await asyncio.to_thread(_socket_call, {'cmd': 'stop'})
            assert reply['ok']
        finally:
            await arm.stop()

    asyncio.run(run())


def test_mcp_tool_server_against_live_controller():
    server = _load_tool_server()

    async def run():
        arm = ArmController(simulate=True)
        await arm.start()
        try:
            out = await asyncio.to_thread(server.actuator_gesture, 'slump')
            assert 'OK GESTURE' in out, out
            out = await asyncio.to_thread(server.actuator_move, 'retract', 200)
            assert 'OK MOVE' in out, out
            out = await asyncio.to_thread(server.actuator_status)
            assert 'pos' in out, out
            out = await asyncio.to_thread(server.actuator_emotion, 'Sad')
            assert 'OK GESTURE' in out, out
        finally:
            await arm.stop()

    asyncio.run(run())


def test_mcp_tool_server_reports_unavailable():
    server = _load_tool_server()
    out = server.actuator_stop()  # nothing is listening on TEST_PORT now
    assert 'Error' in out, out


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in tests:
        fn()
        print(f'{fn.__name__}: PASS')
    print(f'\nALL {len(tests)} ACTUATOR TESTS PASSED')
