'''
mcp tool server exposing lulo's linear-actuator body to the llm.

it never opens the serial port itself: main.py owns the arduino and serves a
small json control socket on localhost, so the running pipeline (automatic
emotion gestures) and the llm (deliberate gestures) drive the same hardware
without fighting over the port. when lulo is not running, every tool returns a
plain-text error instead of crashing the turn.
'''

import json
import os
import socket

from mcp.server.mcpserver import MCPServer

mcp = MCPServer('lulo_actuator')

HOST = os.environ.get('LULO_ARM_HOST', '127.0.0.1')
PORT = int(os.environ.get('LULO_ARM_PORT', '5007'))
TIMEOUT_S = 3.0


def send_command(payload: dict) -> str:
    '''
    forward one command to the arm controller socket.

    inputs:
    payload: dict like {"cmd": "gesture", "name": "bounce"}
    outputs:
    short human-readable status line, or a plain error string
    '''
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(TIMEOUT_S)
            s.connect((HOST, PORT))
            s.sendall((json.dumps(payload) + '\n').encode('utf-8'))
            data = b''
            while b'\n' not in data:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
            if not data:
                return 'Error: no reply from the lulo arm controller.'
            resp = json.loads(data.decode('utf-8').split('\n', 1)[0])
            if resp.get('ok'):
                mode = ' (simulated, no board attached)' if resp.get('simulated') else ''
                detail = resp.get('reply') or 'done'
                return f'Arm: {detail}{mode}'
            return 'Error: the arm controller rejected the command.'
    except ConnectionRefusedError:
        return 'Error: lulo is not running, so the arm is unavailable.'
    except socket.timeout:
        return 'Error: the arm controller timed out.'
    except Exception as e:
        return f'Error talking to the arm: {e}'


@mcp.tool()
def actuator_emotion(emotion: str) -> str:
    """
    Move Lulo's physical body to match an emotion.
    Valid emotions: "Happy", "Surprised", "Neutral", "Concerned", "Sad", "Sleepy".
    The body already reacts automatically to your face emotion each turn, so you
    normally do NOT need to call this. Use it only when you want the body to move
    to a feeling that differs from the emotion you are expressing.
    """
    return send_command({'cmd': 'emotion', 'name': emotion})


@mcp.tool()
def actuator_gesture(gesture: str) -> str:
    """
    Play one physical gesture with Lulo's body.
    Valid gestures:
      "bounce"  - quick double up-down, joyful/celebratory
      "nod"     - small up-down, agreement or acknowledgement
      "startle" - fast rise and settle, surprise or alert
      "slump"   - slow retract/sag, sadness or deflation
      "wave"    - three up-downs, playful greeting
      "rest"    - stop and hold still
      "extend"  - drive fully out and hold
      "retract" - drive fully in and hold
      "stop"    - stop immediately
    Gestures take roughly 0.3-2.5 seconds. Use them sparingly, at most one per
    reply, for emphasis (for example a "bounce" when the patient finishes a set).
    """
    return send_command({'cmd': 'gesture', 'name': gesture})


@mcp.tool()
def actuator_move(direction: str, duration_ms: int = 0) -> str:
    """
    Drive the actuator in one direction. Use this for a custom, deliberate motion
    instead of a canned gesture.
    direction: "extend" (out/up) or "retract" (in/down).
    duration_ms: how long to drive, in milliseconds. 0 drives continuously until
      actuator_stop is called or the end of travel is reached.
    """
    return send_command({'cmd': 'move', 'direction': direction, 'duration_ms': duration_ms})


@mcp.tool()
def actuator_stop() -> str:
    """Stop the actuator immediately and cancel any gesture in progress."""
    return send_command({'cmd': 'stop'})


@mcp.tool()
def actuator_status() -> str:
    """Report the actuator's estimated position (0-100%), direction, and whether it is moving."""
    return send_command({'cmd': 'status'})


if __name__ == '__main__':
    mcp.run()
