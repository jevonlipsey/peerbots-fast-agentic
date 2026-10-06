import json
import socket
import os
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from dotenv import load_dotenv

load_dotenv()

# initialize fastmcp server
mcp = FastMCP("nao_actuation")

DAEMON_HOST = '127.0.0.1'
DAEMON_PORT = int(os.environ.get("DAEMON_PORT", 5005))

def send_command(command: str, args: dict) -> list[TextContent]:
    """Sends a JSON command over TCP to the NAO daemon and returns the result."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5.0)  # 5 second timeout
            s.connect((DAEMON_HOST, DAEMON_PORT))
            
            payload = {"command": command, "args": args}
            s.sendall(json.dumps(payload).encode('utf-8'))
            
            response_data = s.recv(4096)
            if response_data:
                response = json.loads(response_data.decode('utf-8'))
                if response.get("status") == "success":
                    return [TextContent(type="text", text=f"Success: {response.get('message', '')}")]
                else:
                    return [TextContent(type="text", text=f"Error from robot: {response.get('message', 'Unknown error')}")]
            else:
                return [TextContent(type="text", text="Error: No response from robot daemon.")]
    except ConnectionRefusedError:
        return [TextContent(type="text", text="Error: Connection refused. Is the robot daemon running?")]
    except socket.timeout:
        return [TextContent(type="text", text="Error: Connection timed out.")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error sending command: {str(e)}")]


@mcp.tool()
def set_posture(posture: str) -> list[TextContent]:
    """
    Changes the robot's physical posture.
    Valid postures are: "StandInit", "Sit", or "Crouch".
    Use this to adapt to the physical space (e.g. sitting if the user sits).
    """
    if posture not in ["StandInit", "Sit", "Crouch", "Stand"]:
        return [TextContent(type="text", text=f"Error: Invalid posture '{posture}'. Must be StandInit, Sit, or Crouch.")]
    
    return send_command("goToPosture", {"posture": posture})


@mcp.tool()
def look_around(pitch: float, yaw: float) -> list[TextContent]:
    """
    Controls the robot's head joints to look around the room.
    Pitch (up/down): Range is -0.6 (looking up) to 0.5 (looking down). 0.0 is straight ahead.
    Yaw (left/right): Range is -2.0 (looking right) to 2.0 (looking left). 0.0 is straight ahead.
    If a previous picture missed the target, use this tool to guess where it is, then immediately call take_picture again.
    NOTE: This automatically turns off ALBasicAwareness so your head can freely move. You MUST call toggle_awareness(True) when you are done taking pictures to resume tracking the user.
    """
    # clamp values just in case
    pitch = max(-0.6, min(0.5, pitch))
    yaw = max(-2.0, min(2.0, yaw))
    
    return send_command("setAngles", {"pitch": pitch, "yaw": yaw})


@mcp.tool()
def toggle_awareness(state: bool) -> list[TextContent]:
    """
    Turns ALBasicAwareness on or off.
    When True, the robot will automatically track faces and movement in the room.
    Turn this on during normal conversation to feel alive and engaged.
    Turn this off if the constant movement is distracting or if you need to stare perfectly still at a fixed point.
    """
    return send_command("setAwareness", {"state": state})

@mcp.tool()
def save_name(person_id: int, name: str) -> list[TextContent]:
    """
    Saves a human's name to the robot's onboard ALMemory using their ID.
    Always use this tool immediately when a person tells you their name!
    """
    return send_command("saveName", {"id": person_id, "name": name})

@mcp.tool()
def select_camera(camera: str) -> list[TextContent]:
    """
    Selects which physical camera the robot uses.
    Valid values:
    - 'top': Top camera in forehead/eyes (best for faces, people, gazing around the room).
    - 'bottom': Bottom camera angled down at the desk/table (best for examining documents, papers, objects on the table).
    """
    cam_str = camera.strip().lower()
    cam_idx = 0 if cam_str == "top" else (1 if cam_str == "bottom" else None)
    if cam_idx is None:
        return [TextContent(type="text", text="Error: camera must be either 'top' or 'bottom'.")]

    state_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "state")
    cam_file = os.path.join(state_dir, "camera_index.txt")
    try:
        with open(cam_file, "w") as f:
            f.write(str(cam_idx))
        return [TextContent(type="text", text=f"Switched active camera to {cam_str}.")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error setting camera: {e}")]


@mcp.tool()
def set_eye_color(color_hex: str, duration_sec: float = 1.0) -> list[TextContent]:
    """
    Changes the robot's eye LED colors to express emotion or state.
    Provide the color as a hex string (e.g., "#FF0000" for red, "#00FF00" for green, "#0000FF" for blue).
    """
    color_hex = color_hex.lstrip('#')
    if len(color_hex) != 6:
        return [TextContent(type="text", text="Error: color_hex must be a 6-character hex string like '#FF0000'.")]
    
    try:
        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0
    except ValueError:
        return [TextContent(type="text", text="Error: Invalid hex code.")]
        
    return send_command("fadeRGB", {"name": "FaceLeds", "r": r, "g": g, "b": b, "duration": duration_sec})

if __name__ == "__main__":
    mcp.run()