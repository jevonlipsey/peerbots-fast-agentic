"""Chat with ChatGPT; when you ask it to, it extends/retracts the actuator.

The Arduino runs actuator_functions.ino and listens on serial:
  'E' -> extendActuator(), 'R' -> retractActuator(), 'X' -> excitedActuator(),
  'A' -> playTubeAnimation()

Setup:
  pip install openai pyserial
  set OPENAI_API_KEY=your-key        (PowerShell: $env:OPENAI_API_KEY="your-key")
  Edit SERIAL_PORT below to match the Arduino IDE's port (e.g. COM3).
"""
import json
import time

import serial
from openai import OpenAI

SERIAL_PORT = "COM3"
BAUD = 9600
MODEL = "gpt-4o"

client = OpenAI()  # reads OPENAI_API_KEY from the environment
arduino = serial.Serial(SERIAL_PORT, BAUD, timeout=40)  # animation takes ~25 s
time.sleep(2)  # Arduino resets when the port opens

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extend_actuator",
            "description": "Fully extend the linear actuator. Takes about 5 seconds.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retract_actuator",
            "description": "Fully retract the linear actuator. Takes about 5 seconds.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "excited_actuator",
            "description": "Show excitement by moving the actuator up and down, "
                           "switching direction every 2 seconds. Takes about 12 seconds.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_animation",
            "description": "Play the authored URDF animation on the actuator: a "
                           "pre-choreographed sequence of extending and retracting "
                           "motions. Takes about 25 seconds.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

COMMANDS = {
    "extend_actuator": b"E",
    "retract_actuator": b"R",
    "excited_actuator": b"X",
    "play_animation": b"A",
}


def run_tool(name):
    arduino.write(COMMANDS[name])
    reply = arduino.readline().decode().strip()  # waits for "DONE ..."
    return reply or "no response from Arduino"


messages = [
    {
        "role": "system",
        "content": "You control a linear actuator. Use the tools to extend or "
                   "retract it only when the user asks. If the user asks you to be "
                   "excited (or to show excitement), use excited_actuator. If the user asks for the animation, use "
                   "play_animation. "
                   "Keep replies short.",
    }
]

print("Chat with ChatGPT (e.g. 'extend the actuator'). Ctrl+C to quit.")
while True:
    messages.append({"role": "user", "content": input("you> ")})

    while True:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS
        )
        msg = response.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            print("gpt>", msg.content)
            break
        for call in msg.tool_calls:
            print(f"[running {call.function.name}]")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": run_tool(call.function.name),
                }
            )
