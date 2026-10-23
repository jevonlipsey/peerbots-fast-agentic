# lulo_actuator

MCP tool server that lets the OpenAI agent move Lulo's linear-actuator "body".

It does **not** open the serial port itself. `main.py` owns the Arduino and
serves a small JSON control socket on `127.0.0.1:5007` (override with
`LULO_ARM_PORT`). Both the automatic emotion→gesture mapping in `main.py` and
these MCP tools go through that one socket, so there is never a port conflict.
If Lulo is not running, every tool returns a plain-text error.

## tools

| tool | what it does |
| ---- | ------------ |
| `actuator_emotion(emotion)` | move to match an emotion (usually automatic already) |
| `actuator_gesture(gesture)` | `bounce`, `nod`, `startle`, `slump`, `wave`, `rest`, `extend`, `retract`, `stop` |
| `actuator_move(direction, duration_ms)` | custom `extend`/`retract` for a duration |
| `actuator_stop()` | stop now, cancel any gesture |
| `actuator_status()` | estimated position 0-100%, direction, busy |

It is registered in `mcp_config.json` and launched by the main process with the
`{python}` interpreter token, so no `uvx` is required.
