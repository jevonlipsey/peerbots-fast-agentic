# getting it running from scratch

a setup guide for a fresh machine. works on macos and linux; commands for each
are given side by side.

there are three stages. each one ends with something you can actually run, so
you can stop after any of them and still have a working robot.

| stage | what you get | extra installs |
| ----- | ------------ | -------------- |
| 1 | lulo listens on your laptop's mic and speaks from the peerbots tablet | none |
| 2 | lulo speaks locally with kokoro instead of the tablet | kokoro-fastapi |
| 3 | your phone becomes both the microphone and the speaker | adb |

---

## what you need before you start

### both platforms

- **python 3.10 or newer**
- **an openai api key** — https://platform.openai.com/api-keys
- **a peerbots account and api key** — https://peerbots.org, api docs at
  https://api.peerbots.org/docs

### macos

- **xcode command line tools** — builds the fast speech-to-text worker.
  ```bash
  xcode-select --install
  ```
- **homebrew** — https://brew.sh
- **portaudio** — required by `pyaudio` for microphone access.
  ```bash
  brew install portaudio
  ```

### linux

- **portaudio development headers** — required to build `pyaudio`.
  ```bash
  sudo apt-get install portaudio19-dev        # debian / ubuntu
  sudo dnf install portaudio-devel            # fedora
  ```
- **alsa-utils** — provides `aplay`, used to play speech locally in stage 2.
  ```bash
  sudo apt-get install alsa-utils
  ```

there is no coreml worker on linux, so speech-to-text falls back to
`openai-whisper`, which is installed from `requirements.txt` along with its
`torch` dependency. that download is large, but it is automatic.

---

## stage 1 — get it talking

### 1.1 clone it

```bash
git clone https://github.com/jevonlipsey/peerbots-fast-agentic.git
cd peerbots-fast-agentic
```

### 1.2 install the python packages

any environment works — venv, conda, whatever you already use.

```bash
pip install -r requirements.txt
```

if `pyaudio` fails to build, the portaudio step above is missing.

### 1.3 create your keys file

```bash
cp .env.example .env
```

open `.env` and fill in these three. they are **required** — the program exits
immediately without them.

```ini
OPENAI_API_KEY='sk-...'
PEERBOTS_API_KEY='your peerbots api key'
PEERBOTS_USERNAME='your peerbots account username'
```

`PEERBOTS_USERNAME` has to match the account you sign into the face with, or
emotion updates silently go nowhere.

### 1.4 pick your microphone

do this before the first real run. list the audio devices:

```bash
python main.py --list-mics
```

you get every device with its input channel count, for example:

```
-> [2] MacBook Pro Microphone (1 input channels)
   [3] MacBook Pro Speakers (output only)
```

`->` is the system default. **any device marked `output only` can never record**,
so ignore those. if the default is an output device, that is exactly why the
next step would fail.

pick yours by index:

```bash
python main.py --mic-index 2
```

### 1.5 run it

```bash
python main.py
```

first run builds the speech-to-text worker on macos, which takes a few minutes.
after that you will see:

```
[[ checking kokoro tts & droid audio daemon... ]]
[[ audio fallback ]]: kokoro offline, using Peerbots tablet audio
```

**that fallback line is expected at this stage.** when `[[LISTENING]]` appears,
talk out loud and lulo answers through the peerbots face.

`--text` skips the microphone entirely and lets you type instead, which is the
fastest way to prove the rest of the pipeline works.

---

## stage 2 — local voice (optional)

kokoro is a text-to-speech server that runs on your machine. it replaces the
tablet's voice, which is why it is a separate install.

### 2.1 get kokoro

```bash
git clone https://github.com/remsky/Kokoro-FastAPI.git ~/Kokoro-FastAPI
```

follow that repo's instructions to get it serving on http://localhost:8880.

on macos this project will launch it for you — it looks for `start-cpu_mac.sh`
or `start-gpu_mac.sh` in `~/Kokoro-FastAPI`, which is why the location matters.
that auto-launch is macos-only, so **on linux start kokoro yourself** before
running the pipeline.

### 2.2 run with local audio

```bash
python main.py --local-audio
```

`--local-audio` is the important part. without it, kokoro being online does not
help, because the ready audio has nowhere to go and you get the tablet fallback
anyway.

speech then plays through `afplay` on macos and `aplay` on linux. set
`LOCAL_PLAY_CMD` if you use something else:

```bash
LOCAL_PLAY_CMD=ffplay python main.py --local-audio
```

---

## stage 3 — use the phone as mic and speaker (optional)

the phone becomes both the microphone and the speaker, with the peerbots face
full-screen on its display.

### 3.1 install adb

```bash
brew install --cask android-platform-tools        # macos
sudo apt-get install android-tools-adb             # debian / ubuntu
```

### 3.2 turn on debugging on the phone

settings → about phone → tap **build number** seven times to unlock developer
options, then settings → developer options → **USB debugging** on. plug the
phone in and run `adb devices` — accept the prompt on the phone screen.

to skip the cable, also enable **wireless debugging** in the same menu.

### 3.3 bridge the port and run

```bash
adb reverse tcp:8443 tcp:8443
python main.py --phone-mic --phone-audio --no-dual-audio
```

then on the phone, open `http://localhost:8443/` and tap **Enable microphone**.

### why localhost and not your laptop's ip

chrome only hands over the microphone in a "secure context" — `localhost`,
`127.0.0.1`, or `https://`. `http://192.168.x.x:8443` is none of those, so the mic
button fails with no useful error. `adb reverse` makes the phone see your laptop
as `localhost`, which works.

**no cable?** swap the last command for:

```bash
python main.py --phone-mic --phone-audio --no-dual-audio --phone-tunnel
```

it prints an `https://...trycloudflare.com` url to open on the phone instead. no
adb needed at all. the url is random and changes every run.

### if it drops out

`adb reverse` dies whenever the connection does — unplugging, a wifi roam,
sleeping the laptop. the phone page keeps looking connected, so a dead reverse
looks like lulo going deaf. re-run `adb reverse tcp:8443 tcp:8443` and reload the
page.

---

## stage 4 — physical actuator body (optional)

Lulo can move a linear actuator over usb serial: his body extends and retracts to
match his emotion (Happy bounces, Surprised startles, Sad slumps, Concerned nods,
...), and the OpenAI agent can also call actuator tools on purpose.

### 4.1 install arduino-cli

```bash
# linux / macos, no sudo, installs into ~/.local/bin
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | BINDIR=$HOME/.local/bin sh
export PATH="$HOME/.local/bin:$PATH"   # add this to your shell rc to keep it
```

### 4.2 install the board core

```bash
arduino-cli core update-index
arduino-cli core install arduino:avr   # uno / nano / mega
```

### 4.3 flash the firmware

wire the L298N as noted at the top of the sketch (ENA->D5, IN1->D10, IN2->D8,
button D2->GND, shared GND), plug the board in over usb, then:

```bash
arduino-cli board list                          # find the port, e.g. /dev/ttyACM0
arduino-cli compile --fqbn arduino:avr:uno arduino/lulo_arm
arduino-cli upload  --fqbn arduino:avr:uno -p /dev/ttyACM0 arduino/lulo_arm
```

adjust `--fqbn` (`arduino:avr:nano`, `arduino:avr:mega`) and `-p` to your board.
`arduino/actuator_button.ino` is the standalone button-only sketch if you just
want to test the mechanics by hand.

### 4.4 run it

```bash
python main.py                  # auto-detects the board
python main.py --arm-port /dev/ttyACM0
python main.py --list-arm-ports
python main.py --no-arduino     # disable the body entirely
```

on startup you will see `[[arm]]: lulo body online on <port>`, or, when no board
is connected, `simulation mode` where commands are logged but not sent. the rest
of the pipeline works either way, so you can develop without hardware.

notes

- there is no position sensor: the firmware integrates position from drive time,
  assumes fully retracted at power-up, and re-syncs at each end.
- the manual button still works and overrides any gesture.
- env: `ARDUINO_PORT`, `ARDUINO_BAUD` (default 115200), `LULO_ARM_PORT` (the
  localhost control socket the agent's tools talk to, default 5007).

---

## do not run these

`setup/setup.sh` and `setup/setup.bat` are leftovers from the earlier nao
project. they build a python 2.7 naoqi environment and none of it is used here.

---

## troubleshooting

### a wall of `ALSA lib ...` errors

**harmless.** pyaudio enumerates every alsa device at startup and alsa complains
about each pseudo device it cannot open. it is noise, not the failure. look at
the last lines of the traceback for the real error.

### `could not open microphone`

the device cannot capture. run `--list-mics` and check you are not pointing at
something marked `output only`. the program prints the device list automatically
when it fails, along with the underlying error.

on linux, if the plain `default` device fails, try the `pulse` entry from
`--list-mics`, or run through pipewire:

```bash
python main.py --mic-index <the pulse device index>
```

| other symptom | cause |
| ------- | ----- |
| `[[ missing in .env: ... ]]` | one of the three required keys is blank |
| `[[ mcp boot failed, continuing without tools ]]` | harmless. install [uv](https://docs.astral.sh/uv/) if you want the extra tools |
| speech-to-text never initializes | macos only: xcode command line tools are missing |
| `pyaudio` will not install | install portaudio and its headers (see prerequisites) |
| everything works but there is no sound | add `--local-audio`, or leave it on the tablet fallback |
| phone page loads but never hears you | it must be `localhost` or `https`, and you must tap **Enable microphone** |
| `[[ phone-mic ]]: another browser already streaming` | close the extra tabs — only one phone may stream at a time |
| lulo cuts you off mid-sentence | `python main.py --pause-threshold 3.0` |
| `[[arm]]: ... simulation mode` | no board detected. check usb, run `python main.py --list-arm-ports`, or set `ARDUINO_PORT` |
| the body moves the wrong way | swap the L298N motor leads (or swap IN1/IN2) |
| `permission denied` opening the board | add yourself to `dialout`: `sudo usermod -aG dialout $USER`, then log out and back in |

## never commit

`.env` holds live api keys and is gitignored. keep it that way.
