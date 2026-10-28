'''
semantic end-of-turn detection, so endpointing stops being a stopwatch.

the pipeline used to wait a fixed 1.0s of silence before deciding a turn was
over. that single constant was the largest item in the whole latency budget - and
it was also the wrong mechanism, because the right amount to wait depends on
whether the person has finished a thought, not on a clock. a short wait cuts
people off mid-sentence; a long one makes every exchange feel sluggish.

Smart Turn v3.2 (pipecat-ai, BSD-2-Clause) reads the audio directly and returns
the probability that the speaker has finished. it is a Whisper-tiny backbone with
a linear head, ~8M parameters, and its training set includes cafe and office
background noise, which is the case this demo actually has to survive. the whisper
log-mel front end is inside the graph, so raw 16kHz samples go in.

used here as a gate on top of silero rather than a replacement: silero still says
"this frame is speech", and this model answers "and the person is done". the short
silence window becomes a trigger to ask the question instead of the answer itself.

the model is optional. without it, the caller falls back to the fixed window, so a
missing file degrades latency rather than breaking the robot.
'''

import os
import sys
import urllib.request

from rich.console import Console

console = Console()

SAMPLE_RATE = 16000
# 8s at 16kHz. longer turns keep their most recent 8s; shorter ones are padded
# with zeros at the front so the speech sits at the end of the window.
WINDOW_SAMPLES = 128000
TURN_COMPLETE_THRESHOLD = float(os.environ.get('SMART_TURN_THRESHOLD', '0.5'))

_REPO = 'soniqo/Smart-Turn-v3.2-ONNX'
_FILENAME = 'smart-turn-v3.2-int8.onnx'
_DOWNLOAD_URL = f'https://huggingface.co/{_REPO}/resolve/main/{_FILENAME}'

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'
)
MODEL_PATH = os.environ.get('SMART_TURN_MODEL', os.path.join(_CACHE_DIR, _FILENAME))

_session = None
_load_failed = False
_np = None


def model_available():
    '''whether the weights are on disk'''
    return os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024


def download_model(dest=None, quiet=False):
    '''
    fetch the onnx weights once, so a demo never waits on a network call.

    inputs:
    dest: destination path, defaults to MODEL_PATH
    quiet: suppress progress output
    outputs:
    path on success, None on failure
    '''
    dest = dest or MODEL_PATH
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + '.part'
    try:
        if not quiet:
            console.print(f'[dim white]\\[\\[smart-turn]]: downloading {_FILENAME}...[/]')
        with urllib.request.urlopen(_DOWNLOAD_URL, timeout=120) as resp, open(tmp, 'wb') as fh:
            while True:
                block = resp.read(262144)
                if not block:
                    break
                fh.write(block)
        os.replace(tmp, dest)
        if not quiet:
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            console.print(
                f'[bold green]\\[\\[smart-turn]]: model ready ({size_mb:.1f}MB) at {dest}[/]'
            )
        return dest
    except Exception as exc:
        if not quiet:
            console.print(f'[bold yellow]\\[\\[smart-turn]]: download failed: {exc}[/]')
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass
        return None


def get_session():
    '''
    lazily build the onnxruntime session.

    inputs:
    none
    outputs:
    InferenceSession, or None when unavailable (caller falls back to the timer)
    '''
    global _session, _load_failed, _np
    if _session is not None or _load_failed:
        return _session
    if not model_available():
        _load_failed = True
        return None
    try:
        import numpy
        import onnxruntime as ort

        _np = numpy
        opts = ort.SessionOptions()
        # one turn at a time, so extra threads only add scheduling noise
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        _session = ort.InferenceSession(
            MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider']
        )
    except Exception as exc:
        console.print(f'[dim yellow]\\[\\[smart-turn]]: unavailable ({exc}), using fixed window[/]')
        _load_failed = True
        _session = None
    return _session


def turn_complete_probability(samples):
    '''
    probability that the speaker has finished their turn.

    inputs:
    samples: float32 numpy array of 16kHz mono audio, most recent last
    outputs:
    float in 0..1, or None when the model is unavailable
    '''
    session = get_session()
    if session is None:
        return None
    try:
        x = _np.asarray(samples, dtype=_np.float32).reshape(-1)[-WINDOW_SAMPLES:]
        if x.size < WINDOW_SAMPLES:
            # pad at the front: the model expects the speech flush to the end
            x = _np.concatenate(
                [_np.zeros(WINDOW_SAMPLES - x.size, dtype=_np.float32), x]
            )
        (prob,) = session.run(['probability'], {'audio': x.reshape(1, -1)})
        return float(prob[0, 0])
    except Exception:
        return None


def is_turn_complete(samples, threshold=None):
    '''
    inputs:
    samples: float32 numpy array of 16kHz mono audio
    threshold: override for TURN_COMPLETE_THRESHOLD
    outputs:
    (complete, probability); complete is None when the model is unavailable
    '''
    prob = turn_complete_probability(samples)
    if prob is None:
        return None, None
    limit = TURN_COMPLETE_THRESHOLD if threshold is None else threshold
    return prob > limit, prob


def prewarm():
    '''
    run one inference on silence so the first real turn does not pay graph setup.

    inputs:
    none
    outputs:
    True when the model is loaded and warm
    '''
    session = get_session()
    if session is None:
        return False
    try:
        turn_complete_probability(_np.zeros(WINDOW_SAMPLES, dtype=_np.float32))
        return True
    except Exception:
        return False


if __name__ == '__main__':
    if '--download' in sys.argv:
        sys.exit(0 if download_model() else 1)
    if model_available():
        ok = prewarm()
        console.print(f'model at {MODEL_PATH}, warm={ok}')
    else:
        console.print(f'no model at {MODEL_PATH}; run: python scripts/smart_turn.py --download')
