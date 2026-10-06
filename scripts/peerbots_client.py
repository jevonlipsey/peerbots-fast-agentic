'''
peerbots rest client. sends speech + face state to the peerbots cloud,
which forwards it to the phone running the peerbots face panel.

layer 3: llm json -> peerbots_client -> face phone
'''

import os

import httpx
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

### config
PEERBOTS_BASE_URL = 'https://api.peerbots.org/v1'
PEERBOTS_API_KEY = os.environ.get('PEERBOTS_API_KEY', '')
PEERBOTS_USERNAME = os.environ.get('PEERBOTS_USERNAME', '')
DEFAULT_TIMEOUT_S = 10.0
SILENT_TIMEOUT_S = 8.0

_ASYNC_CLIENT = None


def get_async_client():
    """cached persistent client for low-latency reusable http connections"""
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is None or _ASYNC_CLIENT.is_closed:
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=60.0)
        _ASYNC_CLIENT = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S, limits=limits)
    return _ASYNC_CLIENT


async def close_client():
    """closes persistent client on shutdown"""
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is not None and not _ASYNC_CLIENT.is_closed:
        try:
            await _ASYNC_CLIENT.aclose()
        except Exception:
            pass
        _ASYNC_CLIENT = None

### allowed face states, from peerbots_types.py in api-samples
DEFAULT_COLOR = 'White'
VALID_EMOTIONS = ['Neutral', 'Surprised', 'Happy', 'Sad', 'Concerned', 'Sleepy']
VALID_COLORS = [
    'White',
    'Light Blue',
    'Blue',
    'Green',
    'Red',
    'Purple',
    'Pink',
    'Yellow',
    'Orange',
    'Grey',
    'Black',
]

_VALID_EMOTIONS_MAP = {e.lower(): e for e in VALID_EMOTIONS}
_VALID_COLORS_MAP = {c.lower(): c for c in VALID_COLORS}


### util
def _normalize(value, valid, fallback):
    # case-insensitive match against the api enum, fallback if unknown
    if not value:
        return fallback
    cleaned = str(value).strip()
    if isinstance(valid, dict):
        return valid.get(cleaned.lower(), fallback)
    cleaned_lower = cleaned.lower()
    for option in valid:
        if cleaned_lower == option.lower():
            return option
    return fallback


def _get_send_url(username):
    return f'{PEERBOTS_BASE_URL}/send-message/{username}'


def _headers():
    # read the key fresh so .env edits apply without a restart
    key = os.environ.get('PEERBOTS_API_KEY', '') or PEERBOTS_API_KEY
    return {'Accept': 'application/json', 'X-API-KEY': key}


def send_peerbots_message(speech='', emotion='Neutral', color=DEFAULT_COLOR, silent=False):
    """
    send one face message to the configured peerbots user (sync).

    inputs:
    speech: text for the face to speak + display (optional if silent)
    emotion: one of neutral, surprised, happy, sad, concerned, sleepy
    color: face glow color, defaults to white
    silent: if true, sets volume to 0.0 so face updates without tablet tts
    outputs:
    response dict from the api
    """
    username = os.environ.get('PEERBOTS_USERNAME', '') or PEERBOTS_USERNAME
    if not username:
        raise ValueError('PEERBOTS_USERNAME is not set. check your .env file.')
    if not (os.environ.get('PEERBOTS_API_KEY', '') or PEERBOTS_API_KEY):
        raise ValueError('PEERBOTS_API_KEY is not set. check your .env file.')

    safe_emotion = _normalize(emotion, _VALID_EMOTIONS_MAP, 'Neutral')
    safe_color = _normalize(color, _VALID_COLORS_MAP, DEFAULT_COLOR)
    text = (speech or '').strip()

    title = (text[:60] if len(text) <= 60 else text[:57] + '...') if text else 'Face Update'

    payload = {
        'title': title,
        'speech': text,
        'color': safe_color,
        'emotion': safe_emotion,
        'volume': 0.0 if silent else 1.0,
    }

    timeout = SILENT_TIMEOUT_S if silent else DEFAULT_TIMEOUT_S
    url = _get_send_url(username)
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=_headers(), json=payload)
        resp.raise_for_status()
        return resp.json()


async def asend_peerbots_message(speech='', emotion='Neutral', color=DEFAULT_COLOR, silent=False):
    """
    async send one face message via persistent keep-alive httpx client.
    catches all network/timeout errors gracefully to prevent background task crashes.

    inputs:
    speech: text for the face to speak + display
    emotion: one of neutral, surprised, happy, sad, concerned, sleepy
    color: face glow color, defaults to white
    silent: if true, sets volume to 0.0 with 8s timeout
    outputs:
    response dict from the api, or None on error
    """
    username = os.environ.get('PEERBOTS_USERNAME', '') or PEERBOTS_USERNAME
    if not username:
        return None
    if not (os.environ.get('PEERBOTS_API_KEY', '') or PEERBOTS_API_KEY):
        return None

    safe_emotion = _normalize(emotion, _VALID_EMOTIONS_MAP, 'Neutral')
    safe_color = _normalize(color, _VALID_COLORS_MAP, DEFAULT_COLOR)
    text = (speech or '').strip()

    title = (text[:60] if len(text) <= 60 else text[:57] + '...') if text else 'Face Update'

    payload = {
        'title': title,
        'speech': text,
        'color': safe_color,
        'emotion': safe_emotion,
        'volume': 0.0 if silent else 1.0,
    }

    timeout = SILENT_TIMEOUT_S if silent else DEFAULT_TIMEOUT_S
    url = _get_send_url(username)
    try:
        client = get_async_client()
        resp = await client.post(url, headers=_headers(), json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except httpx.TimeoutException:
        # peerbots cloud api can take 4-5s to deliver; don't log screaming traces
        return None
    except Exception as e:
        return None


def fire_peerbots_update(speech='', emotion='Neutral', color=DEFAULT_COLOR, silent=True):
    """safely schedule a background face update without blocking caller or leaking exceptions"""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(asend_peerbots_message(speech=speech, emotion=emotion, color=color, silent=silent))
    except RuntimeError:
        return None


if __name__ == '__main__':
    send_peerbots_message(
        speech='Hi! I am Lulo, your physical therapy buddy!',
        emotion='Happy',
        color='Green',
    )
