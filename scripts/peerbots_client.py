'''
peerbots rest client. sends speech + face state to the peerbots cloud,
which forwards it to the phone running the peerbots face panel.

layer 3: llm json -> peerbots_client -> face phone
'''

import os

import requests
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

### config
PEERBOTS_BASE_URL = 'https://api.peerbots.org/v1'
PEERBOTS_API_KEY = os.environ.get('PEERBOTS_API_KEY', '')
PEERBOTS_USERNAME = os.environ.get('PEERBOTS_USERNAME', '')

### allowed face states, from peerbots_types.py in api-samples
VALID_EMOTIONS = ['Neutral', 'Surprised', 'Happy', 'Sad', 'Concerned', 'Sleepy']
VALID_COLORS = [
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
    'White',
]


### util
def _normalize(value, valid, fallback):
    # case-insensitive match against the api enum, fallback if unknown
    if not value:
        return fallback
    cleaned = str(value).strip()
    for option in valid:
        if cleaned.lower() == option.lower():
            return option
    return fallback


def _headers():
    # read the key fresh so .env edits apply without a restart
    key = os.environ.get('PEERBOTS_API_KEY', '') or PEERBOTS_API_KEY
    return {'Accept': 'application/json', 'X-API-KEY': key}


def send_peerbots_message(speech: str = '', emotion: str = 'Neutral', color: str = 'Light Blue', silent: bool = False):
    '''
    send one face message to the configured peerbots user.

    inputs:
    speech: text for the face to speak + display (optional if silent)
    emotion: one of Neutral, Surprised, Happy, Sad, Concerned, Sleepy
    color: face glow color, defaults to Light Blue
    silent: if true, sets volume to 0.0 so face updates without tablet tts
    outputs:
    response dict from the api
    '''
    username = os.environ.get('PEERBOTS_USERNAME', '') or PEERBOTS_USERNAME
    if not username:
        raise ValueError('PEERBOTS_USERNAME is not set. check your .env file.')
    if not (os.environ.get('PEERBOTS_API_KEY', '') or PEERBOTS_API_KEY):
        raise ValueError('PEERBOTS_API_KEY is not set. check your .env file.')

    safe_emotion = _normalize(emotion, VALID_EMOTIONS, 'Neutral')
    safe_color = _normalize(color, VALID_COLORS, 'Light Blue')
    text = (speech or '').strip()

    # title for the face panel
    title = (text[:60] if len(text) <= 60 else text[:57] + '...') if text else 'Face Update'

    payload = {
        'title': title,
        'speech': text,
        'color': safe_color,
        'emotion': safe_emotion,
        'volume': 0.0 if silent else 1.0,
    }

    url = f'{PEERBOTS_BASE_URL}/send-message/{username}'
    resp = requests.post(url, headers=_headers(), json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


async def asend_peerbots_message(speech: str = '', emotion: str = 'Neutral', color: str = 'Light Blue', silent: bool = False):
    # async wrapper so the main loop can await without blocking the event loop
    import asyncio

    return await asyncio.to_thread(send_peerbots_message, speech, emotion, color, silent)


if __name__ == '__main__':
    send_peerbots_message(
        speech='Hi! I am Lulo, your physical therapy buddy!',
        emotion='Happy',
        color='Green',
    )
