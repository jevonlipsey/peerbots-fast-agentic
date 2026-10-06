'''
cognitive layer for lulo. gathers facts via mcp tools when needed,
then always synthesizes strict json (speech, emotion, color) for
the peerbots face api.

layer 2: transcription -> openai_response -> peerbots_client
'''

import asyncio
import glob
import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from rich.console import Console

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.file_utils import safe_read, safe_write

load_dotenv()
console = Console()

### config
USE_LOCAL_LLM = False
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OLLAMA_URL = 'http://localhost:11434/v1'
MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4.1-nano')

if USE_LOCAL_LLM:
    MODEL = 'gemma4:e4b'

_CLIENT = None
_KEEPALIVE_TASK = None


def get_client():
    # cached persistent client for low-latency connection reuse
    global _CLIENT
    if _CLIENT is None:
        if USE_LOCAL_LLM:
            _CLIENT = AsyncOpenAI(base_url=OLLAMA_URL, api_key='ollama')
        else:
            key = os.environ.get('OPENAI_API_KEY', '') or OPENAI_API_KEY
            _CLIENT = AsyncOpenAI(api_key=key)
    return _CLIENT


async def _keepalive_worker():
    # pings openai models.list every 45s to keep the tls connection pool hot during silence
    while True:
        try:
            await asyncio.sleep(45)
            client = get_client()
            await client.models.list()
        except asyncio.CancelledError:
            break
        except Exception:
            pass


def start_openai_keepalive(loop=None):
    global _KEEPALIVE_TASK
    if USE_LOCAL_LLM:
        return None
    if _KEEPALIVE_TASK is None or _KEEPALIVE_TASK.done():
        active_loop = loop or asyncio.get_running_loop()
        _KEEPALIVE_TASK = active_loop.create_task(_keepalive_worker())
    return _KEEPALIVE_TASK


def stop_openai_keepalive():
    global _KEEPALIVE_TASK
    if _KEEPALIVE_TASK is not None and not _KEEPALIVE_TASK.done():
        _KEEPALIVE_TASK.cancel()
        _KEEPALIVE_TASK = None


async def close_client():
    # close client on shutdown
    global _CLIENT
    stop_openai_keepalive()
    if _CLIENT is not None:
        try:
            await _CLIENT.close()
        except Exception:
            pass
        _CLIENT = None


async def prewarm_openai():
    # warm up dns, tls, and connection pool before the first turn
    if USE_LOCAL_LLM:
        return
    try:
        client = get_client()
        await client.models.list()
    except Exception as e:
        console.print(f'[dim yellow][[ openai prewarm failed: {e} ]][/]')


def _needs_reasoning_none(model):
    # reasoning-family models refuse tools/structured calls unless this is set
    m = (model or '').lower()
    return 'gpt-5' in m or 'gpt-6' in m or m.startswith('o')

HISTORY_LENGTH = 20
MAX_TOOL_ITERS = 4

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(BASE_DIR, 'state')
HISTORY_FILE = os.path.join(STATE_DIR, 'history.txt')
MCP_CONFIG_PATH = os.path.join(BASE_DIR, 'mcp_config.json')
SKILLS_DIR = os.path.join(BASE_DIR, 'skills')

### allowed face states, must match peerbots api
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

_VALID_EMOTIONS_MAP = {e.lower(): e for e in VALID_EMOTIONS}
_VALID_COLORS_MAP = {c.lower(): c for c in VALID_COLORS}

# pre-compiled regex patterns for streaming json parsing
_RE_EMOTION = re.compile(r'"emotion"\s*:\s*"([^"]+)"')
_RE_COLOR = re.compile(r'"color"\s*:\s*"([^"]+)"')
_RE_SPEECH_START = re.compile(r'"speech"\s*:\s*"')
_RE_UNESCAPED_QUOTE = re.compile(r'(?<!\\)"')
_RE_TERMINAL_PUNCT = re.compile(r'([.!?]+)\s+')
_RE_CLAUSE_PUNCT = re.compile(r'(?:([;:,])\s+|([—–]|--)\s*)')
_RE_LEFTOVER_CLEANUP = re.compile(r'["\}\s]+$')

REPLY_SCHEMA = {
    'name': 'peerbots_reply',
    'strict': True,
    'schema': {
        'type': 'object',
        'properties': {
            'speech': {'type': 'string'},
            'emotion': {'type': 'string', 'enum': VALID_EMOTIONS},
            'color': {'type': 'string', 'enum': VALID_COLORS},
        },
        'required': ['speech', 'emotion', 'color'],
        'additionalProperties': False,
    },
}


class peerbots_reply(BaseModel):
    # strict shape the face api needs
    speech: str
    emotion: str
    color: str


### history
_IN_MEMORY_HISTORY = None


def get_history():
    # cached in-memory history, loads once from disk
    global _IN_MEMORY_HISTORY
    if _IN_MEMORY_HISTORY is None:
        data = safe_read(HISTORY_FILE)
        if not data:
            _IN_MEMORY_HISTORY = []
        else:
            try:
                parsed = json.loads(data)
                _IN_MEMORY_HISTORY = parsed if isinstance(parsed, list) else []
            except Exception as e:
                console.print(f'[bold red][[ error parsing history: {e} ]][/]')
                _IN_MEMORY_HISTORY = []
    return list(_IN_MEMORY_HISTORY)


def save_history(chat_history):
    # updates in-memory history immediately and flushes to disk in background thread
    global _IN_MEMORY_HISTORY
    if len(chat_history) > HISTORY_LENGTH:
        chat_history = chat_history[-HISTORY_LENGTH:]
    _IN_MEMORY_HISTORY = list(chat_history)
    import threading
    payload = json.dumps(_IN_MEMORY_HISTORY, indent=2)
    threading.Thread(target=safe_write, args=(HISTORY_FILE, payload), daemon=True).start()


_CACHED_PROMPT = None
_CACHED_PROMPT_MTIME = 0
_CACHED_SKILLS_MTIME = 0


def load_system_prompt():
    # cached read, only reloads if prompt file or skills change
    global _CACHED_PROMPT, _CACHED_PROMPT_MTIME, _CACHED_SKILLS_MTIME
    path = os.path.join(BASE_DIR, 'config', 'system_prompt.md')
    try:
        mtime = os.path.getmtime(path) if os.path.exists(path) else 0
    except OSError:
        mtime = 0

    skill_files = sorted(glob.glob(os.path.join(SKILLS_DIR, '*', 'SKILL.md')))
    skills_mtime = max([os.path.getmtime(f) for f in skill_files], default=0)

    if _CACHED_PROMPT is not None and mtime == _CACHED_PROMPT_MTIME and skills_mtime == _CACHED_SKILLS_MTIME:
        return _CACHED_PROMPT

    prompt = safe_read(path) or ''
    if '{CONTEXT_DIR}' in prompt:
        prompt = prompt.replace('{CONTEXT_DIR}', os.path.join(BASE_DIR, 'context'))
    for skill_file in skill_files:
        body = safe_read(skill_file)
        if body:
            prompt += f'\n\n---\n\n# skill: {os.path.basename(os.path.dirname(skill_file))}\n\n{body}'

    _CACHED_PROMPT = prompt
    _CACHED_PROMPT_MTIME = mtime
    _CACHED_SKILLS_MTIME = skills_mtime
    return prompt


### mcp plumbing
async def connect_mcp(stack):
    # boot once per process, never raises. returns ([], {}) when unusable.
    try:
        from lib.mcp_loader import load_and_register_mcp_servers
    except Exception as e:
        console.print(f'[dim yellow][[ mcp loader unavailable: {e} ]][/]')
        return [], {}
    if not os.path.exists(MCP_CONFIG_PATH):
        return [], {}
    try:
        tools, router = await load_and_register_mcp_servers(stack, MCP_CONFIG_PATH)
        console.print('[dim white][[ SYSTEM: MCP tools ready. ]][/]')
        return tools, router
    except Exception as e:
        console.print(f'[dim yellow][[ mcp boot failed, continuing without tools: {e} ]][/]')
        return [], {}


def _parse_args(raw):
    # tool args arrive as a json string, never let one bad blob kill the turn
    try:
        parsed = json.loads(raw or '{}')
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _run_tool(name, arguments, tool_router):
    # one tool call -> plain text for the model
    session = tool_router.get(name)
    if session is None:
        return f'Error: tool {name} not found.'
    try:
        result = await session.call_tool(name, arguments=arguments or {})
    except Exception as e:
        return f'Error executing tool {name}: {e}'
    texts = []
    for item in getattr(result, 'content', None) or []:
        text = getattr(item, 'text', None)
        if isinstance(text, str) and text:
            texts.append(text[:1500])
    return '\n'.join(texts) if texts else 'Tool executed successfully with no output.'


### core
def _normalize(value, valid, fallback):
    # snap free-form llm output onto the api enum
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


def _fallback_reply():
    # safe default when the llm returns bad json
    return {
        'speech': 'Hmm, my mushroom brain glitched for a sec. Could you say that again?',
        'emotion': 'Concerned',
        'color': 'Yellow',
    }


def parse_reply(raw_text):
    # pull exactly speech, emotion, color out of raw llm text
    if not raw_text:
        return _fallback_reply()
    cleaned = raw_text.strip()
    # strip markdown fences if the model adds them despite json mode
    if cleaned.startswith('```'):
        cleaned = cleaned.strip('`').strip()
        if cleaned.lower().startswith('json'):
            cleaned = cleaned[4:].strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # last resort: find the first {...} block
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start == -1 or end == -1:
            return _fallback_reply()
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return _fallback_reply()
    try:
        validated = peerbots_reply(
            speech=str(data.get('speech', '')).strip(),
            emotion=_normalize(data.get('emotion'), VALID_EMOTIONS, 'Neutral'),
            color=_normalize(data.get('color'), VALID_COLORS, 'White'),
        )
    except (ValidationError, AttributeError):
        return _fallback_reply()
    if not validated.speech:
        return _fallback_reply()
    return {
        'speech': validated.speech,
        'emotion': validated.emotion,
        'color': validated.color,
    }


### chunking thresholds
MIN_FIRST_CHUNK_WORDS = 3
MIN_FIRST_CHUNK_CHARS = 14
MIN_CLAUSE_CHUNK_WORDS = 4


class StreamingReplyParser:
    """
    incremental parser for structured json:
    extracts emotion and color, yields speech clauses on boundaries.
    """

    def __init__(self):
        self.raw_text = ''
        self.emotion = None
        self.color = None
        self.speech_buffer = ''
        self.speech_started = False
        self.speech_finished = False
        self.first_chunk_sent = False


    def feed(self, delta):
        self.raw_text += delta
        if not self.emotion:
            m = _RE_EMOTION.search(self.raw_text)
            if m:
                self.emotion = _normalize(m.group(1), _VALID_EMOTIONS_MAP, 'Neutral')
        if not self.color:
            m = _RE_COLOR.search(self.raw_text)
            if m:
                self.color = _normalize(m.group(1), _VALID_COLORS_MAP, 'White')
        if not self.speech_started:
            m = _RE_SPEECH_START.search(self.raw_text)
            if m:
                self.speech_started = True
                self.speech_buffer = self.raw_text[m.end():]
        else:
            if not self.speech_finished:
                self.speech_buffer += delta

        sentences = []
        if self.speech_started and not self.speech_finished:
            while True:
                end_match = _RE_UNESCAPED_QUOTE.search(self.speech_buffer)
                if end_match:
                    self.speech_finished = True
                    part = self.speech_buffer[:end_match.start()].strip()
                    part = part.replace('\\"', '"').replace('\\n', ' ')
                    if part:
                        sentences.append(part)
                        self.first_chunk_sent = True
                    self.speech_buffer = ''
                    break

                split_pos = None
                sentence = None

                if not self.first_chunk_sent:
                    # chunk 0: require enough speech buffer (>=3 words or >=14 chars) so audio duration
                    # (~1.5-2.0s) seamlessly hides synthesis of chunk 1, avoiding buffer starvation.
                    term_match = _RE_TERMINAL_PUNCT.search(self.speech_buffer)
                    if term_match:
                        cand = self.speech_buffer[:term_match.start(1) + len(term_match.group(1).rstrip())].strip()
                        if len(cand.split()) >= MIN_FIRST_CHUNK_WORDS or len(cand) >= MIN_FIRST_CHUNK_CHARS:
                            split_pos = term_match.end()
                            sentence = cand
                    if split_pos is None:
                        for cm in _RE_CLAUSE_PUNCT.finditer(self.speech_buffer):
                            punct = cm.group(1) or cm.group(2)
                            cand = self.speech_buffer[:cm.start() + len(punct)].strip()
                            if len(cand.split()) >= MIN_FIRST_CHUNK_WORDS or len(cand) >= MIN_FIRST_CHUNK_CHARS:
                                split_pos = cm.end()
                                sentence = cand
                                break
                else:
                    # subsequent chunks: emit on terminal [.!?] or clause boundaries if >= 4 words
                    term_match = _RE_TERMINAL_PUNCT.search(self.speech_buffer)
                    if term_match:
                        cand = self.speech_buffer[:term_match.start(1) + len(term_match.group(1).rstrip())].strip()
                        split_pos = term_match.end()
                        sentence = cand
                    else:
                        for cm in _RE_CLAUSE_PUNCT.finditer(self.speech_buffer):
                            punct = cm.group(1) or cm.group(2)
                            cand = self.speech_buffer[:cm.start() + len(punct)].strip()
                            if len(cand.split()) >= MIN_CLAUSE_CHUNK_WORDS:
                                split_pos = cm.end()
                                sentence = cand
                                break

                if split_pos is not None and sentence:
                    sentence = sentence.replace('\\"', '"').replace('\\n', ' ')
                    self.speech_buffer = self.speech_buffer[split_pos:]
                    sentences.append(sentence)
                    self.first_chunk_sent = True
                else:
                    break
        return sentences


    def finish(self):
        if not self.speech_buffer:
            return []
        leftover = _RE_LEFTOVER_CLEANUP.sub('', self.speech_buffer).strip()
        leftover = leftover.replace('\\"', '"').replace('\\n', ' ')
        self.speech_buffer = ''
        if leftover and len(leftover) > 1:
            return [leftover]
        return []


### session state tracking
_SESSION_TURN_COUNT = 0
_RECENT_OPENERS = []
_RECENT_EMOTIONS = []


def reset_session_context():
    # resets session turn tracking and anti-repetition memory
    global _SESSION_TURN_COUNT, _RECENT_OPENERS, _RECENT_EMOTIONS
    _SESSION_TURN_COUNT = 0
    _RECENT_OPENERS = []
    _RECENT_EMOTIONS = []


def get_session_context():
    return {
        'turn_count': _SESSION_TURN_COUNT,
        'recent_openers': list(_RECENT_OPENERS),
        'recent_emotions': list(_RECENT_EMOTIONS),
    }


def _build_session_prefix():
    global _SESSION_TURN_COUNT
    _SESSION_TURN_COUNT += 1

    hints = []
    if _RECENT_OPENERS:
        last_openers = _RECENT_OPENERS[-3:]
        openers_str = ', '.join(f"'{o}'" for o in last_openers)
        hints.append(f'Avoid recent openers: {openers_str}')

    if len(_RECENT_EMOTIONS) >= 3 and len(set(_RECENT_EMOTIONS[-3:])) == 1:
        rep_emotion = _RECENT_EMOTIONS[-1]
        hints.append(f"Emotion dampening: you used '{rep_emotion}' for 3 turns in a row; vary your affect if appropriate")

    hint_text = f" ({'; '.join(hints)})" if hints else ''
    return f'[session: turn {_SESSION_TURN_COUNT}]{hint_text}\n\n'


def _record_turn_metadata(speech, emotion):
    global _RECENT_OPENERS, _RECENT_EMOTIONS
    if speech:
        words = speech.strip().split()
        if words:
            opener = ' '.join(words[:min(3, len(words))])
            _RECENT_OPENERS.append(opener)
            if len(_RECENT_OPENERS) > 5:
                _RECENT_OPENERS.pop(0)

    if emotion:
        _RECENT_EMOTIONS.append(emotion)
        if len(_RECENT_EMOTIONS) > 5:
            _RECENT_EMOTIONS.pop(0)


def _base_kwargs(messages):
    kwargs = {
        'model': MODEL,
        'messages': messages,
        'timeout': 30.0,
        'max_completion_tokens': 140,
    }
    if USE_LOCAL_LLM:
        kwargs['extra_body'] = {'options': {'num_ctx': 4096}, 'format': 'json'}
    else:
        if _needs_reasoning_none(MODEL):
            kwargs['reasoning_effort'] = 'none'
        kwargs['response_format'] = {
            'type': 'json_schema',
            'json_schema': REPLY_SCHEMA,
        }
    return kwargs


async def stream_peerbots_reply(user_text, tools_list=None, tool_router=None):
    '''
    streams lulo reply tokens incrementally.

    inputs:
    user_text: transcribed patient utterance
    tools_list: openai tool specs from connect_mcp (or none)
    tool_router: tool name -> mcp session from connect_mcp (or none)
    outputs:
    yields event dicts:
    - {'type': 'face', 'emotion': emotion, 'color': color}
    - {'type': 'sentence', 'text': sentence, 'idx': chunk_index}
    - {'type': 'final', 'reply': reply_dict, 'metrics': metrics_dict}
    '''
    tools_list = tools_list or []
    tool_router = tool_router or {}
    system_prompt = load_system_prompt()
    session_prefix = _build_session_prefix()
    system_content = session_prefix + system_prompt
    chat_history = get_history()
    chat_history.append({'role': 'user', 'content': user_text})
    messages = [{'role': 'system', 'content': system_content}] + chat_history

    start_t = time.time()
    try:
        iters = 0
        while iters < MAX_TOOL_ITERS:
            kwargs = _base_kwargs(messages)
            kwargs['stream'] = True
            if tools_list and not USE_LOCAL_LLM:
                kwargs['tools'] = tools_list

            t_api_start = time.time()
            stream = await get_client().chat.completions.create(**kwargs)
            parser = StreamingReplyParser()
            accumulated_tool_calls = {}
            face_sent = False
            t_first_token = None
            t_first_clause = None
            t_face_sent = None
            chunk_idx = 0

            async for chunk in stream:
                if not chunk.choices:
                    continue
                d = chunk.choices[0].delta
                if d.tool_calls:
                    for tc in d.tool_calls:
                        idx = tc.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                'id': tc.id or '',
                                'name': tc.function.name or '',
                                'arguments': '',
                            }
                        if tc.id:
                            accumulated_tool_calls[idx]['id'] = tc.id
                        if tc.function.name:
                            accumulated_tool_calls[idx]['name'] = tc.function.name
                        if tc.function.arguments:
                            accumulated_tool_calls[idx]['arguments'] += tc.function.arguments

                if d.content:
                    if t_first_token is None:
                        t_first_token = time.time()
                    sentences = parser.feed(d.content)
                    if not face_sent and parser.emotion and parser.color:
                        face_sent = True
                        t_face_sent = time.time()
                        yield {'type': 'face', 'emotion': parser.emotion, 'color': parser.color}
                    for s in sentences:
                        if t_first_clause is None:
                            t_first_clause = time.time()
                        yield {'type': 'sentence', 'text': s, 'idx': chunk_idx}
                        chunk_idx += 1

            # if tools were invoked during this stream, execute and loop
            if accumulated_tool_calls:
                tool_list = [accumulated_tool_calls[k] for k in sorted(accumulated_tool_calls.keys())]
                tools_str = ', '.join(t['name'] for t in tool_list)
                console.print(f'  [dim white]-> \\[tools] {tools_str}[/]')
                assistant_msg = {
                    'role': 'assistant',
                    'tool_calls': [
                        {
                            'id': t['id'],
                            'type': 'function',
                            'function': {'name': t['name'], 'arguments': t['arguments']},
                        }
                        for t in tool_list
                    ],
                }
                messages.append(assistant_msg)
                results = await asyncio.gather(
                    *(
                        _run_tool(
                            t['name'],
                            _parse_args(t['arguments']),
                            tool_router,
                        )
                        for t in tool_list
                    )
                )
                for t, output in zip(tool_list, results):
                    messages.append(
                        {
                            'role': 'tool',
                            'tool_call_id': t['id'],
                            'name': t['name'],
                            'content': output,
                        }
                    )
                iters += 1
                continue

            # finalize conversational stream
            for s in parser.finish():
                if t_first_clause is None:
                    t_first_clause = time.time()
                yield {'type': 'sentence', 'text': s, 'idx': chunk_idx}
                chunk_idx += 1

            reply = parse_reply(parser.raw_text)
            if not face_sent:
                t_face_sent = time.time()
                yield {'type': 'face', 'emotion': reply['emotion'], 'color': reply['color']}

            elapsed_ms = int((time.time() - start_t) * 1000)
            ttft_ms = int((t_first_token - t_api_start) * 1000) if t_first_token else elapsed_ms
            first_clause_ms = int((t_first_clause - t_api_start) * 1000) if t_first_clause else 0
            face_ms = int((t_face_sent - t_api_start) * 1000) if t_face_sent else 0
            api_total_ms = int((time.time() - t_api_start) * 1000)

            yield {
                'type': 'final',
                'reply': reply,
                'metrics': {
                    'ttft_ms': ttft_ms,
                    'first_clause_ms': first_clause_ms,
                    'face_ms': face_ms,
                    'api_total_ms': api_total_ms,
                    'total_ms': elapsed_ms,
                },
            }
            _record_turn_metadata(reply.get('speech'), reply.get('emotion'))
            chat_history.append({'role': 'assistant', 'content': json.dumps(reply)})
            save_history(chat_history)
            return

        fallback = _fallback_reply()
        yield {'type': 'face', 'emotion': fallback['emotion'], 'color': fallback['color']}
        yield {'type': 'sentence', 'text': fallback['speech'], 'idx': 0}
        yield {'type': 'final', 'reply': fallback, 'metrics': {'ttft_ms': 0, 'total_ms': 0}}
    except Exception as e:
        console.print(f'[bold red][[ openai streaming error: {e} ]][/]')
        fallback = _fallback_reply()
        yield {'type': 'face', 'emotion': fallback['emotion'], 'color': fallback['color']}
        yield {'type': 'sentence', 'text': fallback['speech'], 'idx': 0}
        yield {'type': 'final', 'reply': fallback, 'metrics': {'ttft_ms': 0, 'total_ms': 0}}


async def get_peerbots_reply(user_text, tools_list=None, tool_router=None):
    '''
    gather full reply from stream for non-streaming consumers

    inputs:
    user_text: transcribed patient utterance
    tools_list: openai tool specs from connect_mcp (or none)
    tool_router: tool name -> mcp session from connect_mcp (or none)
    outputs:
    reply dict matching the peerbots api shape
    '''
    final_reply = None
    async for event in stream_peerbots_reply(user_text, tools_list, tool_router):
        if event['type'] == 'final':
            final_reply = event['reply']
    return final_reply or _fallback_reply()


async def _cli_test():
    # quick manual check: python scripts/openai_response.py "hello lulo"
    from contextlib import AsyncExitStack

    text = ' '.join(sys.argv[1:]) or 'hi lulo, i am ready to exercise'
    async with AsyncExitStack() as stack:
        tools, router = await connect_mcp(stack)
        reply = await get_peerbots_reply(text, tools, router)
    print(json.dumps(reply, indent=2))


if __name__ == '__main__':
    if not USE_LOCAL_LLM and not OPENAI_API_KEY:
        console.print('[bold red][[ OPENAI_API_KEY is not set ]][/]')
        sys.exit(1)
    asyncio.run(_cli_test())
