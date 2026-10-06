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


def get_client():
    # lazy init so importing this module never crashes on missing keys
    if USE_LOCAL_LLM:
        return AsyncOpenAI(base_url=OLLAMA_URL, api_key='ollama')
    key = os.environ.get('OPENAI_API_KEY', '') or OPENAI_API_KEY
    return AsyncOpenAI(api_key=key)


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

REPLY_SCHEMA = {
    'name': 'peerbots_reply',
    'strict': True,
    'schema': {
        'type': 'object',
        'properties': {
            'emotion': {'type': 'string', 'enum': VALID_EMOTIONS},
            'color': {'type': 'string', 'enum': VALID_COLORS},
            'speech': {'type': 'string'},
        },
        'required': ['emotion', 'color', 'speech'],
        'additionalProperties': False,
    },
}


class peerbots_reply(BaseModel):
    # strict shape the face api needs
    speech: str
    emotion: str
    color: str


### history
def get_history():
    data = safe_read(HISTORY_FILE)
    if not data:
        return []
    try:
        parsed = json.loads(data)
        return parsed if isinstance(parsed, list) else []
    except Exception as e:
        console.print(f'[bold red][[ error parsing history: {e} ]][/]')
        return []


def save_history(chat_history):
    if len(chat_history) > HISTORY_LENGTH:
        chat_history = chat_history[-HISTORY_LENGTH:]
    safe_write(HISTORY_FILE, json.dumps(chat_history, indent=2))


def load_system_prompt():
    # fresh read each turn so prompt edits apply without restart
    path = os.path.join(BASE_DIR, 'config', 'system_prompt.md')
    prompt = safe_read(path)
    if '{CONTEXT_DIR}' in prompt:
        prompt = prompt.replace('{CONTEXT_DIR}', os.path.join(BASE_DIR, 'context'))
    # optional skills plumbing: every skills/*/SKILL.md is appended verbatim.
    # no skills installed yet, drop a SKILL.md in skills/ to add one.
    for skill_file in sorted(glob.glob(os.path.join(SKILLS_DIR, '*', 'SKILL.md'))):
        body = safe_read(skill_file)
        if body:
            prompt += f'\n\n---\n\n# skill: {os.path.basename(os.path.dirname(skill_file))}\n\n{body}'
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
    for option in valid:
        if cleaned.lower() == option.lower():
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
            color=_normalize(data.get('color'), VALID_COLORS, 'Light Blue'),
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


class StreamingReplyParser:
    '''
    incremental parser for structured json:
    extracts emotion and color early, yields speech sentences on boundaries.
    '''

    def __init__(self):
        self.raw_text = ''
        self.emotion = None
        self.color = None
        self.speech_buffer = ''
        self.speech_started = False
        self.speech_finished = False


    def feed(self, delta):
        self.raw_text += delta
        if not self.emotion:
            m = re.search(r'"emotion"\s*:\s*"([^"]+)"', self.raw_text)
            if m:
                self.emotion = _normalize(m.group(1), VALID_EMOTIONS, 'Neutral')
        if not self.color:
            m = re.search(r'"color"\s*:\s*"([^"]+)"', self.raw_text)
            if m:
                self.color = _normalize(m.group(1), VALID_COLORS, 'Light Blue')
        if not self.speech_started:
            m = re.search(r'"speech"\s*:\s*"', self.raw_text)
            if m:
                self.speech_started = True
                self.speech_buffer = self.raw_text[m.end():]
        else:
            if not self.speech_finished:
                self.speech_buffer += delta

        sentences = []
        if self.speech_started and not self.speech_finished:
            while True:
                end_match = re.search(r'(?<!\\)"', self.speech_buffer)
                if end_match:
                    self.speech_finished = True
                    part = self.speech_buffer[:end_match.start()].strip()
                    part = part.replace('\\"', '"').replace('\\n', ' ')
                    if part:
                        sentences.append(part)
                    self.speech_buffer = ''
                    break
                # match sentence boundaries (.!?) or natural clause boundaries (;, or comma if 4+ words accumulated)
                boundary = re.search(r'([.!?]+|[;:])\s+', self.speech_buffer)
                if not boundary and len(self.speech_buffer.split()) >= 5:
                    boundary = re.search(r'(,\s+)', self.speech_buffer)

                if boundary:
                    split_pos = boundary.end()
                    sentence = self.speech_buffer[:boundary.start(1) + len(boundary.group(1).rstrip())].strip()
                    sentence = sentence.replace('\\"', '"').replace('\\n', ' ')
                    self.speech_buffer = self.speech_buffer[split_pos:]
                    if sentence:
                        sentences.append(sentence)
                else:
                    break
        return sentences


    def finish(self):
        if not self.speech_buffer:
            return []
        leftover = re.sub(r'["\}\s]+$', '', self.speech_buffer).strip()
        leftover = leftover.replace('\\"', '"').replace('\\n', ' ')
        self.speech_buffer = ''
        if leftover and len(leftover) > 1:
            return [leftover]
        return []


def _base_kwargs(messages):
    kwargs = {
        'model': MODEL,
        'messages': messages,
        'timeout': 30.0,
        'max_completion_tokens': 90,
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
    - {'type': 'sentence', 'text': sentence}
    - {'type': 'final', 'reply': reply_dict, 'metrics': metrics_dict}
    '''
    tools_list = tools_list or []
    tool_router = tool_router or {}
    system_prompt = load_system_prompt()
    chat_history = get_history()
    chat_history.append({'role': 'user', 'content': user_text})
    messages = [{'role': 'system', 'content': system_prompt}] + chat_history

    start_t = time.time()
    try:
        iters = 0
        while iters < MAX_TOOL_ITERS:
            kwargs = _base_kwargs(messages)
            kwargs['stream'] = True
            if tools_list and not USE_LOCAL_LLM:
                kwargs['tools'] = tools_list

            stream = await get_client().chat.completions.create(**kwargs)
            parser = StreamingReplyParser()
            accumulated_tool_calls = {}
            face_sent = False
            ttft_ms = None

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
                    if ttft_ms is None:
                        ttft_ms = int((time.time() - start_t) * 1000)
                    sentences = parser.feed(d.content)
                    if not face_sent and parser.emotion and parser.color:
                        face_sent = True
                        yield {'type': 'face', 'emotion': parser.emotion, 'color': parser.color}
                    for s in sentences:
                        yield {'type': 'sentence', 'text': s}

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
                yield {'type': 'sentence', 'text': s}

            reply = parse_reply(parser.raw_text)
            if not face_sent:
                yield {'type': 'face', 'emotion': reply['emotion'], 'color': reply['color']}

            elapsed_ms = int((time.time() - start_t) * 1000)
            yield {
                'type': 'final',
                'reply': reply,
                'metrics': {
                    'ttft_ms': ttft_ms or elapsed_ms,
                    'total_ms': elapsed_ms,
                },
            }
            chat_history.append({'role': 'assistant', 'content': json.dumps(reply)})
            save_history(chat_history)
            return

        fallback = _fallback_reply()
        yield {'type': 'face', 'emotion': fallback['emotion'], 'color': fallback['color']}
        yield {'type': 'sentence', 'text': fallback['speech']}
        yield {'type': 'final', 'reply': fallback, 'metrics': {'ttft_ms': 0, 'total_ms': 0}}
    except Exception as e:
        console.print(f'[bold red][[ openai streaming error: {e} ]][/]')
        fallback = _fallback_reply()
        yield {'type': 'face', 'emotion': fallback['emotion'], 'color': fallback['color']}
        yield {'type': 'sentence', 'text': fallback['speech']}
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
