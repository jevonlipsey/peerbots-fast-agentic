'''
boots every server in mcp_config.json and builds the openai tool
registry plus a name -> session router. generic servers only
(fetch, time, memory, filesystem). no robot actuation here.
'''

import json
import os
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _resolve(val, base_dir):
    # './x' paths are relative to the config file, everything else passes through
    if isinstance(val, str) and val.startswith('./'):
        return os.path.abspath(os.path.join(base_dir, val[2:]))
    return val


async def load_and_register_mcp_servers(stack: AsyncExitStack, config_path: str):
    '''
    boot all configured servers concurrently via the shared exit stack.

    inputs:
    stack: async exit stack that owns server lifetimes
    config_path: path to mcp_config.json
    outputs:
    (tools_list, tool_router) for the openai chat api
    '''
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(config_path))
    tools_list = []
    tool_router = {}

    devnull = open(os.devnull, 'w')
    for server_name, server_config in config.get('mcpServers', {}).items():
        command = server_config['command']
        args = [_resolve(a, base_dir) for a in server_config.get('args', [])]
        env = dict(os.environ)
        for key, val in server_config.get('env', {}).items():
            env[key] = _resolve(val, base_dir)
        params = StdioServerParameters(command=command, args=args, env=env)

        transport = await stack.enter_async_context(
            stdio_client(params, errlog=devnull)
        )
        session = await stack.enter_async_context(
            ClientSession(transport[0], transport[1])
        )
        await session.initialize()

        result = await session.list_tools()
        for tool in result.tools:
            # mcp 1.x calls it inputSchema, 2.x renamed it input_schema
            schema = getattr(tool, 'input_schema', None) or getattr(
                tool, 'inputSchema', None
            )
            tools_list.append(
                {
                    'type': 'function',
                    'function': {
                        'name': tool.name,
                        'description': tool.description or '',
                        'parameters': schema or {'type': 'object'},
                    },
                }
            )
            # last server wins on name collisions
            tool_router[tool.name] = session

    return tools_list, tool_router
