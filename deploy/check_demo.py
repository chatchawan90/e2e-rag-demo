"""Check the browser and HTTP MCP against a running demo without paid API calls."""
import asyncio
import sys

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from websockets.asyncio.client import connect


async def check(base):
    async with httpx.AsyncClient(timeout=90) as client:
        health = await client.get(base + '/healthz')
        health.raise_for_status()
        print('Health:', health.json())
    ws_url = base.replace('https://', 'wss://').replace('http://', 'ws://') + '/_stcore/stream'
    async with connect(ws_url, subprotocols=['streamlit'], max_size=8_000_000, open_timeout=90) as ws:
        request = BackMsg()
        request.rerun_script.query_string = ''
        await ws.send(request.SerializeToString())
        public = False
        while True:
            message = ForwardMsg.FromString(await asyncio.wait_for(ws.recv(), timeout=90))
            if message.HasField('delta') and message.delta.HasField('new_element'):
                element = message.delta.new_element
                kind = element.WhichOneof('type')
                if kind == 'exception':
                    raise RuntimeError(element.exception.type + ': ' + element.exception.message)
                if kind == 'alert' and 'Public demo' in element.alert.body:
                    public = True
            if message.HasField('script_finished'):
                break
        assert public, 'Expected the public demo notice'
        print('Public browser interface: passed')
        async with streamable_http_client(base + '/mcp') as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = {tool.name for tool in (await session.list_tools()).tools}
                assert names == {'search_regulations', 'get_passage', 'list_documents'}, names
                print('Public HTTP MCP tools: passed')
                docs = await session.call_tool('list_documents', {})
                assert not docs.isError
                for query in ['มาตรฐานน้ำทิ้ง บีโอดี', 'What is the BOD limit for wastewater in Thailand?', 'wastewater ' * 500]:
                    result = await session.call_tool('search_regulations', {'query': query, 'mode': 'hybrid', 'k': 3})
                    assert not result.isError, result
                    assert any('chunk_id=' in getattr(c, 'text', '') for c in result.content)
                    print('Hybrid search passed; query length:', len(query))


if __name__ == '__main__':
    asyncio.run(asyncio.wait_for(check(sys.argv[1].rstrip('/')), timeout=240))
