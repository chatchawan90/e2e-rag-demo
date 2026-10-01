"""Public, read-only MCP over Streamable HTTP; no hosted answer API calls."""
from __future__ import annotations

import argparse
import os
from functools import partial
from typing import Annotated, Literal
from urllib.parse import urlsplit

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field
from anyio import CapacityLimiter, to_thread
from starlette.responses import JSONResponse

from . import mcp_server


def create_app(public_url: str = ''):
    hosts = ['127.0.0.1:*', 'localhost:*']
    origins = ['http://127.0.0.1:*', 'http://localhost:*']
    if public_url:
        url = urlsplit(public_url)
        if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in ('', '/'):
            raise ValueError('Public URL must be an HTTPS origin, such as https://demo.example.com')
        hosts.append(url.netloc)
        origins.append(f'https://{url.netloc}')
    server = FastMCP('envsearch-demo', stateless_http=True, json_response=True,
                     max_request_body_size=65536,
                     instructions='Search this public reference library and cite document titles and pages. These documents may not reflect current law. No user uploads or paid answer tools are exposed.',
                     transport_security=TransportSecuritySettings(allowed_hosts=hosts, allowed_origins=origins))
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
    retrieval_limit = CapacityLimiter(1)

    @server.tool(annotations=annotations)
    async def search_regulations(query: Annotated[str, Field(min_length=1, max_length=6000)],
                           k: Annotated[int, Field(ge=1, le=15)] = 6,
                           language: Literal['any', 'en', 'th'] = 'any',
                           jurisdiction: Annotated[str, Field(max_length=40)] | None = None,
                           doc_ids: Annotated[list[str], Field(max_length=30)] | None = None,
                           mode: Literal['auto', 'bm25', 'vector', 'hybrid'] = 'auto') -> str:
        """Search public US/Thai environmental regulations; return passages, source titles, pages and chunk IDs. auto uses hybrid if vectors are present. Keep language=any for cross-language search. No API key needed."""
        return await to_thread.run_sync(partial(mcp_server.search_regulations, query, k=k, language=language,
                                               jurisdiction=jurisdiction, doc_ids=doc_ids, mode=mode,
                                               rerank=False, candidate_k=32, per_doc_cap=3), limiter=retrieval_limit)

    @server.tool(annotations=annotations)
    def get_passage(chunk_id: Annotated[str, Field(min_length=1, max_length=200)],
                    context: Annotated[int, Field(ge=0, le=3)] = 1) -> str:
        """Read a returned chunk ID plus nearby passages from its source document."""
        return mcp_server.get_passage(chunk_id, context)

    server.add_tool(mcp_server.list_documents, annotations=annotations)

    @server.custom_route('/healthz', methods=['GET'])
    async def health(_request):
        try:
            index = mcp_server._index(load_vectors=False)
            if not index.chunks:
                return JSONResponse({'status': 'empty-library'}, status_code=503)
            return JSONResponse({'status': 'ok', 'documents': len(index.docs), 'passages': len(index.chunks)})
        except (ValueError, OSError):
            return JSONResponse({'status': 'index-unavailable'}, status_code=503)

    return server.streamable_http_app()


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8602')))
    parser.add_argument('--public-url', default=os.environ.get('ENVSEARCH_MCP_PUBLIC_URL', ''))
    args = parser.parse_args()
    uvicorn.run(create_app(args.public_url), host=args.host, port=args.port, access_log=False, limit_concurrency=16)


if __name__ == '__main__':
    main()
