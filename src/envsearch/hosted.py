"""One-process browser + MCP demo, sharing a compact embedding model."""
import os
from pathlib import Path
import subprocess
import threading
from urllib.parse import urlsplit

from .config import REPO_ROOT


def main():
    if os.environ.get('ENVSEARCH_PUBLIC_DEMO') != '1':
        raise RuntimeError('Hosted entrypoint requires ENVSEARCH_PUBLIC_DEMO=1')
    # Initialize the shared compact model before loading the browser framework.
    from .onnx_embeddings import encode
    encode(['query: demo'])
    import uvicorn
    from streamlit.web import bootstrap
    from .mcp_http import create_app
    public_url = os.environ.get('ENVSEARCH_MCP_PUBLIC_URL') or os.environ.get('RENDER_EXTERNAL_URL', '')
    if public_url:
        os.environ['ENVSEARCH_MCP_PUBLIC_URL'] = public_url
    app = create_app(public_url)
    port = int(os.environ.get('PORT', '10000'))
    if not 1 <= port <= 65535:
        raise ValueError('PORT is invalid')
    config = (REPO_ROOT / 'deploy/nginx.conf').read_text().replace('__PORT__', str(port))
    config_path = Path('/tmp/envsearch-nginx.conf')
    config_path.write_text(config)
    subprocess.run(['nginx', '-t', '-c', str(config_path)], check=True)
    proxy = subprocess.Popen(['nginx', '-c', str(config_path), '-g', 'daemon off;'])
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=8602, access_log=False,
                                          limit_concurrency=8))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    options = {'server.address': '127.0.0.1', 'server.port': 8601, 'server.headless': True,
               'server.maxUploadSize': 1, 'browser.gatherUsageStats': False,
               'theme.base': 'light', 'theme.primaryColor': '#2459A6',
               'theme.backgroundColor': '#FFFFFF', 'theme.secondaryBackgroundColor': '#F0F4F9',
               'theme.textColor': '#203047'}
    if public_url:
        options.update({'browser.serverAddress': urlsplit(public_url).hostname, 'browser.serverPort': 443})
    try:
        bootstrap.load_config_options(flag_options=options)
        bootstrap.run(str(Path(__file__).with_name('web_app.py')), False, [], options)
    finally:
        server.should_exit = True
        proxy.terminate()
        proxy.wait(timeout=10)
        thread.join(timeout=10)


if __name__ == '__main__':
    main()
