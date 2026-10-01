"""Practical setup instructions for this repository's local MCP server."""
import json
import os
import shlex
import sys

import streamlit as st


def mcp_panel(settings):
    if os.environ.get("ENVSEARCH_PUBLIC_DEMO") == "1":
        return public_mcp_panel()
    home = str(settings.home.resolve())
    command = [sys.executable, "-m", "envsearch.mcp_server"]
    config = {"mcpServers": {"envsearch": {"command": command[0], "args": command[1:],
                                         "env": {"ENVSEARCH_HOME": home}}}}
    config_json = json.dumps(config, indent=2)
    st.subheader("Connect another assistant to your document library")
    st.markdown("""<style>
    .st-key-mcp_overview {border-top: 4px solid #2459A6; background: rgba(36,89,166,0.04);}
    .st-key-mcp_prepare, .st-key-mcp_connect {border-top: 4px solid #52677F;}
    .st-key-mcp_test, .st-key-mcp_use {border-top: 4px solid #24715A;}
    .st-key-mcp_remote {border-top: 4px solid #B87920; background: rgba(184,121,32,0.05);}
    </style>""", unsafe_allow_html=True)
    with st.container(border=True, key="mcp_overview"):
        st.subheader("What you can use today", help="MCP (Model Context Protocol) lets an AI application discover and call tools supplied by another program. Envsearch supplies tools that read your indexed documents.")
        st.write("**This is already an MCP server.** It uses FastMCP and runs locally through **stdio**: your client starts a Python process and exchanges messages through its input/output.")
        st.write("**Connection:** Claude Desktop → Envsearch MCP process → your saved document index.")
        st.info("Use the local setup below today. There is no MCP HTTP URL in this version. localhost:8501 is the Streamlit interface, so entering it as a remote MCP URL will not work.")
        st.caption("The MCP client starts and stops its own server process. You do not need to keep this web app or a separate terminal server running.")

    with st.container(border=True, key="mcp_prepare"):
        st.subheader("1. Prepare your library", help="The MCP server reads the same saved index as this app. An index is the searchable copy of your documents.")
        st.write("Open the Library tab and confirm your documents are listed. If it is empty, add a PDF in Add PDFs first.")
        st.write("**Data folder used by the configuration below:**")
        st.code(home, language="text")
        st.caption("Commands on this page are for this machine's macOS/Linux shell and Python environment. If you move the project, regenerate the paths.")
        with st.expander("Optional: check that the server starts in Terminal"):
            st.code(shlex.join(["env", f"ENVSEARCH_HOME={home}", *command]), language="bash")
            st.write("Run this in Terminal. It may appear to do nothing: stdio waits for an MCP client, and has no browser page or ‘listening on port’ message. This only checks startup, not tool calls. Press Control+C before continuing.")
        with st.expander("Installing a local copy on another machine"):
            st.write("For a fresh local installation, copy the repository source without its .venv or data folders. Keep the original PDFs separately. In the new repository folder, with Python 3.10+ available, run:")
            st.code('python3 -m venv .venv\n.venv/bin/python -m pip install -e ".[web,vectors]"\n.venv/bin/envsearch serve', language="bash")
            st.write("Upload the original PDFs through Add PDFs on the new machine, then use that machine's generated configuration from Tools. Vector search may download its embedding model on first use; keyword search works without that model.")
            st.caption("Copying an existing data folder is not a complete migration: uploaded documents store absolute PDF paths, which need updating in the upload records and index on the destination. Otherwise search may work while PDF previews or rebuilding fail. Re-uploading avoids those stale paths. This fresh-install procedure does not transfer gold additions or saved evaluations.")

    with st.container(border=True, key="mcp_connect"):
        st.subheader("2. Connect Claude Desktop", help="Claude Desktop is one example of an MCP client. Other clients can use the same executable, arguments, environment and stdio transport, but may use a different configuration format.")
        st.markdown("1. Open **Claude Desktop → Settings → Developer → Edit Config**.\n"
                    "2. Add the `envsearch` entry below inside `mcpServers`. If other servers are already configured, keep their entries.\n"
                    "3. Save the file, fully quit Claude Desktop, then open it again.\n"
                    "4. Find `envsearch` in the client's available connectors/tools and enable its tools as needed.")
        st.caption("On macOS the file is ~/Library/Application Support/Claude/claude_desktop_config.json. These steps apply to the desktop application, not the Claude website.")
        st.code(config_json, language="json")
        st.download_button("Download MCP configuration", config_json, file_name="envsearch-mcp.json", mime="application/json",
                           help="A configuration snippet with local paths and no API keys. Merge its envsearch entry into your client's configuration; downloading it does not connect the client automatically.")
        st.markdown("Client setup reference: [official local MCP connection guide](https://modelcontextprotocol.io/docs/develop/connect-local-servers).")
        with st.expander("Using another client that supports local MCP"):
            st.write("Choose **stdio** (sometimes called ‘local’ or ‘command’), then enter these values:")
            st.write("**Command / executable**")
            st.code(command[0], language="text")
            st.write("**Arguments** — two separate entries")
            st.code(json.dumps(command[1:]), language="json")
            st.write("**Environment variable**")
            st.code(f"ENVSEARCH_HOME={home}", language="text")
            st.caption("If the client only accepts a server URL, it cannot connect directly to this stdio server. See remote hosting below.")

    with st.container(border=True, key="mcp_test"):
        st.subheader("3. Verify the connection", help="A server appearing in a client is only the first check. Call a tool to confirm it can read your saved library.")
        st.write("In a new client conversation, send this request:")
        st.code("Use envsearch's list_documents tool and show me the available documents.", language="text")
        st.write("**Success looks like:** a visible call to `list_documents`, followed by your document titles and IDs. A plain conversational reply without a tool call does not confirm the connection.")
        st.write("Then test retrieval:")
        st.code("Use envsearch's search_regulations with query='มาตรฐานน้ำทิ้ง', mode='bm25', k=3. Show the source titles and page numbers.", language="text")
        st.caption("Use a phrase from one of your own documents if you do not have wastewater documents. Keyword mode (bm25) makes this first check independent of embedding-model downloads. No matches can mean the phrase is absent; it is not necessarily a connection failure.")
        with st.expander("Test tools directly with MCP Inspector (without a chat model)"):
            st.write("Install Node.js 22.19 or newer if needed. Run this command in Terminal; npx downloads the Inspector package if it is missing:")
            st.code(shlex.join(["env", f"ENVSEARCH_HOME={home}", "npx", "@modelcontextprotocol/inspector", "--", *command]), language="bash")
            st.write("Open the local Inspector URL printed in Terminal. Connect to the configured stdio server, list its tools, and call `list_documents` with `{}`. You should see the four tools listed below and your indexed documents.")
            st.write("Next call `search_regulations` with:")
            st.code(json.dumps({"query": "มาตรฐานน้ำทิ้ง", "mode": "bm25", "k": 3}, ensure_ascii=False, indent=2), language="json")
            st.caption("The Inspector's local web URL is a testing interface, not a remotely hosted Envsearch MCP endpoint. Stop it with Control+C when finished.")
            st.markdown("Reference: [official MCP Inspector documentation](https://github.com/modelcontextprotocol/inspector).")
        with st.expander("Troubleshooting"):
            st.markdown("- **Server does not appear:** check the JSON syntax and absolute executable path, then fully restart the client.\n"
                        "- **No module named envsearch:** use this project's virtual-environment Python. Reinstall the project in that environment with `python -m pip install -e .` from the repo folder.\n"
                        "- **Missing index / no documents:** check `ENVSEARCH_HOME` points to the data folder containing `index`, then add PDFs through the app.\n"
                        "- **New PDF is not visible:** restart the MCP client/server. It caches the index until the process restarts.\n"
                        "- **Vector or reranker model cannot load:** first verify with `mode='bm25'` and `rerank=false`; install the vectors dependencies and allow the required model download for semantic search.\n"
                        "- **Cited-answer tool asks for a key:** see the optional Anthropic setup below. The key entered in this web app is not passed to the MCP process.")
            st.caption("Claude Desktop logs on macOS: ~/Library/Logs/Claude/mcp.log and mcp-server-envsearch.log. Check the error text without sharing API keys.")

    with st.container(border=True, key="mcp_use"):
        st.subheader("4. Choose how the assistant uses the tools", help="A tool is a named operation the client can call. The client chooses when to call it; connecting a server does not force every message to use it.")
        st.table([
            {"Tool": "list_documents", "Use it to": "See document titles and IDs", "Server API key": "None"},
            {"Tool": "search_regulations", "Use it to": "Find passages with optional filters, hybrid search and reranking", "Server API key": "None"},
            {"Tool": "get_passage", "Use it to": "Read a returned chunk_id with nearby text", "Server API key": "None"},
            {"Tool": "ask_with_citations", "Use it to": "Run this repo's own Claude answer pipeline", "Server API key": "Anthropic (paid API calls)"},
        ])
        st.write("**Start with search:** the client's own model reads the returned passages and writes the answer. Envsearch does not need an answer API key for this, though your client may have its own subscription or model charges.")
        st.code("Use envsearch to search Thai wastewater rules. Keep language='any', jurisdiction='TH', mode='auto'. Read relevant passages and cite document titles and pages. If the sources don't support an answer, say so.", language="text")
        st.caption("auto uses hybrid retrieval when saved vectors exist, otherwise keyword search. Filters on this app's sidebar are not automatically applied to MCP calls; pass them as tool arguments. Search returns a limited number of passages, not every relevant passage.")
        st.write("The raw search tool does not run the web app's scope router. `ask_with_citations` uses the repo's scope-checked answer pipeline. MCP calls do not currently populate this app's Monitoring tab.")
        with st.expander("Optional: let the server generate a cited answer"):
            st.write("Only `ask_with_citations` needs an Anthropic API key in the MCP process. The OpenAI choice and keys entered in this web app do not configure this tool. The MCP entrypoint also does not load this app's .env file.")
            st.write("In the client configuration, replace the `envsearch.env` object with the following, and replace the placeholder locally with your Anthropic key:")
            st.code(json.dumps({"ENVSEARCH_HOME": home, "ANTHROPIC_API_KEY": "REPLACE_WITH_YOUR_ANTHROPIC_KEY"}, indent=2), language="json")
            st.caption("This puts a secret in your local client configuration. Keep that file private and out of version control. The downloadable configuration above contains no keys. A client's secure environment/secret setting can be used instead if supported.")
            st.write("Restart the client, then ask it to call `ask_with_citations` with your question. This makes paid Anthropic calls, including scope checking and, by default, bilingual query expansion.")

    with st.container(border=True, key="mcp_remote"):
        st.subheader("Deploy a public browser and MCP demo", help="The Docker deployment runs the browser app and read-only HTTP MCP server together on a separate host.")
        st.write("This repo now includes a Render Free deployment. It serves the browser at the service URL and Streamable HTTP MCP at the same URL followed by /mcp.")
        st.markdown("[Deploy on Render](https://render.com/deploy?repo=https://github.com/chatchawan90/e2e-rag-demo) · [Step-by-step deployment guide](https://github.com/chatchawan90/e2e-rag-demo/blob/main/deploy/README.md)")
        st.write("Choose the Free plan and do not add a payment method for a strict zero-bill setup. The service sleeps after inactivity and has monthly limits. Browser chat uses visitors' own API keys, which may incur provider charges.")
        st.caption("The hosted snapshot contains only the public reference library. Uploads, editing, evaluation runs, activity logs, reranking and the paid server-answer tool are disabled. Your full local app remains available.")


def public_mcp_panel():
    base = os.environ.get("ENVSEARCH_MCP_PUBLIC_URL", "").rstrip("/")
    st.subheader("Connect to this demo through MCP")
    with st.container(border=True):
        st.write("Use these same public documents from an MCP-compatible assistant. The server offers search, document listing and passage lookup; it does not call a paid answer API.")
        if base:
            st.write("**Server URL**")
            st.code(base + "/mcp", language="text")
        else:
            st.info("Use this site's public HTTPS address followed by /mcp. For example: https://your-service.onrender.com/mcp")
        st.write("**Transport:** Streamable HTTP. **Authentication:** none; this library is public.")
        st.markdown("1. Open your client's MCP server settings and add a remote server.\n2. Paste the server URL above and choose Streamable HTTP if asked.\n3. Connect, then ask it to call `list_documents`.\n4. Ask it to search with `search_regulations` and cite the returned sources.")
        st.code("Use envsearch-demo to search Thai wastewater rules. Keep language='any' and mode='hybrid'. Cite document titles and pages.", language="text")
        st.caption("Client support for anonymous remote MCP varies. Use a client that accepts a custom Streamable HTTP URL. Your client's own model may have separate charges.")
    with st.container(border=True):
        st.subheader("Free demo limits")
        st.write("The free host sleeps after inactivity. Open this browser page first to wake it, then retry the MCP connection if it timed out. Monthly hosting limits also apply.")
        st.write("This demo is read-only. Uploads, paid server answers, and reranking are disabled. Hybrid search uses a compact int8 version of the same multilingual E5 embedding model; rankings may differ slightly from the local full-precision version.")
        st.markdown("[Run the full app locally](https://github.com/chatchawan90/e2e-rag-demo) to use your own documents, evaluation and editing features.")
