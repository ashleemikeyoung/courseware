# ChatGPT MCP connection notes

## Local server status

- Entry point: `/Users/ash/Development/RAG/mcp_server.py`
- Transport: MCP over stdio
- Launch command:

```bash
/opt/anaconda3/envs/rag/bin/python /Users/ash/Development/RAG/mcp_server.py
```

- Working directory:

```bash
/Users/ash/Development/RAG
```

- Conda environment: `rag`
- Indexed Chroma chunks at last verification: `1761`
- Projects seen at last verification: `GCU`, `Replevin`

The server exposes these tools:

- `search_documents`
- `get_document_list`
- `list_projects`
- `rescan_documents`
- `ingest_content`
- `ask_local`
- `read_document`
- `summarize_documents`

## What was verified

From the `rag` Conda environment, the MCP server imports successfully, starts
over stdio, lists its tools through an MCP client session, and answers a
`search_documents` call against the existing local Chroma index.

The launch check used the same command a local stdio MCP client should use:

```bash
conda run -n rag python /Users/ash/Development/RAG/mcp_server.py
```

## ChatGPT connection path

ChatGPT does not connect directly to a developer-machine stdio MCP server.
OpenAI's supported path for a private/local MCP server is Secure MCP Tunnel.
The tunnel client runs on the Mac, opens an outbound HTTPS connection to OpenAI,
and forwards MCP requests to this local stdio server.

Official OpenAI references:

- ChatGPT MCP app/developer mode help:
  https://help.openai.com/en/articles/12584461-developer-mode-and-full-mcp-connectors-in-chatgpt
- Secure MCP Tunnel guide:
  https://developers.openai.com/api/docs/guides/secure-mcp-tunnels

## Required OpenAI-side prerequisites

You need all of the following before ChatGPT can scan or use this server:

- ChatGPT developer mode access for the target workspace/account.
- OpenAI Platform tunnel permissions for the relevant Platform organization:
  `Tunnels Read + Use` to run/select a tunnel, and `Tunnels Read + Manage` to
  create or edit one.
- A `tunnel_id` from Platform tunnel settings.
- A runtime API key for `tunnel-client`.
- The `tunnel-client` binary installed on this Mac.

At the time of this note, `tunnel-client` was not found on the local `PATH`.

## Tunnel profile for this server

Once `tunnel-client` and tunnel credentials are available, initialize a local
stdio profile like this, replacing the placeholders:

```bash
export CONTROL_PLANE_API_KEY="sk-..."

tunnel-client init \
  --sample sample_mcp_stdio_local \
  --profile rag-local-stdio \
  --tunnel-id tunnel_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \
  --mcp-command "/opt/anaconda3/envs/rag/bin/python /Users/ash/Development/RAG/mcp_server.py"
```

Validate it:

```bash
tunnel-client doctor --profile rag-local-stdio --explain
```

Run it and keep it running while ChatGPT scans tools or calls the MCP server:

```bash
tunnel-client run --profile rag-local-stdio
```

## Connect from ChatGPT

1. Open ChatGPT on the web.
2. Enable developer mode if your plan/workspace supports it.
3. Go to ChatGPT Plugins / Apps and create a developer-mode app.
4. Choose `Tunnel` as the connection type.
5. Select the available tunnel, or paste the `tunnel_id`.
6. Scan tools.
7. Create the draft app.
8. Start a new ChatGPT chat and select the draft app from the tools/apps menu.

If the tunnel is not visible in ChatGPT, check that the tunnel is associated
with the target ChatGPT workspace and that the app creator has `Tunnels Read +
Use`.

## Security note

This server can read and index local documents. It also exposes write-like
local actions such as `ingest_content` and `rescan_documents`. Review tool
permissions carefully before publishing the app to a workspace.
