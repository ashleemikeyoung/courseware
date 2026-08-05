# ~/Development/RAG/memory -- session/episodic memory for the RAG stack

Separate from `chroma_db` on purpose. Chroma answers "what's relevant"
(chunks + embeddings, kept lean and fast). This answers "what happened"
(questions asked, answers given, which chunks got used, how they scored).
See the design discussion for the full reasoning.

Runs as `libsql-server` (the project formerly called `sqld`), self-hosted,
no cloud account, no external service.

**Currently single-machine.** Everything runs on this Mac, bound to
`127.0.0.1` only -- not exposed to the LAN. The client/server split (this
file talks HTTP to a container instead of opening a `.db` file directly)
is still worth having even on one machine, since it's what makes the
`incognito` guarantee and the multi-writer safety real rather than
theoretical. But there's no second machine as a client right now, and the
setup below reflects that. A "connect a second machine later" section can
come back if and when that's actually true -- no sense documenting a
topology that isn't in use.

I've written every file in this folder already: `schema.sql`,
`docker-compose.yml`, `mint_token.py`, `init_schema.py`, `memory_config.py`,
`memory_client.py`. What's left needs your terminal -- I don't have shell
access on your Mac, only the file read/write you're seeing here, so the
steps below are things to run yourself.

---

## First-time setup

### 1. Generate the auth keypair (local only, nothing external)

```bash
cd ~/Development/RAG/memory/keys
openssl genpkey -algorithm ed25519 -out jwt_private_key.pem
openssl pkey -in jwt_private_key.pem -pubout -out jwt_public_key.pem
chmod 600 jwt_private_key.pem
```

### 2. Start the server

```bash
cd ~/Development/RAG/memory
docker compose up -d
docker compose logs -f   # confirm it says "listening" then Ctrl-C
```

If you don't have Docker Desktop yet, that's the one external dependency
here: `brew install --cask docker`, launch it once so the daemon is
running, then repeat step 2.

### 3. Mint an auth token and fill in .env

```bash
cd ~/Development/RAG/memory
cp .env.example .env
conda activate rag   # skip if your prompt already shows (rag)
pip install pyjwt[crypto] libsql-client python-dotenv
python3 mint_token.py
```

Note: `--break-system-packages` was wrongly included in an earlier version
of this doc -- that flag is for Debian/Ubuntu's externally-managed system
Python (PEP 668) and does nothing useful inside a conda environment.
Everything in this project installs into the `rag` conda environment.

Paste the printed token into `.env` as `LIBSQL_AUTH_TOKEN`. Leave
`LIBSQL_URL=http://localhost:8080` as-is.

### 4. Apply the schema

```bash
python3 init_schema.py
```

Should print `Applied 6 statement(s) from schema.sql`.

### 5. Sanity check

```bash
python3 -c "from memory_client import start_session; s = start_session(project='test', machine='mac', mode='qa'); tid = s.log_turn(question='ping', answer='pong'); s.close(); print('logged turn', tid)"
python3 -c "from memory_client import recent_turns; print(recent_turns(5))"
```

You should see the ping/pong turn come back.

---

## If a second machine joins later

The pieces are already shaped for it and nothing above needs to be
redone, but two things need to change deliberately when it happens, not
casually:

1. `docker-compose.yml` -- change the port line back to `"8080:8080"` (or
   bind to a specific LAN interface) so it's reachable from off-box.
2. The second machine gets its own copy of `memory_config.py`,
   `memory_client.py`, and a `.env` pointing `LIBSQL_URL` at this Mac's LAN
   IP, using the *same* `LIBSQL_AUTH_TOKEN` already minted here.

**Only one machine should ever run `docker compose up -d` for this
stack.** Every other machine is a client only, never its own container --
two primaries writing to what looks like the same logical database is
exactly the corruption case worth avoiding, independent of whether the
underlying files happen to be shared over NFS or not.

---

## What's NOT done yet

Session logging IS wired in now: `orchestrator.py` logs every Q&A turn,
`writer.py` logs per-section and per-document turns with quality scores,
both with an `/incognito` (orchestrator.py) escape hatch. See the
conversation log from 2026-08-03 for the fuller feature set built on top
of this since first setup: citations, retrieval-gap tracking, false-positive
flagging, repeat-question caching, and PII redaction (`../pii.py`) with a
persistent `/redact` toggle.

Still open: the ingest-time synopsis + PII-scan hook (populating the
`documents` and `pii_scans` tables automatically as `rag.py` indexes new
files) and folding `search_synopses()` into `orchestrator.py`'s query
flow. Both designed, neither wired yet.

## Backups

`~/Development/RAG/memory/data/iku.db` is the actual libSQL file
(SQLite-format under the hood, `iku.db` is sqld's own default name).
`backup.sh` is a manual single-file copier (`backup.sh <source_file>
<backup_dir>`), not a tree sweep, so it won't pick this up automatically
just by living under RAG -- run it explicitly the same way you would for
`rag.py`:

```bash
./backup.sh ~/Development/RAG/memory/data/iku.db ~/Development/RAG/backups
```
