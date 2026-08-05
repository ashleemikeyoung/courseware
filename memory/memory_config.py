"""
memory_config.py -- connection settings for memory-db, read from
~/Development/memory/.env so the auth token never has to be typed into
code or committed anywhere.

On the Mac (where the server runs): LIBSQL_URL=http://localhost:8080
On alice: LIBSQL_URL=http://<mac's LAN IP>:8080

Both machines use the same LIBSQL_AUTH_TOKEN, minted once via mint_token.py
on the Mac and copied over -- see README.md.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

LIBSQL_URL = os.getenv("LIBSQL_URL", "http://localhost:8080")
LIBSQL_AUTH_TOKEN = os.getenv("LIBSQL_AUTH_TOKEN", "")

if not LIBSQL_AUTH_TOKEN:
    print(
        "  [Warning] LIBSQL_AUTH_TOKEN is not set in "
        f"{Path(__file__).parent / '.env'}. Requests to memory-db will fail "
        "auth once SQLD_AUTH_JWT_KEY_FILE is configured on the server."
    )
