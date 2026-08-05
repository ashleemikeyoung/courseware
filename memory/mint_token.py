"""
mint_token.py -- generate a long-lived auth token for memory-db, signed
with the Ed25519 key pair generated in keys/. Nothing here calls out to
Turso or any other external service; the keypair and the token are both
entirely local.

Usage (after keys/jwt_private_key.pem exists -- see README.md step 1):

    pip install pyjwt[crypto] --break-system-packages   # one-time
    python3 mint_token.py                                # prints a token

This is a LAN-only, self-hosted server with no public exposure, so the
token is minted with a long expiry (5 years) rather than something you'd
need to rotate constantly. Re-run this script any time you want a fresh
token; it doesn't invalidate old ones (the key pair is what you'd rotate
to do that).
"""

import datetime
from pathlib import Path

import jwt  # pip install pyjwt[crypto]

KEY_DIR = Path(__file__).parent / "keys"
PRIVATE_KEY_PATH = KEY_DIR / "jwt_private_key.pem"


def main():
    if not PRIVATE_KEY_PATH.exists():
        print(f"  No private key at {PRIVATE_KEY_PATH}")
        print("  Run the openssl commands in README.md step 1 first.")
        raise SystemExit(1)

    private_key = PRIVATE_KEY_PATH.read_text()

    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "iat": now,
        "exp": now + datetime.timedelta(days=365 * 5),
    }

    token = jwt.encode(payload, private_key, algorithm="EdDSA")
    print(token)


if __name__ == "__main__":
    main()
