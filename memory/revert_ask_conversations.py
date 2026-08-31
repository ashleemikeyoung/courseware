"""
One-time fix: undo the ask_conversations composite-key migration that was
added for multi-user login, now that the login feature itself has been
rolled back. Restores the table to its original shape:

    CREATE TABLE ask_conversations (
        project TEXT PRIMARY KEY,
        messages TEXT NOT NULL DEFAULT '[]',
        turn_seq INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
        cleared_at TEXT
    )

Safe to run once. If the table is already in the old shape (no user_id
column), this does nothing.

Run with: python3 memory/revert_ask_conversations.py
(same conda env / .env as init_schema.py)
"""

import libsql_client

from memory_config import LIBSQL_URL, LIBSQL_AUTH_TOKEN


def main():
    client = libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
    try:
        cols = {row[1] for row in client.execute("PRAGMA table_info(ask_conversations)").rows}
        if "user_id" not in cols:
            print("ask_conversations already in the pre-login shape. Nothing to do.")
            return

        print("Reverting ask_conversations to a plain project-keyed table...")
        client.execute(
            "CREATE TABLE ask_conversations_v1 ("
            "project TEXT PRIMARY KEY, "
            "messages TEXT NOT NULL DEFAULT '[]', "
            "turn_seq INTEGER NOT NULL DEFAULT 0, "
            "updated_at TEXT NOT NULL DEFAULT (datetime('now')), "
            "cleared_at TEXT)"
        )
        # One row per project. If more than one user had saved a conversation
        # for the same project, this keeps the most recently written row --
        # fine here since only the bootstrap admin ever used this table.
        client.execute(
            "INSERT INTO ask_conversations_v1 (project, messages, turn_seq, updated_at, cleared_at) "
            "SELECT project, messages, turn_seq, updated_at, cleared_at FROM ask_conversations AS ac "
            "WHERE ac.rowid = (SELECT MAX(rowid) FROM ask_conversations AS ac2 "
            "WHERE ac2.project = ac.project)"
        )
        client.execute("DROP TABLE ask_conversations")
        client.execute("ALTER TABLE ask_conversations_v1 RENAME TO ask_conversations")
        print("Done. ask_conversations is back to a single PRIMARY KEY(project).")
    finally:
        client.close()


if __name__ == "__main__":
    main()
