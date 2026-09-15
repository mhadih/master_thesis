"""Shared database configuration helper.

Reads connection parameters from environment variables (see `.env.example`).
A minimal `.env` file loader is included so scripts work without the
third-party `python-dotenv` package.

Usage from a standalone script, e.g. ``src/data_collection/retrieve_data.py``
or ``S04/edit_distance.py``::

    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.db_config import get_code_recorder_config

    conn = psycopg2.connect(**get_code_recorder_config())
"""

import os
from pathlib import Path


def _load_dotenv(filename=".env"):
    """Minimal .env loader: KEY=VALUE lines -> os.environ (no override).

    Searches the current working directory, then the directory containing
    this file and its parents (up to the filesystem root).
    """
    if filename in os.environ.get("_DB_CONFIG_DOTENV_LOADED", ""):
        return
    search_dirs = [Path.cwd()] + list(Path(__file__).resolve().parents)
    seen = set()
    for directory in search_dirs:
        if directory in seen:
            continue
        seen.add(directory)
        env_file = directory / filename
        if not env_file.is_file():
            continue
        try:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                os.environ.setdefault(key, value)
        except OSError:
            continue
        os.environ["_DB_CONFIG_DOTENV_LOADED"] = str(env_file)
        return


_load_dotenv()


def _get(key, default):
    value = os.getenv(key, default)
    return value


def _port(key, default):
    try:
        return int(_get(key, default))
    except (TypeError, ValueError):
        return default


def get_code_recorder_config():
    """Connection kwargs for the S03 ``code_recorder`` database."""
    return {
        "dbname": _get("CODE_RECORDER_DBNAME", "code_recorder"),
        "user": _get("CODE_RECORDER_USER", "hadi"),
        "password": _get("CODE_RECORDER_PASSWORD", ""),
        "host": _get("CODE_RECORDER_HOST", "localhost"),
        "port": _port("CODE_RECORDER_PORT", 5432),
    }


def get_s04_config():
    """Connection kwargs for the local S04 database."""
    return {
        "dbname": _get("S04_DBNAME", "code_recorder_s04"),
        "user": _get("S04_USER", "postgres"),
        "password": _get("S04_PASSWORD", ""),
        "host": _get("S04_HOST", "localhost"),
        "port": _port("S04_PORT", 5432),
    }


def get_remote_config():
    """Connection kwargs for the remote LMS database (S04/dump_db.py)."""
    return {
        "dbname": _get("REMOTE_DBNAME", ""),
        "user": _get("REMOTE_USER", ""),
        "password": _get("REMOTE_PASSWORD", ""),
        "host": _get("REMOTE_HOST", ""),
        "port": _port("REMOTE_PORT", 5432),
    }
