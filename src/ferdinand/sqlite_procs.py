
import sqlite3
import os

DTYPE_MAP = {
    "int64": "INTEGER",
    "float64": "REAL",
    "object": "TEXT"
}

def connect_db(sqlite_file):
    """
    Establishes a connection to the SQLite database and returns the connection object.
    Returns:
        sqlite3.Connection: A connection object to the SQLite database.
    Raises:
        sqlite3.OperationalError: If the database connection fails due to an invalid
        path or unavailable database file.
    """
    if os.path.exists(sqlite_file):
        try:
            connection = sqlite3.connect(sqlite_file)
            return connection
        except sqlite3.OperationalError as e:
            raise sqlite3.OperationalError(f"Failed to connect to database at {sqlite_file}: {e}")
    else:
        raise FileNotFoundError(f"ERROR: expected SQLite database {sqlite_file} not found!")
    