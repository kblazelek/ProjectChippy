import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "project_chippy.db"

def get_connection() -> sqlite3.Connection:
    """
    Creates, configures, and returns an SQLite database connection.
    Ensures that the directory structure exists before attempting to connect.
    """
    # 1. Ensure the 'data' folder exists. SQLite will auto-create the .db file,
    # but it will throw an error if the parent directory does not exist.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Open the connection
    conn = sqlite3.connect(DB_PATH)
    
    # 3. Enable Foreign Key Constraints
    # SQLite has foreign keys disabled by default for legacy compatibility.
    # We MUST turn them on so our ON DELETE CASCADE rules work properly
    # between the captured_images and animal_detections tables.
    conn.execute("PRAGMA foreign_keys = ON")
    
    # 4. Use Row Factory
    # This allows us to access columns by name (e.g., row['animal_class'])
    # instead of by index (e.g., row[2]), making queries much easier to read.
    conn.row_factory = sqlite3.Row
    
    return conn