from .connection import get_connection

def initialize_database():
    """
    Creates tables, applies schema migrations using PRAGMA user_version,
    and builds indexes on startup.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Check current version
        cursor.execute("PRAGMA user_version")
        version = cursor.fetchone()[0]
        
        # Version 0 -> Version 1: Initial Tables & Indexes
        if version < 1:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS captured_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT NOT NULL,
                    is_favorite BOOLEAN DEFAULT 0
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS animal_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL,
                    animal_class TEXT NOT NULL,
                    confidence REAL,
                    bbox_x INTEGER, bbox_y INTEGER,
                    bbox_width INTEGER, bbox_height INTEGER,
                    FOREIGN KEY (image_id) REFERENCES captured_images(id) ON DELETE CASCADE
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON captured_images(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_animal_class ON animal_detections(animal_class)')
            
            cursor.execute("PRAGMA user_version = 1")
            print("Database initialized to Version 1.")

        # Version 1 -> Version 2: Store conversation history for LLM prompts
        if version < 2:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_key TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversation_messages_key_created_at
                ON conversation_messages(conversation_key, created_at)
            ''')
            cursor.execute("PRAGMA user_version = 2")
            print("Database initialized to Version 2.")

        conn.commit()
