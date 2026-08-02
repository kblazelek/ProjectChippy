import datetime
from typing import List, Dict, Any
from .connection import get_connection

def save_detection(image_path: str, detections: List[Dict[str, Any]]) -> int:
    """
    Saves a captured image and all animals detected within it.
    
    Args:
        image_path: The file path to the saved image.
        detections: A list of dictionaries, e.g., 
                    [{'class': 'Fox', 'confidence': 0.85, 'bbox': [10, 20, 100, 200]}]
                    
    Returns:
        The ID of the newly inserted image.
    """
    # The 'with' statement acts as a transaction manager. 
    # If anything fails, it automatically rolls back the changes.
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Save the parent image record
        cursor.execute(
            "INSERT INTO captured_images (image_path) VALUES (?)",
            (image_path,)
        )
        new_image_id = cursor.lastrowid
        
        # 2. Save all individual animal detections linked to this image
        for det in detections:
            # Assuming bbox is a list or tuple: [x, y, width, height]
            bbox = det.get('bbox', [0, 0, 0, 0]) 
            
            cursor.execute('''
                INSERT INTO animal_detections 
                (image_id, animal_class, confidence, bbox_x, bbox_y, bbox_width, bbox_height)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                new_image_id,
                det.get('class', 'Unknown'),
                det.get('confidence', 0.0),
                bbox[0], bbox[1], bbox[2], bbox[3]
            ))
            
        # The 'with' block automatically calls conn.commit() upon successful exit
        return new_image_id

def toggle_favorite(image_id: int, is_favorite: bool) -> bool:
    """
    Updates the favorite status of a specific image.
    
    Args:
        image_id: The ID of the image in the captured_images table.
        is_favorite: True to favorite, False to unfavorite.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        # Convert the python boolean to an integer (1 or 0) for SQLite
        fav_int = 1 if is_favorite else 0
        
        cursor.execute('''
            UPDATE captured_images 
            SET is_favorite = ? 
            WHERE id = ?
        ''', (fav_int, image_id))
        
        # Return True if a row was actually updated
        return cursor.rowcount > 0

def get_detections_for_day(target_date_str: str) -> List[Dict[str, Any]]:
    """
    Fetches all images and their detections for a specific day.
    Uses the optimized > and < index strategy.
    
    Args:
        target_date_str: Date string in 'YYYY-MM-DD' format.
    """
    date_obj = datetime.datetime.strptime(target_date_str, '%Y-%m-%d')
    start_time = date_obj.strftime('%Y-%m-%d 00:00:00')
    
    next_day_obj = date_obj + datetime.timedelta(days=1)
    end_time = next_day_obj.strftime('%Y-%m-%d 00:00:00')
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                i.id AS image_id,
                i.timestamp,
                i.image_path,
                i.is_favorite,
                d.id AS detection_id,
                d.animal_class,
                d.confidence,
                d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height
            FROM captured_images i
            JOIN animal_detections d ON i.id = d.image_id
            WHERE i.timestamp >= ? AND i.timestamp < ?
            ORDER BY i.timestamp DESC
        ''', (start_time, end_time))
        
        # Because we enabled sqlite3.Row in connection.py, 
        # we can easily convert the rows to standard Python dictionaries.
        return [dict(row) for row in cursor.fetchall()]

def get_all_favorites() -> List[Dict[str, Any]]:
    """
    Fetches all images that have been marked as favorites by the user.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Group by image_id to prevent duplicates if a favorited 
        # picture has multiple animals in it. We use GROUP_CONCAT 
        # to get a comma-separated list of the animals in the photo!
        cursor.execute('''
            SELECT 
                i.id AS image_id,
                i.timestamp,
                i.image_path,
                GROUP_CONCAT(d.animal_class) as animals_present
            FROM captured_images i
            JOIN animal_detections d ON i.id = d.image_id
            WHERE i.is_favorite = 1
            GROUP BY i.id
            ORDER BY i.timestamp DESC
        ''')
        
        return [dict(row) for row in cursor.fetchall()]