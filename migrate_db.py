import sqlite3

def migrate_database():
    db_path = "backend/data/app.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA foreign_keys = OFF;")
    
    # Create new table with updated schema
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id VARCHAR(36) NOT NULL UNIQUE,
        task_type VARCHAR(50),
        model_id VARCHAR(50),
        top_label VARCHAR(100),
        confidence FLOAT,
        severity VARCHAR(20),
        all_scores TEXT,
        localization_type VARCHAR(20) DEFAULT 'heatmap',
        bounding_boxes TEXT,
        image_width INTEGER,
        image_height INTEGER,
        overlay_path VARCHAR(500),
        analysis_time_ms INTEGER,
        analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(scan_id) REFERENCES scans(id)
    )
    """)
    
    # Copy data from old table to new table
    # This preserves all historical records. Old records will simply have NULL for the new fields,
    # and new records can have NULL for top_label, confidence, severity, etc.
    cursor.execute("""
    INSERT INTO results_new (
        id, scan_id, top_label, confidence, severity, all_scores, 
        localization_type, bounding_boxes, analysis_time_ms, analyzed_at
    )
    SELECT 
        id, scan_id, top_label, confidence, severity, all_scores, 
        localization_type, bounding_boxes, analysis_time_ms, analyzed_at
    FROM results;
    """)
    
    # Drop old table and rename new table
    cursor.execute("DROP TABLE results;")
    cursor.execute("ALTER TABLE results_new RENAME TO results;")
    
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    conn.commit()
    conn.close()
    print("Database migration completed successfully.")

if __name__ == "__main__":
    migrate_database()
