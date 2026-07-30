import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "supply_prescript.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS executed_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shipment_id TEXT,
            predicted_delay INTEGER,
            recommended_action TEXT,
            predicted_cost REAL,
            execution_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_decision(
    shipment_id,
    predicted_delay,
    recommended_action,
    predicted_cost
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO executed_decisions
        (
            shipment_id,
            predicted_delay,
            recommended_action,
            predicted_cost
        )
        VALUES (?, ?, ?, ?)
    """, (
        shipment_id,
        predicted_delay,
        recommended_action,
        predicted_cost
    ))

    conn.commit()
    conn.close()