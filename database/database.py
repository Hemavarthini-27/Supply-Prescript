import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "supply_prescript.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    # Existing executed decisions table
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
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_training_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_version TEXT,
        accuracy REAL,
        trigger_reason TEXT,
        training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Closed-loop feedback table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS decision_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id INTEGER,
            shipment_id TEXT,
            predicted_delay INTEGER,
            actual_delay INTEGER,
            predicted_cost REAL,
            actual_cost REAL,
            delay_difference INTEGER,
            cost_difference REAL,
            outcome TEXT,
            feedback_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (decision_id)
                REFERENCES executed_decisions(id)
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

    decision_id = cursor.lastrowid

    conn.close()

    return decision_id

def save_training_history(
    model_version,
    accuracy,
    trigger_reason
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO model_training_history
        (
            model_version,
            accuracy,
            trigger_reason
        )
        VALUES (?, ?, ?)
    """, (
        model_version,
        accuracy,
        trigger_reason
    ))

    conn.commit()
    conn.close()

def save_feedback(
    decision_id,
    shipment_id,
    predicted_delay,
    actual_delay,
    predicted_cost,
    actual_cost
):
    delay_difference = actual_delay - predicted_delay
    cost_difference = actual_cost - predicted_cost

    # Determine whether the decision was successful
    if actual_delay <= predicted_delay and actual_cost <= predicted_cost:
        outcome = "Successful"

    elif actual_delay <= predicted_delay or actual_cost <= predicted_cost:
        outcome = "Partially Successful"

    else:
        outcome = "Unsuccessful"

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO decision_feedback
        (
            decision_id,
            shipment_id,
            predicted_delay,
            actual_delay,
            predicted_cost,
            actual_cost,
            delay_difference,
            cost_difference,
            outcome
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        decision_id,
        shipment_id,
        predicted_delay,
        actual_delay,
        predicted_cost,
        actual_cost,
        delay_difference,
        cost_difference,
        outcome
    ))

    conn.commit()
    conn.close()

    return {
        "outcome": outcome,
        "delay_difference": delay_difference,
        "cost_difference": cost_difference
    }