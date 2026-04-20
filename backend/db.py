import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "app_data.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS risk_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            disease TEXT NOT NULL,
            risk_score REAL NOT NULL,
            predicted_class INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS diet_plan (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            disease TEXT NOT NULL,
            calories_target INTEGER NOT NULL,
            protein_target_g INTEGER NOT NULL,
            carb_target_g INTEGER NOT NULL,
            fat_target_g INTEGER NOT NULL,
            sodium_limit_mg INTEGER NOT NULL,
            potassium_limit_mg INTEGER NOT NULL,
            recommendation_note TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS meal_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            planned_calories INTEGER NOT NULL,
            consumed_calories INTEGER NOT NULL,
            deviation_flag INTEGER NOT NULL,
            deviation_percent REAL NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alert (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()


def insert_risk_profile(user_id: str, disease: str, risk_score: float, predicted_class: int) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO risk_profile (user_id, disease, risk_score, predicted_class) VALUES (?, ?, ?, ?)",
        (user_id, disease, risk_score, predicted_class),
    )
    conn.commit()
    conn.close()


def insert_diet_plan(user_id: str, disease: str, plan: Dict[str, Any]) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO diet_plan (
            user_id, disease, calories_target, protein_target_g, carb_target_g,
            fat_target_g, sodium_limit_mg, potassium_limit_mg, recommendation_note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            disease,
            plan["calories_target"],
            plan["protein_target_g"],
            plan["carb_target_g"],
            plan["fat_target_g"],
            plan["sodium_limit_mg"],
            plan["potassium_limit_mg"],
            plan["recommendation_note"],
        ),
    )
    conn.commit()
    conn.close()


def insert_meal_log(user_id: str, planned_calories: int, consumed_calories: int, deviation_flag: bool, deviation_percent: float, notes: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO meal_log (user_id, planned_calories, consumed_calories, deviation_flag, deviation_percent, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, planned_calories, consumed_calories, int(deviation_flag), deviation_percent, notes),
    )
    conn.commit()
    meal_id = cur.lastrowid
    conn.close()
    return int(meal_id)


def insert_alert(user_id: str, severity: str, message: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO alert (user_id, severity, message) VALUES (?, ?, ?)",
        (user_id, severity, message),
    )
    conn.commit()
    alert_id = cur.lastrowid
    conn.close()
    return int(alert_id)


def get_open_alert_count(user_id: Optional[str] = None) -> int:
    conn = get_conn()
    if user_id:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM alert WHERE status = 'open' AND user_id = ?",
            (user_id,),
        ).fetchone()
    else:
        row = conn.execute("SELECT COUNT(*) AS count FROM alert WHERE status = 'open'").fetchone()
    conn.close()
    return int(row["count"]) if row else 0
