# utils.py

import streamlit as st
import pandas as pd
import sqlite3
import re
from datetime import datetime
import numpy as np
import emoji

# Shared Constant
DB_PATH = "chat_data.db"

# ----------------------------
# Database Initialization
# ----------------------------
def init_db(db_path=DB_PATH):
    """Creates the local SQLite database with a table for chat messages if it does not exist."""
    conn = sqlite3.connect(db_path, check_same_thread=False) # Added check_same_thread=False for Streamlit
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            sender TEXT NOT NULL,
            message TEXT NOT NULL
        );
    """)
    # Check and add message_type column if it doesn't exist
    cursor.execute("PRAGMA table_info(chat_messages)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'message_type' not in columns:
        try:
            cursor.execute("ALTER TABLE chat_messages ADD COLUMN message_type TEXT DEFAULT 'text'")
            # st.info("Attempted to add 'message_type' column to database (if it didn't exist).") # Avoid st calls in utils if possible
            print("Attempted to add 'message_type' column to database (if it didn't exist).")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                 print(f"Warning: Could not add 'message_type' column: {e}") # Use print for non-UI feedback
                 # st.warning(f"Could not add 'message_type' column: {e}", icon="⚠️")
    conn.commit()
    conn.close()

# ----------------------------
# Parsing Function
# ----------------------------
def parse_whatsapp_chat(file_content: str) -> pd.DataFrame:
    """
    Parses WhatsApp chat logs with minute-level timestamps.
    Expected format:
        "04/02/25, 10:57 am - neel: Congratulations"
    Handles:
      • 2-digit and 4-digit years
      • Unusual Unicode spaces (they are replaced by normal spaces)
      * Detects media messages "<Media omitted>"
      * Detects deleted messages based on common pattern
    """
    pattern = re.compile(
        r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s*([apAP][mM])\s-\s([^:]+):\s(.*)$"
    )
    system_pattern = re.compile(
         r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s*([apAP][mM])\s-\s(.*)$"
    )
    chat_data = []
    for line in file_content.splitlines():
        clean_line = line.replace('\u202F', ' ').replace('\u2009', ' ')
        match = pattern.match(clean_line)
        system_match = system_pattern.match(clean_line)
        timestamp = None
        sender = "System"
        message = ""
        message_type = "unknown"
        if match:
            date_str, time_str, am_pm, sender_match, message = match.groups()
            sender = sender_match.strip()
            message = message.strip()
            message_type = "text"
            timestamp_str = f"{date_str} {time_str} {am_pm.upper()}"
            if message == "<Media omitted>":
                 message_type = "media"
            elif "this message was deleted" in message.lower():
                 message_type = "deleted"
                 message = "This message was deleted"
            for fmt in ("%d/%m/%y %I:%M %p", "%d/%m/%Y %I:%M %p"):
                try:
                    timestamp = datetime.strptime(timestamp_str, fmt)
                    break
                except ValueError: continue
        elif system_match:
            date_str, time_str, am_pm, message_content = system_match.groups()
            message = message_content.strip()
            if "<Media omitted>" in message:
                message_type = "media"
                parts = message.split(':', 1)
                if len(parts) > 1 and parts[1].strip() == "<Media omitted>": sender = parts[0].strip()
                message = "<Media omitted>"
            elif "this message was deleted" in message.lower():
                message_type = "deleted"
                parts = message.split(':', 1)
                if len(parts) > 1 and "this message was deleted" in parts[1].lower(): sender = parts[0].strip()
                message = "This message was deleted"
            else:
                 sender = "System"
                 message_type = "system"
            timestamp_str = f"{date_str} {time_str} {am_pm.upper()}"
            for fmt in ("%d/%m/%y %I:%M %p", "%d/%m/%Y %I:%M %p"):
                try:
                    timestamp = datetime.strptime(timestamp_str, fmt)
                    break
                except ValueError: continue
        if timestamp:
            chat_data.append({
                "timestamp": timestamp, "sender": sender,
                "message": message, "message_type": message_type
            })
    df = pd.DataFrame(chat_data)
    return df

# ----------------------------
# Database Insertion
# ----------------------------
def insert_chat_data(df: pd.DataFrame, db_path=DB_PATH):
    """Inserts parsed chat data into the database."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    df_to_insert = df.copy()
    df_to_insert["timestamp"] = df_to_insert["timestamp"].astype(str)
    if 'message_type' not in df_to_insert.columns: df_to_insert['message_type'] = 'text'
    db_columns = ['timestamp', 'sender', 'message', 'message_type']
    for col in db_columns:
         if col not in df_to_insert.columns:
              default_val = 'text' if col == 'message_type' else '' # Basic default
              df_to_insert[col] = default_val
    df_to_insert = df_to_insert[db_columns]
    try:
        df_to_insert.to_sql("chat_messages", conn, if_exists="append", index=False)
        conn.commit()
    except Exception as e:
         print(f"Error inserting data: {e}") # Use print or logging
         # st.error(f"Error inserting data: {e}")
         conn.rollback() # Rollback on error
    finally:
        conn.close()


# ----------------------------
# Feature Extraction
# ----------------------------
def extract_features(db_path=DB_PATH):
    """Loads chat data and computes per-sender features."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        df = pd.read_sql("SELECT timestamp, sender, message, message_type FROM chat_messages WHERE sender != 'System'", conn)
    except pd.io.sql.DatabaseError as e:
         if "no such column: message_type" in str(e):
              print("Warning: DB column 'message_type' not found, using default.")
              # st.warning("DB column 'message_type' not found...", icon="⚠️") # Avoid st call
              df = pd.read_sql("SELECT timestamp, sender, message FROM chat_messages WHERE sender != 'System'", conn)
              df['message_type'] = 'text'
         else:
              print(f"Database error during feature extraction: {e}") # Use print/logging
              # st.error(f"Database error: {e}", icon="🚨")
              return None
    finally:
        if conn: conn.close()

    if df.empty: return None

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        df.dropna(subset=["timestamp"], inplace=True)
        if df.empty: return None

        # --- True Response Time ---
        df_time_sorted = df.sort_values("timestamp").copy()
        df_time_sorted['previous_sender'] = df_time_sorted['sender'].shift(1)
        df_time_sorted['previous_timestamp'] = df_time_sorted['timestamp'].shift(1)
        is_response_mask = ((df_time_sorted['sender'] != df_time_sorted['previous_sender']) & df_time_sorted['previous_sender'].notna())
        responses_df = df_time_sorted[is_response_mask].copy()
        if not responses_df.empty:
            responses_df['response_time_seconds'] = (responses_df['timestamp'] - responses_df['previous_timestamp']).dt.total_seconds()
            median_responses = responses_df.groupby('sender')['response_time_seconds'].median()
        else:
            median_responses = pd.Series(dtype=float, index=pd.Index([], name='sender'))

        # --- Other Features ---
        df_orig = df.copy()
        df_orig["hour"] = df_orig["timestamp"].dt.hour
        df_orig["day_of_week"] = df_orig["timestamp"].dt.dayofweek
        df_orig['is_weekend'] = df_orig['day_of_week'] >= 5
        text_mask = df_orig['message_type'] == 'text'
        df_orig["msg_len"] = df_orig.loc[text_mask, "message"].astype(str).str.len()
        df_orig["emoji_count"] = df_orig.loc[text_mask, "message"].astype(str).apply(lambda x: sum(1 for c in x if c in emoji.EMOJI_DATA))
        df_orig["is_question"] = df_orig.loc[text_mask, "message"].astype(str).str.strip().str.endswith('?')
        df_orig["is_deleted_type"] = df_orig['message_type'] == 'deleted'
        df_orig = df_orig.sort_values(["sender", "timestamp"])
        df_orig["time_diff_own"] = df_orig.groupby("sender")["timestamp"].diff().dt.total_seconds()

        # --- Aggregation ---
        agg_funcs = {
            "total_msgs": pd.NamedAgg("message", "count"),
            "text_msgs": pd.NamedAgg("message_type", lambda x: (x == 'text').sum()),
            "media_msgs": pd.NamedAgg("message_type", lambda x: (x == 'media').sum()),
            "media_ratio": pd.NamedAgg("message_type", lambda x: (x == 'media').sum() / len(x) if len(x) > 0 else 0),
            "avg_msg_len": pd.NamedAgg("msg_len", 'mean'),
            "total_emojis": pd.NamedAgg("emoji_count", 'sum'),
            "avg_emojis": pd.NamedAgg("emoji_count", lambda x: x.sum() / (x.notna()).sum() if (x.notna()).sum() > 0 else 0),
            "median_time_between_msgs": pd.NamedAgg("time_diff_own", 'median'),
            "night_owl_ratio": pd.NamedAgg("hour", lambda h: ((h >= 22) | (h < 6)).mean()),
            "morning_bird_ratio": pd.NamedAgg("hour", lambda h: ((h >= 5) & (h < 9)).mean()),
            "deleted_msgs": pd.NamedAgg("is_deleted_type", 'sum'),
            "question_msgs": pd.NamedAgg("is_question", 'sum'),
            "question_ratio": pd.NamedAgg("is_question", lambda q: q.sum() / (q.notna()).sum() if (q.notna()).sum() > 0 else 0),
            "weekend_ratio": pd.NamedAgg("is_weekend", 'mean')
        }
        features = df_orig.groupby("sender").agg(**agg_funcs).reset_index()

        # --- Merge True Median Response Time ---
        features = pd.merge(features, median_responses.rename('true_median_response_time'), on='sender', how='left')
        other_cols_to_fill = features.columns.difference(['sender', 'true_median_response_time'])
        features[other_cols_to_fill] = features[other_cols_to_fill].fillna(0)

        return features

    except Exception as e:
         print(f"An error occurred during feature extraction: {e}") # Use print/logging
         import traceback
         print(traceback.format_exc())
         # st.error(f"An error occurred during feature extraction: {e}", icon="🔥")
         # st.error(traceback.format_exc())
         return None

# ----------------------------
# Badge Assignment Logic
# ----------------------------
def assign_badges(features_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates ranks and assigns badges based on thresholds."""
    if features_df is None or features_df.empty:
        print("Assign Badges: No feature data.") # Use print/logging
        # st.warning("Cannot assign badges: No feature data available.")
        return features_df if features_df is not None else pd.DataFrame(columns=['sender', 'badges'])

    if 'badges' not in features_df.columns:
        features_df['badges'] = [[] for _ in range(len(features_df))]

    if len(features_df) < 3:
        print(f"Assign Badges: Skipped ranking (found {len(features_df)} users, requires >= 3).") # Use print/logging
        # st.info(f"Badge assignment based on ranking skipped...")
        return features_df # Return with empty badges list

    # --- Define Badge Criteria ---
    badge_criteria = [
        ('total_msgs', 0.80, '💬 Chatterbox', True),
        ('avg_msg_len', 0.80, '✍️ Verbose Writer', True),
        ('avg_emojis', 0.80, '😂 Emoji Enthusiast', True),
        ('media_ratio', 0.75, '🖼️ Media Maven', True),
        ('median_time_between_msgs', 0.20, '💨 Rapid Fire', False),
        ('true_median_response_time', 0.20, '⚡ Quick Responder', False), # Badge for new feature
        ('night_owl_ratio', 0.80, '🦉 Night Owl', True),
        ('morning_bird_ratio', 0.80, '☀️ Early Bird', True),
        ('deleted_msgs', 0.90, '👻 Phantom Deleter', True),
        ('question_ratio', 0.80, '❓ Inquisitive Mind', True),
        ('weekend_ratio', 0.70, '🎉 Weekend Warrior', True),
    ]
    badges_assigned = {sender: [] for sender in features_df['sender']}

    for feature, threshold, badge, high_is_good in badge_criteria:
        if feature in features_df.columns and features_df[feature].notna().any():
            try:
                ranks = features_df[feature].rank(pct=True, method='dense', na_option='keep')
                for idx, rank in ranks.items():
                    if idx not in features_df.index: continue
                    sender = features_df.loc[idx, 'sender']
                    if pd.isna(rank): continue
                    if badge in badges_assigned[sender]: continue
                    if high_is_good:
                        if rank >= threshold: badges_assigned[sender].append(badge)
                    else:
                        if rank <= threshold: badges_assigned[sender].append(badge)
            except Exception as e:
                 print(f"Error assigning badge '{badge}' for feature '{feature}': {e}") # Use print/logging
                 # st.error(f"Error assigning badge '{badge}' for feature '{feature}': {e}", icon="🚨")

    features_df['badges'] = features_df['sender'].map(badges_assigned)
    return features_df


# --- Add this dictionary to your utils.py file ---

BADGE_METADATA = {
    '💬 Chatterbox': {
        'description': 'Sends a high volume of messages (Top 20%).',
        'image_url': 'https://cdn-icons-png.flaticon.com/128/1041/1041916.png' # Example icon URL
    },
    '✍️ Verbose Writer': {
        'description': 'Writes longer messages on average (Top 20%).',
        'image_url': 'https://cdn-icons-png.flaticon.com/128/3131/3131611.png'
    },
    '😂 Emoji Enthusiast': {
        'description': 'Uses emojis frequently (Top 20%).',
        'image_url': 'https://cdn-icons-png.flaticon.com/128/1384/1384061.png'
    },
    '🖼️ Media Maven': {
        'description': 'Sends a high proportion of media messages (Top 25%).',
        'image_url': 'https://cdn-icons-png.flaticon.com/128/1375/1375106.png'
    },
    '💨 Rapid Fire': {
        'description': 'Sends consecutive messages quickly (Bottom 20% time between own msgs).',
        'image_url': 'https://cdn-icons-png.flaticon.com/128/891/891917.png'
    },
    '⚡ Quick Responder': {
        'description': 'Responds quickly to others (Bottom 20% true response time).',
        'image_url': 'https://cdn-icons-png.flaticon.com/128/870/870183.png'
    },
    '🦉 Night Owl': {
        'description': 'Active late at night or early morning (Top 20%).',
        'image_url': 'https://cdn-icons-png.flaticon.com/128/2839/2839231.png'
    },
    '☀️ Early Bird': {
        'description': 'Active early in the morning (Top 20%).',
        'image_url': 'https://cdn-icons-png.flaticon.com/128/3236/3236760.png'
    },
    '👻 Phantom Deleter': {
        'description': 'Deletes messages more often than others (Top 10%).',
        'image_url': 'https://cdn-icons-png.flaticon.com/128/616/616566.png'
    },
    '❓ Inquisitive Mind': {
        'description': 'Asks a high proportion of questions (Top 20%).',
        'image_url': 'https://cdn-icons-png.flaticon.com/128/1055/1055641.png'
    },
    '🎉 Weekend Warrior': {
        'description': 'Sends a high proportion of messages on weekends (Top 30%).',
        'image_url': 'https://cdn-icons-png.flaticon.com/128/3408/3408343.png'
    }
    # Add more badges here if you define others
}
# --- End of addition to utils.py ---