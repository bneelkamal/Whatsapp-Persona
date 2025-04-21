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


# --- Place this updated code inside your utils.py file ---
# Make sure to replace the OLD extract_features function with this one.
# Also add the new get_db_engine function above it.

import streamlit as st # May be needed for secrets access in get_db_engine
import pandas as pd
import sqlite3
import re
from datetime import datetime
import numpy as np
import emoji
import sqlalchemy # <-- Need this import
import traceback
import os

# Shared Constant for SQLite path
DB_PATH = "chat_data.db"

# --- BADGE_METADATA dictionary should be defined here ---
BADGE_METADATA = {
    '💬 Chatterbox': {'description': 'Sends a high volume of messages (Top 20%).','image_url': 'https://cdn-icons-png.flaticon.com/128/1041/1041916.png'},
    '✍️ Verbose Writer': {'description': 'Writes longer messages on average (Top 20%).', 'image_url': 'https://cdn-icons-png.flaticon.com/128/3131/3131611.png'},
    '😂 Emoji Enthusiast': {'description': 'Uses emojis frequently (Top 20%).', 'image_url': 'https://cdn-icons-png.flaticon.com/128/1384/1384061.png'},
    '🖼️ Media Maven': {'description': 'Sends a high proportion of media messages (Top 25%).', 'image_url': 'https://cdn-icons-png.flaticon.com/128/1375/1375106.png'},
    '💨 Rapid Fire': {'description': 'Sends consecutive messages quickly (Bottom 20% time between own msgs).', 'image_url': 'https://cdn-icons-png.flaticon.com/128/891/891917.png'},
    '⚡ Quick Responder': {'description': 'Responds quickly to others (Bottom 20% true response time).', 'image_url': 'https://cdn-icons-png.flaticon.com/128/870/870183.png'},
    '🦉 Night Owl': {'description': 'Active late at night or early morning (Top 20%).', 'image_url': 'https://cdn-icons-png.flaticon.com/128/2839/2839231.png'},
    '☀️ Early Bird': {'description': 'Active early in the morning (Top 20%).', 'image_url': 'https://cdn-icons-png.flaticon.com/128/3236/3236760.png'},
    '👻 Phantom Deleter': {'description': 'Deletes messages more often than others (Top 10%).', 'image_url': 'https://cdn-icons-png.flaticon.com/128/616/616566.png'},
    '❓ Inquisitive Mind': {'description': 'Asks a high proportion of questions (Top 20%).', 'image_url': 'https://cdn-icons-png.flaticon.com/128/1055/1055641.png'},
    '🎉 Weekend Warrior': {'description': 'Sends a high proportion of messages on weekends (Top 30%).', 'image_url': 'https://cdn-icons-png.flaticon.com/128/3408/3408343.png'}
    # ... (add all other badge definitions if you have more) ...
}

# --- Keep other utils functions like init_db, parse_whatsapp_chat, insert_chat_data ---
# (Make sure they are still present in your utils.py)

# --- NEW Helper Function for DB Connection ---
def get_db_engine(db_type='sqlite', connection_string=None, secrets=None):
    """Creates a SQLAlchemy engine based on db_type."""
    # Default to DB_PATH for sqlite if connection_string is None
    if db_type == 'sqlite' and connection_string is None:
        connection_string = DB_PATH

    if not connection_string:
         print(f"ERROR: Connection info/string is required for db_type '{db_type}'.")
         return None

    try:
        if db_type == 'sqlite':
            # connection_string is the file path for SQLite
            # Ensure the path is treated correctly, especially on different OS
            db_uri = f'sqlite:///{os.path.abspath(connection_string)}'
            print(f"Creating SQLite engine for: {db_uri}") # Debug print
            return sqlalchemy.create_engine(db_uri)
        elif db_type == 'supabase_postgres':
            # connection_string should be the full URI from secrets or passed directly
            supa_conn_str = connection_string # Assume it's passed directly
            # Optional: Fallback to secrets if direct connection_string is missing
            if not supa_conn_str and secrets and 'supabase' in secrets and 'db_connection_string' in secrets['supabase']:
                supa_conn_str = secrets['supabase']['db_connection_string']

            if not supa_conn_str:
                print("ERROR: Supabase DB connection string not found in arguments or secrets.")
                return None
            print("Creating Supabase/PostgreSQL engine.") # Debug print
            return sqlalchemy.create_engine(supa_conn_str)
        else:
            print(f"ERROR: Unsupported db_type '{db_type}'")
            return None
    except Exception as e:
        print(f"Error creating DB engine for {db_type}: {e}")
        print(traceback.format_exc()) # Print full traceback for engine errors
        return None

# --- REFACTORED Feature Extraction Function ---
def extract_features(db_type='sqlite', connection_info=None, secrets=None, table_name='chat_messages'):
    """
    Loads chat data and computes per-sender features from specified DB.
    :param db_type: 'sqlite' or 'supabase_postgres'
    :param connection_info: DB file path for sqlite, or connection string for Supabase/Postgres
    :param secrets: Streamlit secrets object (optional, used if connection_info relies on secrets)
    :param table_name: Name of the table containing chat messages
    """
    engine = get_db_engine(db_type=db_type, connection_string=connection_info, secrets=secrets)

    if engine is None:
        print(f"[{datetime.now()}] Failed to get DB engine for {db_type}. Cannot extract features.")
        return None # Return None if engine creation failed

    print(f"[{datetime.now()}] Attempting feature extraction from {db_type} table '{table_name}'...")

    # Use double quotes for potential mixed-case table names in Postgres
    # Ensure required columns are selected
    query = f"""
        SELECT timestamp, sender, message, message_type
        FROM "{table_name}"
        WHERE sender IS NOT NULL AND sender != 'System'
    """

    df = None
    try:
        # Using context manager ensures connection is closed/returned to pool
        with engine.connect() as connection:
             df = pd.read_sql(query, connection, parse_dates=['timestamp']) # Optimize date parsing
        if df is not None:
             print(f"[{datetime.now()}] Read {len(df)} rows from DB table '{table_name}'.")
        else:
             print(f"[{datetime.now()}] pd.read_sql returned None from table '{table_name}'.") # Should not happen unless error above

    except Exception as e:
        # Attempt fallback if 'message_type' column might be missing
        # Check for common error messages for missing column in SQLite and PostgreSQL
        if 'message_type' in str(e) or \
           'no such column: message_type' in str(e).lower() or \
           'column "message_type" does not exist' in str(e).lower():
            print(f"[{datetime.now()}] Warning: DB column 'message_type' not found in '{table_name}', attempting fallback.")
            alt_query = f"""
                SELECT timestamp, sender, message
                FROM "{table_name}"
                WHERE sender IS NOT NULL AND sender != 'System'
            """
            try:
                with engine.connect() as connection:
                    df = pd.read_sql(alt_query, connection, parse_dates=['timestamp'])
                if df is not None and not df.empty:
                     df['message_type'] = 'text' # Assign default
                     print(f"[{datetime.now()}] Read {len(df)} rows using fallback query and added default 'message_type'.")
                elif df is not None:
                     print(f"[{datetime.now()}] Read 0 rows using fallback query.")

            except Exception as e2:
                print(f"[{datetime.now()}] Database error during fallback feature extraction: {e2}")
                return None # Exit if fallback also fails
        else:
            print(f"[{datetime.now()}] Database error during feature extraction: {e}")
            print(traceback.format_exc())
            return None # Exit on other SQL errors
    finally:
        if engine:
            engine.dispose() # Good practice to dispose engine when done if app is long-running

    # Check if DataFrame is still None or became empty
    if df is None or df.empty:
        print(f"[{datetime.now()}] No data retrieved from database or DataFrame is empty after query/fallback for table '{table_name}'.")
        return None

    # --- Feature Calculation Logic (largely unchanged, added minor safety/logging) ---
    try:
        print(f"[{datetime.now()}] Starting feature calculation for {len(df)} rows...")
        # Timestamps should be parsed by read_sql now, ensure they are timezone-naive or consistent UTC
        # If timestamps have timezone, make them timezone-naive for calculations or handle appropriately
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']) and df['timestamp'].dt.tz is not None:
             print(f"[{datetime.now()}] Converting timezone-aware timestamps to naive (UTC base) for calculation.")
             df['timestamp'] = df['timestamp'].dt.tz_convert(None) # Convert to naive UTC

        df.dropna(subset=["timestamp"], inplace=True) # Drop rows where timestamp parsing failed

        if df.empty:
            print(f"[{datetime.now()}] DataFrame empty after timestamp processing.")
            return None

        # --- True Response Time calculation (as before) ---
        df_time_sorted = df.sort_values("timestamp").copy()
        df_time_sorted['previous_sender'] = df_time_sorted['sender'].shift(1)
        df_time_sorted['previous_timestamp'] = df_time_sorted['timestamp'].shift(1)
        is_response_mask = ((df_time_sorted['sender'] != df_time_sorted['previous_sender']) & df_time_sorted['previous_sender'].notna())
        responses_df = df_time_sorted[is_response_mask].copy()
        if not responses_df.empty:
            responses_df['response_time_seconds'] = (responses_df['timestamp'] - responses_df['previous_timestamp']).dt.total_seconds()
            responses_df = responses_df[responses_df['response_time_seconds'] >= 0] # Filter out negative times
            median_responses = responses_df.groupby('sender')['response_time_seconds'].median()
        else:
            median_responses = pd.Series(dtype=float, index=pd.Index([], name='sender'))

        # --- Other Features calculation (as before) ---
        df_orig = df.copy()
        df_orig["hour"] = df_orig["timestamp"].dt.hour
        df_orig["day_of_week"] = df_orig["timestamp"].dt.dayofweek
        df_orig['is_weekend'] = df_orig['day_of_week'] >= 5
        text_mask = df_orig['message_type'] == 'text'
        df_orig["msg_len"] = df_orig.loc[text_mask, "message"].astype(str).str.len().fillna(0)
        df_orig["emoji_count"] = df_orig.loc[text_mask, "message"].astype(str).apply(lambda x: sum(1 for c in x if c in emoji.EMOJI_DATA)).fillna(0)
        df_orig["is_question"] = df_orig.loc[text_mask, "message"].astype(str).str.strip().str.endswith('?').fillna(False)
        df_orig["is_deleted_type"] = (df_orig['message_type'] == 'deleted').fillna(False)
        df_orig = df_orig.sort_values(["sender", "timestamp"])
        df_orig["time_diff_own"] = df_orig.groupby("sender")["timestamp"].diff().dt.total_seconds()
        # df_orig["time_diff_own"] = df_orig["time_diff_own"].fillna(pd.Timedelta(seconds=0)) # Optional: fill first msg diff

        # --- Aggregation (as before) ---
        agg_funcs = {
            "total_msgs": pd.NamedAgg("message", "count"),
            "text_msgs": pd.NamedAgg("message_type", lambda x: (x == 'text').sum()),
            "media_msgs": pd.NamedAgg("message_type", lambda x: (x == 'media').sum()),
            "media_ratio": pd.NamedAgg("message_type", lambda x: (x == 'media').sum() / len(x) if len(x) > 0 else 0),
            "avg_msg_len": pd.NamedAgg("msg_len", 'mean'),
            "total_emojis": pd.NamedAgg("emoji_count", 'sum'),
            "avg_emojis": pd.NamedAgg("emoji_count", lambda x: x.sum() / (x != 0).sum() if (x != 0).sum() > 0 else 0),
            "median_time_between_msgs": pd.NamedAgg("time_diff_own", 'median'),
            "night_owl_ratio": pd.NamedAgg("hour", lambda h: ((h >= 22) | (h < 6)).mean()),
            "morning_bird_ratio": pd.NamedAgg("hour", lambda h: ((h >= 5) & (h < 9)).mean()),
            "deleted_msgs": pd.NamedAgg("is_deleted_type", 'sum'),
            "question_msgs": pd.NamedAgg("is_question", 'sum'),
            "question_ratio": pd.NamedAgg("is_question", lambda q: q.sum() / (q.notna()).sum() if (q.notna()).sum() > 0 else 0),
            "weekend_ratio": pd.NamedAgg("is_weekend", 'mean')
        }
        features = df_orig.groupby("sender").agg(**agg_funcs).reset_index()

        # --- Merge True Median Response Time (as before) ---
        features = pd.merge(features, median_responses.rename('true_median_response_time'), on='sender', how='left')
        # Fill NaNs in calculated features
        cols_to_fill_zero = ['total_msgs', 'text_msgs', 'media_msgs', 'media_ratio', 'avg_msg_len', 'total_emojis', 'avg_emojis', 'night_owl_ratio', 'morning_bird_ratio', 'deleted_msgs', 'question_msgs', 'question_ratio', 'weekend_ratio']
        for col in cols_to_fill_zero:
             if col in features.columns:
                  features[col] = features[col].fillna(0)

        print(f"[{datetime.now()}] Feature extraction calculation complete. Found features for {len(features)} senders.")
        return features

    except Exception as e:
        print(f"[{datetime.now()}] An error occurred during feature calculation: {e}")
        print(traceback.format_exc())
        return None # Return None if calculation fails


# --- assign_badges function should remain unchanged ---
# ... (make sure it's still present in utils.py) ...
def assign_badges(features_df: pd.DataFrame) -> pd.DataFrame:
     """Calculates ranks and assigns badges based on thresholds."""
     # ... (keep existing code) ...
     return features_df # Make sure it returns the df


# --- get_user_activity function should also be present ---
@st.cache_data(ttl=600)
def get_user_activity(username: str, db_path: str = DB_PATH) -> pd.DataFrame:
     """Fetches daily message count for a specific user from the SQLite DB."""
     # ... (keep existing code with os.path.exists check) ...
     return pd.DataFrame(columns=['activity_date', 'message_count']) # Ensure return on failure


# --- End of code for utils.py ---


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

# --- Add this function inside your utils.py file ---

import os # Make sure os is imported in utils.py
import sqlite3 # Make sure sqlite3 is imported
import pandas as pd # Make sure pandas is imported

# (Assuming DB_PATH is already defined in utils.py)
# DB_PATH = "chat_data.db"

@st.cache_data(ttl=600) # Cache activity data for 10 mins
def get_user_activity(username: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Fetches daily message count for a specific user from the SQLite DB.
    Includes check for DB file existence.
    """
    # Check if DB file exists before trying to connect
    if not os.path.exists(db_path):
        print(f"Warning: Database file '{db_path}' not found when fetching activity for {username}.")
        return pd.DataFrame(columns=['activity_date', 'message_count']) # Return empty

    try:
        # Use check_same_thread=False for Streamlit compatibility
        conn = sqlite3.connect(db_path, check_same_thread=False)
        query = """
            SELECT
                STRFTIME('%Y-%m-%d', timestamp) as activity_date,
                COUNT(*) as message_count
            FROM chat_messages
            WHERE sender = ?
            GROUP BY activity_date
            ORDER BY activity_date;
        """
        df_activity = pd.read_sql_query(query, conn, params=(username,))
        conn.close()
        if not df_activity.empty:
            # Convert date column to datetime objects
            df_activity['activity_date'] = pd.to_datetime(df_activity['activity_date'])
        return df_activity
    except Exception as e:
        print(f"Error fetching activity data for {username} from {db_path}: {e}")
        # Optionally display a subtle error in the UI if needed, e.g., using st.sidebar.error
        return pd.DataFrame(columns=['activity_date', 'message_count']) # Return empty df on error

# --- End of function to add to utils.py ---
