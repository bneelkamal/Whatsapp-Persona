# pages/2_Update_Sender_Names.py

import streamlit as st
import pandas as pd
import sqlite3
from utils import DB_PATH # Import constant from utils

st.subheader("✏️ Update Sender Names")

conn = None # Initialize conn outside try
try:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # Exclude 'System' sender if it exists in the DB from the update list
    unique_senders_df = pd.read_sql_query("SELECT DISTINCT sender FROM chat_messages WHERE sender != 'System'", conn)
except Exception as e:
    st.error(f"Error connecting to database: {e}")
    st.stop()
finally:
    if conn:
        conn.close()

if unique_senders_df.empty:
    st.info("No user senders found in the database to update. Have you uploaded data yet?")
else:
    st.caption("Update names if WhatsApp didn't export them correctly (e.g., shows phone numbers).")
    mapping = {}
    with st.form("update_names_form"):
        for idx, row in unique_senders_df.iterrows():
            old_name = row["sender"]
            # Ensure unique keys for text inputs even if old_name has special characters
            input_key = f"update_{old_name}_{idx}"
            new_name = st.text_input(f"Update name for '{old_name}'", value=old_name, key=input_key)
            # Basic validation: prevent empty names or just whitespace
            stripped_new_name = new_name.strip()
            if not stripped_new_name:
                 mapping[old_name] = old_name # Will show warning later if attempted change
            else:
                mapping[old_name] = stripped_new_name

        submitted = st.form_submit_button("Apply Name Updates")

        if submitted:
            conn = None
            try:
                conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                cursor = conn.cursor()
                updates_made = 0
                warnings = 0
                for old_name, new_name in mapping.items():
                     # Check again for empty new_name before executing DB update
                    if old_name != new_name:
                        if new_name: # Ensure new name is not empty
                            cursor.execute("UPDATE chat_messages SET sender = ? WHERE sender = ?", (new_name, old_name))
                            updates_made += cursor.rowcount # Count how many rows were affected
                        else:
                            st.warning(f"Skipped update for '{old_name}' because the new name was empty.", icon="⚠️")
                            warnings += 1
                conn.commit()
                if updates_made > 0:
                    st.success(f"Sender names updated successfully! {updates_made} messages affected.")
                elif warnings == 0:
                     st.info("No names were changed.")

                # Refresh and show the updated list after commit
                updated_df = pd.read_sql_query("SELECT DISTINCT sender FROM chat_messages WHERE sender != 'System'", conn)
                st.write("Current distinct user sender names:")
                st.dataframe(updated_df)

            except Exception as e:
                 st.error(f"Error updating database: {e}")
                 if conn: conn.rollback()
            finally:
                 if conn: conn.close()


    # Display current names outside the form as well
    st.write("---")
    st.write("Current distinct user sender names:")
    conn = None
    try:
         conn = sqlite3.connect(DB_PATH, check_same_thread=False)
         current_df = pd.read_sql_query("SELECT DISTINCT sender FROM chat_messages WHERE sender != 'System'", conn)
         st.dataframe(current_df)
    except Exception as e:
         st.error(f"Error fetching current names: {e}")
    finally:
         if conn: conn.close()