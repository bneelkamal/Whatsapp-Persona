# pages/4_User_Profile.py

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime, timedelta

# --- Import shared functions/constants ---
# Assumes utils.py is in the parent directory or accessible via Python path
try:
    from utils import DB_PATH, extract_features, assign_badges, BADGE_METADATA # Import BADGE_METADATA if defined in utils.py
except ImportError:
    st.error("Failed to import from utils.py. Make sure utils.py is accessible.")
    # Define BADGE_METADATA here as a fallback if not using Option A (less recommended)
    BADGE_METADATA = {
        '💬 Chatterbox': {'description': 'Sends a high volume of messages (Top 20%).', 'image_url': 'https://cdn-icons-png.flaticon.com/128/1041/1041916.png'},
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
    }
    # --- End of fallback definition ---


# --- Page Configuration ---
st.set_page_config(page_title="User Profile", layout="wide")
st.title("👤 User Profile Viewer")

# --- Helper Function to get Usernames ---
@st.cache_data(ttl=600) # Cache for 10 minutes
def get_usernames(db_path=DB_PATH):
    """Fetches distinct user sender names from the database."""
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        # Exclude 'System' sender
        query = "SELECT DISTINCT sender FROM chat_messages WHERE sender IS NOT NULL AND sender != 'System' ORDER BY sender"
        df_users = pd.read_sql_query(query, conn)
        conn.close()
        return df_users['sender'].tolist()
    except Exception as e:
        st.error(f"Error fetching usernames from database: {e}")
        return []

# --- Helper function to get user activity data ---
@st.cache_data(ttl=600)
def get_user_activity(username, db_path=DB_PATH):
    """Fetches daily message count for a specific user."""
    try:
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
            df_activity['activity_date'] = pd.to_datetime(df_activity['activity_date'])
        return df_activity
    except Exception as e:
        st.error(f"Error fetching activity data for {username}: {e}")
        return pd.DataFrame(columns=['activity_date', 'message_count']) # Return empty df on error

# --- Run Feature Extraction and Badge Assignment (Cached) ---
# We run this once to get data for all users, then filter
@st.cache_data(ttl=600) # Cache the results
def get_all_features_and_badges(db_path=DB_PATH):
    """Extracts features and assigns badges for all users."""
    print(f"[{datetime.now()}] Running feature extraction and badge assignment for profile page...") # For debugging cache
    features_df = extract_features(db_path=db_path)
    if features_df is not None and not features_df.empty:
        features_with_badges_df = assign_badges(features_df.copy())
        return features_with_badges_df
    else:
        return None

# --- User Selection ---
all_users_features_badges = get_all_features_and_badges()

if all_users_features_badges is None or all_users_features_badges.empty:
    st.warning("Could not retrieve user features. Analysis might not have been run or data is unavailable.", icon="⚠️")
    st.info("Please ensure data is uploaded and try running the 'Analyzer' page first if issues persist.")
    usernames = get_usernames() # Try getting basic usernames anyway for selection
    if not usernames:
        st.stop() # Stop if absolutely no users found
else:
    usernames = sorted(all_users_features_badges['sender'].unique()) # Get names from features df

st.subheader("Select User")
select_prompt = "-- Select Username --"
display_usernames = [select_prompt] + usernames

selected_username = st.selectbox(
    "Choose a user profile to view:",
    options=display_usernames,
    key="profile_page_user_select",
    label_visibility="collapsed"
)

st.divider()

# --- Display Profile if User Selected ---
if selected_username != select_prompt:

    # --- Get Data for Selected User ---
    user_features = None
    user_badges = []
    if all_users_features_badges is not None:
        user_data_row = all_users_features_badges[all_users_features_badges['sender'] == selected_username]
        if not user_data_row.empty:
            user_features = user_data_row.iloc[0] # Get the series for the user
            user_badges = user_features.get('badges', []) # Get badge list

    user_activity_df = get_user_activity(selected_username)

    # --- Display Layout ---
    st.header(f"📊 Profile: {selected_username}")

    col_info, col_activity = st.columns([1, 2]) # Adjust ratio as needed

    with col_info:
        st.subheader("Key Stats")
        if user_features is not None:
            st.metric("Total Messages Sent", f"{user_features.get('total_msgs', 0):,.0f}")
            avg_len = user_features.get('avg_msg_len', 0)
            st.metric("Avg. Message Length", f"{avg_len:.1f} chars" if avg_len else "N/A")
            med_resp = user_features.get('true_median_response_time') # Use the better metric
            if pd.notna(med_resp):
                st.metric("Median Response Time", f"{med_resp / 60:.1f} min" if med_resp > 60 else f"{med_resp:.1f} sec")
            else:
                 st.metric("Median Response Time", "N/A")
            media_rat = user_features.get('media_ratio', 0)
            st.metric("Media Message Ratio", f"{media_rat:.1%}")
            # Add more metrics you find interesting
            q_rat = user_features.get('question_ratio', 0)
            st.metric("Question Ratio", f"{q_rat:.1%}")

        else:
            st.info("Detailed features not available.")
            # Optionally calculate basic total messages here from activity_df if needed as fallback
            if not user_activity_df.empty:
                 st.metric("Total Messages Sent", f"{user_activity_df['message_count'].sum():,.0f}")


    with col_activity:
        st.subheader("📅 Activity Over Time")
        if not user_activity_df.empty:
            try:
                # Ensure date range covers the full period for better visualization
                date_range = pd.date_range(start=user_activity_df['activity_date'].min(), end=user_activity_df['activity_date'].max(), freq='D')
                # Reindex to include days with zero messages
                activity_full_range = user_activity_df.set_index('activity_date').reindex(date_range, fill_value=0).reset_index().rename(columns={'index': 'activity_date'})

                fig = px.bar(activity_full_range, x='activity_date', y='message_count',
                             title="Messages Sent Per Day",
                             labels={'activity_date': 'Date', 'message_count': 'Messages Sent'})
                fig.update_layout(title_x=0.5, margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as plot_e:
                st.error(f"Could not display activity graph: {plot_e}")
        else:
            st.info("No activity data found for this user.")

    st.divider()

    # --- Display Badges ---
    st.subheader("🏆 Badges Earned")
    if user_badges:
        unique_badges = sorted(list(set(user_badges))) # Ensure unique badges
        num_badges = len(unique_badges)
        max_cols = 7 # Adjust as needed
        num_rows = (num_badges + max_cols - 1) // max_cols

        badge_idx = 0
        for _ in range(num_rows):
            cols = st.columns(max_cols)
            for i in range(max_cols):
                if badge_idx < num_badges:
                    badge_name = unique_badges[badge_idx]
                    badge_info = BADGE_METADATA.get(badge_name)

                    if badge_info:
                        with cols[i]:
                            try:
                                st.image(
                                    badge_info.get('image_url', ''), # Safely get URL
                                    caption=badge_name,
                                    help=badge_info.get('description', 'No description'), # Tooltip
                                    width=80 # Adjust size
                                )
                            except Exception as img_e:
                                with cols[i]:
                                    badge_name_for_error = badge_info.get('name', badge_idx)
                                    st.caption(f"{badge_name_for_error} (Image Error)") # Clearly indicates error
                            # You can optionally log the detailed error to the console for debugging:
                            # print(f"DEBUG: Failed to load image for badge '{badge_name_for_error}': {img_e}")
                    else:
                        # Badge name exists but no metadata found
                         with cols[i]:
                            st.caption(f"{badge_name}")
                            st.error("❓", help=f"Metadata missing for badge: {badge_name}")
                    badge_idx += 1
                else:
                    pass # Empty columns

    else:
         st.info("No badges assigned to this user based on current analysis.")

else:
    st.info("Select a username above to view their profile.")