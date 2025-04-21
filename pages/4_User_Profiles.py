# pages/4_User_Profile.py

import streamlit as st
import pandas as pd
# import sqlite3 # No longer needed directly if get_user_activity is in utils
import plotly.express as px
from datetime import datetime, timedelta # Keep datetime
import os # Keep os for get_user_activity check in utils (or here if not moved)
import traceback # Keep for error logging

# --- !!! MOVE set_page_config HERE !!! ---
# Page Configuration MUST be the first Streamlit command
st.set_page_config(page_title="User Profile (Upload)", layout="wide")
# --- End of moved set_page_config ---


# --- Import shared functions/constants ---
# Assumes utils.py is in the parent directory or accessible via Python path
try:
    # Import DB_PATH for activity query, BADGE_METADATA, and the get_user_activity function
    # Removed extract_features, assign_badges as they are not called directly here anymore
    from utils import DB_PATH, BADGE_METADATA, get_user_activity
    utils_import_success = True
    # Check if BADGE_METADATA was actually imported and is not empty
    badge_metadata_available = 'BADGE_METADATA' in locals() and isinstance(BADGE_METADATA, dict) and BADGE_METADATA
except ImportError as e:
    # Displaying errors here is fine now, as set_page_config was already called
    st.error(f"Failed to import required items from utils.py: {e}", icon="🚨")
    st.error("Ensure utils.py exists, is accessible, and contains DB_PATH, BADGE_METADATA, get_user_activity.", icon="⚙️")
    # Define fallbacks
    utils_import_success = False
    badge_metadata_available = False
    BADGE_METADATA = {} # Define as empty dict to prevent errors later
    DB_PATH = "chat_data.db" # Define fallback path
except Exception as general_e: # Catch other potential import errors
    st.error(f"An unexpected error occurred during imports: {general_e}", icon="🚨")
    utils_import_success = False
    badge_metadata_available = False
    BADGE_METADATA = {}
    DB_PATH = "chat_data.db"

# --- Page Content Starts Here ---
st.title("👤 User Profile Viewer (Based on Uploaded Data)")

# --- Check prerequisites ---
if not utils_import_success:
     st.error("Cannot load page due to import errors from utils.py.")
     st.stop() # Stop execution if essential utils cannot be imported

# --- Get Feature/Badge Data from Session State ---
all_users_features_badges = None
if 'features_with_badges_sqlite' in st.session_state:
    # Load data saved by the Analyzer page (Page 3)
    print(f"[{datetime.now()}] Loading features/badges from session state for User Profile.") # Debug log
    all_users_features_badges = st.session_state['features_with_badges_sqlite']
    # Validate loaded data - check if it's a non-empty DataFrame
    if not isinstance(all_users_features_badges, pd.DataFrame) or all_users_features_badges.empty:
         st.warning("Session state data for profiles is invalid or empty. Please re-run analysis on 'Analyzer (Local Upload)' page.", icon="⚠️")
         all_users_features_badges = None # Reset if data is bad
    # else:
         # st.sidebar.success("Loaded profile data from session.") # Optional success message

# --- REMOVED local function definitions for get_usernames and get_user_activity ---
# --- REMOVED direct call to get_all_features_and_badges() ---


# --- Check Data Availability and Select User ---
if all_users_features_badges is None:
    st.warning("Could not retrieve user features. Please run the analysis on the 'Analyzer (Local Upload)' page first.", icon="⚠️")
    # Provide a link to the analyzer page
    try: # Use page link if available (newer Streamlit versions)
        st.page_link("pages/3_Analyzer.py", label="Go to Analyzer Page", icon="📊")
    except Exception: # Fallback for older versions or if page link fails
        st.info("Navigate to the 'Analyzer (Local Upload)' page via the sidebar.")
    st.stop() # Stop if no feature data is available
else:
    # Ensure 'sender' column exists before proceeding
    if 'sender' not in all_users_features_badges.columns:
         st.error("Feature data loaded from session state is missing the 'sender' column.", icon="❌")
         st.stop()
    # Get usernames from the loaded data
    usernames = sorted(all_users_features_badges['sender'].unique())

# --- User Selection ---
st.subheader("Select User Profile")
select_prompt = "-- Select Username --"
display_usernames = [select_prompt] + usernames
selected_username = st.selectbox(
    "Choose a user profile to view:",
    options=display_usernames,
    key="profile_page_user_select", # Keep unique key
    label_visibility="collapsed"
)

st.divider()

# --- Display Profile if User Selected ---
if selected_username != select_prompt:

    # --- Get Data for Selected User from the Loaded DataFrame ---
    user_features = None
    user_badges = [] # Default to empty list
    try:
        # Use .loc for robust selection based on index (if sender is index) or boolean mask
        if 'sender' in all_users_features_badges.columns:
             user_data_row = all_users_features_badges.loc[all_users_features_badges['sender'] == selected_username]
        else: # Fallback if sender wasn't reset_index'd properly
             st.error("Cannot find 'sender' column to select user.")
             user_data_row = pd.DataFrame() # Empty dataframe

        if not user_data_row.empty:
            user_features = user_data_row.iloc[0] # Get the pandas Series for the user
            # Safely get badges, ensure it's a list
            badges_data = user_features.get('badges', []) # Assumes 'badges' column with lists exists
            if isinstance(badges_data, list):
                 user_badges = badges_data
            elif isinstance(badges_data, str) and badges_data != '---': # Handle if it was stored as string somehow
                 user_badges = [b.strip() for b in badges_data.split(',')]
            # If badges_data is '---' or other non-list, user_badges remains []
        else:
            st.warning(f"Selected user '{selected_username}' not found in the loaded feature data.")

    except Exception as e:
         st.error(f"Error accessing data for selected user '{selected_username}': {e}")
         print(traceback.format_exc())


    # --- Get Activity Data (Calls function from utils.py - Still reads DB) ---
    # This still relies on the temporary SQLite DB existing when this page runs.
    # Ensure get_user_activity is imported correctly from utils
    user_activity_df = get_user_activity(selected_username, db_path=DB_PATH)

    # --- Display Layout ---
    st.header(f"👤 Profile: {selected_username}")
    col_info, col_activity = st.columns([1, 2]) # Adjust ratio as needed

    with col_info:
        st.subheader("Key Stats")
        if user_features is not None:
            # Display metrics safely using .get() and checking pd.notna
            st.metric("Total Messages Sent", f"{user_features.get('total_msgs', 0):,.0f}")
            avg_len = user_features.get('avg_msg_len', 0)
            st.metric("Avg. Message Length", f"{avg_len:.1f} chars" if pd.notna(avg_len) and avg_len else "N/A")
            med_resp = user_features.get('true_median_response_time')
            if pd.notna(med_resp): st.metric("Median Response Time", f"{med_resp / 60:.1f} min" if med_resp > 60 else f"{med_resp:.1f} sec")
            else: st.metric("Median Response Time", "N/A")
            media_rat = user_features.get('media_ratio', 0)
            st.metric("Media Message Ratio", f"{media_rat:.1%}" if pd.notna(media_rat) else "N/A")
            q_rat = user_features.get('question_ratio', 0)
            st.metric("Question Ratio", f"{q_rat:.1%}" if pd.notna(q_rat) else "N/A")
        else:
            st.info("Detailed features unavailable (Run Analyzer first).")
            # Fallback using activity data if possible
            if not user_activity_df.empty:
                 st.metric("Total Messages Sent (from activity)", f"{user_activity_df['message_count'].sum():,.0f}")

    with col_activity:
        st.subheader("📅 Activity Over Time")
        if not user_activity_df.empty:
            try:
                # Fill date range for bar chart
                min_date = user_activity_df['activity_date'].min()
                max_date = user_activity_df['activity_date'].max()
                if pd.notna(min_date) and pd.notna(max_date): # Ensure dates are valid
                    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
                    activity_full_range = user_activity_df.set_index('activity_date').reindex(date_range, fill_value=0).reset_index().rename(columns={'index': 'activity_date'})
                    fig = px.bar(activity_full_range, x='activity_date', y='message_count',
                                title="Messages Sent Per Day", height=300, # Adjust height if needed
                                labels={'activity_date': 'Date', 'message_count': 'Messages Sent'})
                    fig.update_layout(title_x=0.5, margin=dict(t=40, b=10, l=10, r=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                     st.info("Activity data contains invalid dates.")
            except Exception as plot_e:
                st.error(f"Could not display activity graph: {plot_e}")
                print(traceback.format_exc())
        else:
            st.info("No activity data found (DB file might be missing or user has no messages).")

    st.divider()

    # --- Display Badges ---
    st.subheader("🏆 Badges Earned")
    if user_badges: # Check if list is not empty
        unique_badges = sorted(list(set(user_badges)))
        num_badges = len(unique_badges)
        max_cols = 7 # Adjust as needed
        num_rows = (num_badges + max_cols - 1) // max_cols

        badge_idx = 0
        for row_num in range(num_rows):
            cols = st.columns(max_cols)
            for col_num in range(max_cols):
                if badge_idx < num_badges:
                    badge_name = unique_badges[badge_idx]
                    badge_info = BADGE_METADATA.get(badge_name) # Use the imported metadata

                    if badge_info:
                         with cols[col_num]:
                                try:
                                    # --- !!! REMOVE help=... FROM st.image !!! ---
                                    st.image(
                                        badge_info.get('image_url', ''),
                                        caption=badge_name,
                                        # help=badge_info.get('description', 'No description available.'), # <-- REMOVE THIS ARGUMENT
                                        width=80 # Adjust size as needed
                                    )
                                except Exception as img_e:
                                    badge_name_for_error = badge_info.get('name', badge_name)
                                    failed_url = badge_info.get('image_url', 'URL_MISSING_IN_METADATA')
                                    st.caption(f"{badge_name_for_error} (Image Error)")
                                    # Print detailed info to logs
                                    print(f"--- ERROR loading image ---")
                                    # ... (keep detailed logging print statements) ...
                                    print(f"DEBUG: Failed to load image for badge '{badge_name}': {img_e}")
                                    print(f"---------------------------")
                                
                    else:           
                         # Handle badge name present but no metadata found
                         with cols[col_num]:
                          st.caption(f"{badge_name}")
                          # --- !!! Ensure st.error here also has NO help argument !!! ---
                          # st.error("❓", help=f"Metadata missing for {badge_name}") # Incorrect
                          st.error("❓ Definition Missing") # Correct - message only

                    badge_idx += 1
                else:
                    # Optionally add empty placeholders to keep alignment, or just pass
                    # with cols[col_num]:
                    #     st.write("") # Placeholder
                    pass
    elif user_features is not None: # Only say 'no badges' if features were actually loaded
        st.info("This user hasn't earned any badges based on the analysis.")
    else:
        st.info("Badge data unavailable (Run analyzer first).")

else:
    # Message when no user is selected yet
    st.info("Select a username from the dropdown above to view their profile details.")


# --- Badge Legend Section ---
# (This section remains the same - Assuming BADGE_METADATA is loaded correctly)
st.write("---") # Separator
st.subheader("📖 Badge Legend")
if badge_metadata_available and BADGE_METADATA:
    legend_data = []
    for badge_name, metadata in sorted(BADGE_METADATA.items()):
        legend_data.append({
            "Badge": badge_name,
            "Description": metadata.get('description', 'N/A')
        })
    if legend_data:
        badge_legend_df = pd.DataFrame(legend_data)
        st.dataframe(
            badge_legend_df,
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No badge definitions found in BADGE_METADATA.")
elif utils_import_success:
     st.warning("Badge metadata is missing or empty. Ensure BADGE_METADATA is defined correctly in utils.py.", icon="🔧")
else:
     st.error("Cannot display badge legend due to import errors.", icon="❌")
# --- End of Badge Legend Section ---
