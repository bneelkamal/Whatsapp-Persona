# pages/3_Analyzer.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import traceback # For detailed error printing if needed
from datetime import datetime # Import datetime for logging/debugging

# --- Import shared functions/constants ---
# Assumes utils.py is in the parent directory or accessible via Python path
try:
    # Import necessary functions and constants from utils.py
    from utils import DB_PATH, extract_features, assign_badges, BADGE_METADATA
    badge_metadata_available = True
    utils_import_success = True
except ImportError as e:
    st.error(f"Failed to import required items from utils.py: {e}", icon="🚨")
    st.error("Please ensure utils.py exists, is accessible, and contains DB_PATH, extract_features, assign_badges, and BADGE_METADATA.", icon="⚙️")
    # Define fallbacks to allow the script to run partially, but warn user
    badge_metadata_available = False
    utils_import_success = False
    BADGE_METADATA = {}
    DB_PATH = "chat_data.db" # Default fallback path

# --- Page Configuration ---
st.set_page_config(page_title="Chat Analyzer (Local)", layout="wide") # Config specific to this page
st.subheader("📊 Analyzer - User Features and Badges (Local Upload)")

# --- Main Analysis Section ---
# Use a unique key for the button
if st.button("Run Analysis on Uploaded Data", key="analyzer_run_button"):
    if not utils_import_success:
        st.error("Cannot run analysis because essential functions could not be imported from utils.py.", icon="❌")
        st.stop() # Stop execution if utils aren't loaded

    analysis_done = False # Flag to track if analysis steps completed
    with st.spinner("Extracting features from local DB... This might take a moment..."):
        # Specify SQLite type and connection info (path)
        # Ensure DB_PATH is correctly pointing to your local DB file name
        features = extract_features(db_type='sqlite', connection_info=DB_PATH)

    if features is None or features.empty:
        st.warning("Could not extract features from local DB. Have you uploaded data via Page 1?", icon="⚠️")
        st.session_state['analysis_run_success'] = False
        # Clear session state if features are empty to avoid stale data
        if 'features_with_badges_sqlite' in st.session_state:
            del st.session_state['features_with_badges_sqlite']
            print(f"[{datetime.now()}] Cleared session state 'features_with_badges_sqlite' due to empty features.")
    else:
        st.session_state['analysis_run_success'] = True
        st.success("Feature extraction complete!", icon="✅")

        # --- Display Extracted Features ---
        st.write("#### Extracted Sender Features:")
        try:
            numeric_cols = features.select_dtypes(include=np.number).columns.tolist()
            display_features = features.copy()
            # Formatting logic for display
            for col in numeric_cols:
                if col not in display_features.columns: continue
                if display_features[col].isnull().all():
                    display_features[col] = "N/A"; continue
                if display_features[col].isnull().any():
                    display_features[col] = display_features[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                else:
                    if pd.api.types.is_integer_dtype(features[col].dropna()):
                        display_features[col] = display_features[col].map('{:,.0f}'.format)
                    else:
                        display_features[col] = display_features[col].map('{:.2f}'.format)

            # Display the formatted features table
            st.dataframe(display_features.set_index('sender'), use_container_width=True)

        except Exception as e:
            st.error(f"Error displaying features table: {e}", icon="⚠️")
            print(traceback.format_exc()) # Log detailed error to console

        # --- Assign Badges ---
        features_with_badges = None # Initialize
        with st.spinner("Assigning badges..."):
            try:
                # Pass a copy to avoid modifying the original features df if not intended
                features_with_badges = assign_badges(features.copy())
                if features_with_badges is None:
                     st.warning("Badge assignment function returned None.", icon="⚠️")
            except Exception as e:
                 st.error(f"Error during badge assignment: {e}", icon="📛")
                 print(traceback.format_exc())


                 


        if features_with_badges is not None:

            # pages/3_Analyzer.py
# ... inside the 'if features_with_badges is not None:' block ...

        # --- Create badges_str column ---
            if 'badges' in features_with_badges.columns:
                features_with_badges['badges_str'] = features_with_badges['badges'].apply(
                    lambda x: ', '.join(x) if (x and isinstance(x, list) and len(x)>0) else '---'
                )
            else:
                features_with_badges['badges_str'] = '---'

            # --- !!! FIX DATA TYPES BEFORE SAVING !!! ---
            try:
                # Columns derived from summing booleans might end up as object/bool if NaNs were involved before fillna(0)
                bool_sum_cols = ['question_msgs', 'deleted_msgs', 'text_msgs', 'media_msgs'] # Add others if needed
                for col in bool_sum_cols:
                    if col in features_with_badges.columns:
                        # Fill any remaining NaNs with 0 and cast to integer
                        features_with_badges[col] = features_with_badges[col].fillna(0).astype(int)
                print(f"[{datetime.now()}] Ensured integer types for columns: {bool_sum_cols}")
            except Exception as e:
                st.error(f"Error fixing data types before saving state: {e}", icon="⚠️")
            # --- End of data type fix ---


            # --- SAVE TO SESSION STATE ---
            try:
                st.session_state['features_with_badges_sqlite'] = features_with_badges.copy() # Save a copy
                st.success("Analysis results saved for profile view.", icon="💾")
                print(f"[{datetime.now()}] Saved 'features_with_badges_sqlite' to session state.") # Debug log
            except Exception as session_e:
                st.error(f"Failed to save results to session state: {session_e}", icon="⚠️")
                if 'features_with_badges_sqlite' in st.session_state:
                    del st.session_state['features_with_badges_sqlite']
            # --- End of saving ---

            # ... rest of display logic for Analyzer page ...


            
            # --- Display Earned Badges Table ---
            st.write("---")
            st.write("#### ✨ User Badges Earned:")
            try:
                if 'sender' in features_with_badges.columns and 'badges_str' in features_with_badges.columns:
                    badges_display_for_table = features_with_badges[['sender', 'badges_str']].set_index('sender')
                    st.dataframe(badges_display_for_table, use_container_width=True)
                else:
                    st.info("Could not display earned badges (missing 'sender' or 'badges_str' column).")
            except Exception as e:
                 st.error(f"Error displaying earned badges table: {e}", icon="⚠️")


            # --- Visualizations ---
            st.write("---")
            st.write("#### Visualizations:")
            try:
                # Calculate badge count safely
                if 'badges' in features_with_badges.columns:
                    features_with_badges['badge_count'] = features_with_badges['badges'].apply(lambda x: len(x) if isinstance(x, list) else 0)
                else:
                    features_with_badges['badge_count'] = 0

                # Define plot axes
                plot_x_axis = 'total_msgs'
                plot_y_axis = 'true_median_response_time' if 'true_median_response_time' in features_with_badges.columns else 'median_time_between_msgs'

                # Check if necessary columns exist for plotting
                if plot_x_axis in features_with_badges.columns and plot_y_axis in features_with_badges.columns:
                    # Create DataFrame for plotting, handle potential NaNs
                    plot_df = features_with_badges.dropna(subset=[plot_x_axis, plot_y_axis]).copy()

                    if not plot_df.empty:
                        # Define columns for hover data (Corrected Logic)
                        hover_columns = ["sender", "badge_count", plot_y_axis]
                        if 'badges_str' in plot_df.columns: # Check plot_df specifically
                            hover_columns.insert(1, "badges_str")
                        elif 'badges' in plot_df.columns:
                             hover_columns.insert(1, "badges") # Fallback

                        # Generate scatter plot
                        fig = px.scatter(
                            plot_df, x=plot_x_axis, y=plot_y_axis, size="badge_count", color="badge_count",
                            color_continuous_scale=px.colors.sequential.Viridis, # Or choose another scale
                            hover_data=hover_columns,
                            labels={ # Add readable labels
                                plot_x_axis: plot_x_axis.replace('_', ' ').title(),
                                plot_y_axis: plot_y_axis.replace('_', ' ').title(),
                                "badge_count": "Badge Count",
                                "sender": "Sender",
                                "badges_str": "Badges Earned"
                            },
                            title=f"{plot_x_axis.replace('_', ' ').title()} vs. {plot_y_axis.replace('_', ' ').title()} (Size/Color by Badge Count)"
                        )
                        fig.update_traces(marker=dict(sizemin=5)) # Ensure minimum marker size
                        # Update y-axis title if it's a time metric
                        if 'response_time' in plot_y_axis or 'time_between_msgs' in plot_y_axis:
                            fig.update_layout(yaxis_title=f"{plot_y_axis.replace('_', ' ').title()} (seconds)")
                        fig.update_layout(title_x=0.5) # Center title
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.info(f"No data available for plotting after removing missing values for '{plot_x_axis}' and '{plot_y_axis}'.")
                else:
                    # Inform user if essential columns for plotting are missing
                    missing_cols = [col for col in [plot_x_axis, plot_y_axis] if col not in features_with_badges.columns]
                    st.info(f"Required columns ({', '.join(missing_cols)}) for the scatter plot are not available in the feature data.")

            except Exception as e:
                st.error(f"Could not generate scatter plot: {e}", icon="📊")
                print(traceback.format_exc()) # Log detailed error

        else: # Handle case where features_with_badges itself is None after assignment attempt
             st.error("Failed to proceed after badge assignment.", icon="❌")
             # Clear session state if badge assignment failed
             if 'features_with_badges_sqlite' in st.session_state:
                 del st.session_state['features_with_badges_sqlite']
                 print(f"[{datetime.now()}] Cleared session state 'features_with_badges_sqlite' due to badge assignment failure.")


        analysis_done = True # Mark analysis as completed

# End of main analysis block `if st.button...`
#-----------------------------------------------

# Show initial message only if button hasn't been clicked successfully yet
if not st.session_state.get('analysis_run_success', False):
    st.info("Click the 'Run Analysis' button above to generate features, badges, and visualizations for the locally uploaded chat data.")


# --- Badge Legend Section (Always Visible if Metadata Available) ---
st.write("---") # Separator
st.subheader("📖 Badge Legend")

if badge_metadata_available and BADGE_METADATA:
    # Prepare data for the legend table
    legend_data = []
    # Sort alphabetically by badge name for consistency
    for badge_name, metadata in sorted(BADGE_METADATA.items()):
        legend_data.append({
            "Badge": badge_name,
            "Description": metadata.get('description', 'N/A')
            # Can add Image column later if desired
            # "Icon URL": metadata.get('image_url', '')
        })

    if legend_data:
        badge_legend_df = pd.DataFrame(legend_data)
        st.dataframe(
            badge_legend_df,
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No badge definitions found in BADGE_METADATA dictionary.")
elif utils_import_success:
     st.warning("Badge metadata is missing or empty. Ensure BADGE_METADATA is defined correctly in utils.py.", icon="🔧")
else:
     st.error("Cannot display badge legend because utils.py could not be imported correctly.", icon="❌")
# --- End of Badge Legend Section ---
