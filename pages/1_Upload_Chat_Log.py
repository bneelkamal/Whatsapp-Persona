# pages/1_Upload_Chat_Log.py

import streamlit as st
import pandas as pd
from utils import parse_whatsapp_chat, insert_chat_data # Import functions from utils

st.subheader("⬆️ Upload Chat Log") # Use emojis in title!

uploaded_file = st.file_uploader("Upload your WhatsApp chat log (.txt)", type="txt")

if uploaded_file is None:
    st.info("Please upload a file to begin analysis.")
else:
    try:
        # Read file content, trying different encodings if utf-8 fails
        try:
            file_content = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            st.warning("UTF-8 decoding failed, trying 'latin-1' encoding.", icon="⚠️")
            uploaded_file.seek(0) # Reset file pointer
            file_content = uploaded_file.read().decode("latin-1")
        st.success("File read successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop() # Stop execution if file read fails

    # Add a preview of the raw file content
    with st.expander("Raw File Preview (First 10 lines)"):
        st.text_area("Raw Preview", "\n".join(file_content.splitlines()[:10]), height=150, disabled=True, label_visibility="collapsed")

    df_parsed = parse_whatsapp_chat(file_content)

    if df_parsed.empty:
        st.warning("No messages parsed. Check the file format and content. Common issues include incorrect date/time format or encrypted messages.", icon="⚠️")
    else:
        st.write("### Parsed Chat Data (Preview)")
        st.dataframe(df_parsed.head())
        st.write(f"Total messages parsed (including potential System/Media): {len(df_parsed)}")
        user_messages = df_parsed[df_parsed['sender'] != 'System']
        st.write(f"Messages assigned to users: {len(user_messages)}")
        st.write(f"Unique users found: {user_messages['sender'].nunique()}")

        if st.button("✅ Insert Parsed Data into Database"):
            with st.spinner("Inserting data..."):
                insert_chat_data(df_parsed) # Use function from utils
            st.success("Data inserted successfully into the database!")
            st.info("Navigate to 'Update Sender Names' or 'Analyzer' using the sidebar.")
            # No rerun needed typically in multi-page apps, user navigates away