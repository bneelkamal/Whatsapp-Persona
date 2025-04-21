# app.py

import streamlit as st
from utils import init_db, DB_PATH # Import necessary items from utils

# Initialize DB when the main app starts
init_db()

# Set page config (optional, but good practice)
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="📊",
    layout="wide"
)

# Main App Title
st.title("📊 WhatsApp Chat Analysis App")

# You can add introductory text or general info here if needed
st.markdown("""
Welcome to the WhatsApp Chat Analyzer!

Use the navigation sidebar (click the `>` arrow if hidden) to:
1.  **Upload** your exported chat log.
2.  **Update** sender names if needed.
3.  **Analyze** the chat data to see features and badges!
4.  **User Profile**          
""")

st.sidebar.success("Select a page above.")