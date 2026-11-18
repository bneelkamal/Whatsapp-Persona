# WhatsApp Chat Analyzer (Whatsapp-Persona)

A multi-page Streamlit application for parsing, storing, and analyzing exported WhatsApp chat logs.

## Features
- **Chat Ingestion & Persistence:** Upload exported WhatsApp chat logs and store parsed records in a local SQLite database (`utils.py`).
- **Sender Normalization:** Clean and merge sender names before analysis.
- **Behavioral Analytics & Badges:** Explore activity patterns, interaction metrics, and user profile summaries across pages.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run Home.py
```