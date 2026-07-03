import streamlit as st

st.set_page_config(
    page_title="Pitch Pundit",
    page_icon="⚽",
    layout="centered",
)

chat_page = st.Page("app.py", title="Pitch IQ Chat", icon="💬", default=True)
pundit_page = st.Page("pundit_app.py", title="Pundit Agent", icon="🎙️")

pg = st.navigation([chat_page, pundit_page])
pg.run()
