import streamlit as st

st.set_page_config(page_title="Project5", page_icon="🧩", layout="centered")

st.title("Project5")
st.caption("A simple standalone Streamlit app template.")

col1, col2 = st.columns(2)
with col1:
    st.metric("Users", 123)
    st.metric("Errors", 2)
with col2:
    st.metric("Sessions", 456)
    st.metric("Latency (ms)", 87)
