import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Project1", page_icon="🧩", layout="centered")

st.title("Project1")
st.caption("A simple standalone Streamlit app template.")

with st.sidebar:
    st.header("Controls")
    name = st.text_input("Your name", value="Streamlit User")
    intensity = st.slider("Intensity", 0, 100, 50)
    show_chart = st.checkbox("Show sample chart", True)

st.write(f"Hello, {name}! 👋")
st.write("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if show_chart:
    st.line_chart({"Series A": [1, 5, 2, 6, 2, intensity]})

uploaded = st.file_uploader("Upload a file")
if uploaded:
    st.success(f"Received {uploaded.name} ({uploaded.size} bytes)")
