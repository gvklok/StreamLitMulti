import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Project2", page_icon="🧩", layout="centered")

st.title("Project2")
st.caption("A simple standalone Streamlit app template.")

with st.sidebar:
    st.header("Controls")
    name = st.text_input("Your name", value="Streamlit User")
    intensity = st.slider("Intensity", 0, 100, 35)
    show_chart = st.checkbox("Show sample chart", True)

st.write(f"Hello, {name}! 👋")
st.write("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if show_chart:
    st.bar_chart({"Series B": [3, 1, 4, 1, 5, intensity]})
