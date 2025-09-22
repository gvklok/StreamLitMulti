import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Project3", page_icon="🧩", layout="centered")

st.title("Project3")
st.caption("A simple standalone Streamlit app template.")

with st.sidebar:
    st.header("Controls")
    name = st.text_input("Your name", value="Streamlit User")
    num = st.number_input("How many items?", min_value=1, max_value=20, value=5)

st.write(f"Hello, {name}! 👋")
st.write("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

st.write("Generated items:")
for i in range(int(num)):
    st.write(f"- Item {i+1}")
