import streamlit as st

st.set_page_config(page_title="Project6", page_icon="🧩", layout="centered")

st.title("Project6")
st.caption("A simple standalone Streamlit app template.")

opt = st.selectbox("Pick one", ["Alpha", "Beta", "Gamma"]) 
st.write("You picked:", opt)
