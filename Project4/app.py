import streamlit as st

st.set_page_config(page_title="Project4", page_icon="🧩", layout="centered")

st.title("Project4")
st.caption("A simple standalone Streamlit app template.")

with st.expander("Details"):
    st.write("This is a minimal app with an expander and a form.")

with st.form("demo"):
    a = st.text_input("Text A", "foo")
    b = st.text_input("Text B", "bar")
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.success(f"Submitted: A={a}, B={b}")
