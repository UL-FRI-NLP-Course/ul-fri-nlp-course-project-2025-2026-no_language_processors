import streamlit as st

st.title("Hello from ARNES! 🚀")
st.write("Streamlit is working on the cluster.")

name = st.text_input("Enter your name:")
if name:
    st.success(f"Hello, {name}!")
