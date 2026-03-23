import streamlit as st
import requests

st.title("RAG Chat Assistant")

query = st.text_input("Ask something:")

if st.button("Send"):
    response = requests.post(
        "http://localhost:8000/stream",
        json={"query": query},
        stream=True
    )

    result = ""
    placeholder = st.empty()

    for chunk in response.iter_content(chunk_size=10):
        if chunk:
            result += chunk.decode()
            placeholder.markdown(result) 