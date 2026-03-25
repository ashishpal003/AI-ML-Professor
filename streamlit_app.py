import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AI-ML Professor",
    layout="wide"
)

# --------- session -----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# ----------- ui ---------------
st.title("📚 AI-ML Professor")
st.caption("Your personal AI tutor powered by RAG")

# ------------- sidebar for file upload ---------
st.sidebar.header("📄 Upload Document (in .pdf format)")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file and not st.session_state.file_uploaded:
    if st.sidebar.button("Upload File"):
        with st.spinner("Uploading and processing..."):
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
            }

            res = requests.post(f"{API_URL}/upload", files=files)

            if res.status_code == 200:
                st.sidebar.success("✅ File processed successfully!")
            else:
                st.sidebar.error("❌ Upload failed")

# ----------- chat ----------------
st.subheader("💬 Chat")

chat_container = st.container()

with chat_container:
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["assistant"])

# ------------- input ------------------
user_query = st.chat_input("Ask something about Machine Learning and AI...")

# ------------- stream response ---------
if user_query:

    # display user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Placeholder for assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        full_response = ""

        try:
            response = requests.post(
                f"{API_URL}/stream",
                json={
                    "query": user_query,
                    "session_id": st.session_state.session_id
                },
                stream=True
            )

            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    token = chunk.decode("utf-8")
                    full_response += token
                    response_placeholder.markdown(full_response)
        
        except Exception as e:
            full_response = "❌ Error connecting to backend"

    # save to history
    st.session_state.chat_history.append({
        "user": user_query,
        "assistant": full_response
    })
