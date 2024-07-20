import streamlit as st
from langchain.retrievers import WikipediaRetriever


st.set_page_config(
    page_title="Quiz AI",
    page_icon="💯",
)
st.title("💯 Quiz AI")

with st.sidebar:
    choice = st.selectbox(
        "Choose what you want to use",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],
        )
    else:
        topic = st.text_input("What do you want to search?")
        if topic:
            retriever = WikipediaRetriever(top_k_results=5)
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(topic)
            st.write(docs)
