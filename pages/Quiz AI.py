import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader


st.set_page_config(
    page_title="Quiz AI",
    page_icon="💯",
)
st.title("💯 Quiz AI")


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


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
        if file:
            docs = split_file(file)
        docs
    else:
        topic = st.text_input("What do you want to search?")
        if topic:
            retriever = WikipediaRetriever(top_k_results=2)
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(topic)
            st.write(docs)
