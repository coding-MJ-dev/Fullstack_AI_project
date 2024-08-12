import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import os
from dotenv import load_dotenv


# Set the page configuration
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


# hearing when llm start/ end/ and add up tocken to AI message box
class ChatCallBackHandler(BaseCallbackHandler):
    def on_llm_start(self, *args, **kwargs):
        self.message = ""  # Reset message on LLM start
        self.message_box = st.empty()  # Create an empty placeholder for the message

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token  # Accumulate tokens
        self.message_box.markdown(
            self.message
        )  # Update the message box with the accumulated message


try:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallBackHandler(),  # Use the custom callback handler
        ],
    )
except Exception as e:
    st.error(f"Failed to initialize ChatOpenAI: {e}")
    st.stop()


# Ensure the directories exist
os.makedirs("./.cache/files/", exist_ok=True)
os.makedirs("./.cache/embeddings/", exist_ok=True)


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    # Create a FAISS retriever
    retriever = FAISS.from_documents(docs, cached_embeddings)

    return retriever


# class SimpleMemory:
#     def __init__(self):
#         self.context = ""

#     def update_memory(self, new_context):
#         self.context += new_context

#     def get_context(self):
#         return self.context


# if "memory" not in st.session_state:
#     st.session_state.memory = SimpleMemory()


def save_message(message, role):
    if "messages" not in st.session_state:  # Ensure session state has a messages list
        st.session_state["messages"] = []
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    if "messages" in st.session_state:  # Check if there are messages in session state
        for message in st.session_state["messages"]:
            send_message(
                message["message"],
                message["role"],
                save=False,
            )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. Don't make anything up.

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("DocumentGPT")

st.markdown(
    """
    Welcome!
                
    Use this chatbot to ask questions to an AI about your files!

    Upload your files on the sidebar.
    """
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        docs = retriever.similarity_search(message)
        formatted_docs = format_docs(docs)

        # Update memory with the retrieved context
        st.session_state.memory.update_memory(formatted_docs)

        chain_input = {
            "context": st.session_state.memory.get_context(),
            "question": message,
        }

        chain = prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(chain_input)
        # No need to call send_message here as it will be handled by the callback
else:
    st.session_state["messages"] = []  # Initialize messages list if not present
