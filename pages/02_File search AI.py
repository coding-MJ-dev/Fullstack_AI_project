import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain.storage import LocalFileStore


st.set_page_config(
    page_title="File search AI",
    page_icon="🗒️",
)
st.title("🗒️ File search AI")
st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions about your documentation!

It will answer based on the content inside the files.

Please upload your file on the sidebar!
"""
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    memory_key="chat_history",
    return_messages=True,
)


@st.cache_data(show_spinner="Embedding file...")
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
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def save_memory(input, output):
    st.session_state["chat_history"].append({"input": input, "output": output})


def load_memory(input):
    return memory.load_memory_variables({})["chat_history"]


def invoke_chain(message):
    result = chain.invoke(message)
    save_memory(message, result.content)
    print(result)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """

            Answer the question using ONLY the following context and not your trading data. If you don't know the answer just say you don't know. DON'T make anything up.            
            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message(
        "Hi, I'm ready to answer! please ask anything about your file.",
        "ai",
        save=False,
    )
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "chat_history": load_memory,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            invoke_chain(message)
            memory.load_memory_variables({})


else:
    st.session_state["messages"] = []
    st.session_state["chat_history"] = []
