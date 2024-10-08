import streamlit as st
import subprocess
import math
import glob
import openai
import os
from pydub import AudioSegment
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler


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


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallBackHandler(),  # Use the custom callback handler
    ],
)

# For development purposes
has_transcript = os.path.exists(
    "./.cache/DP_Longest_Common_Subsequence_Leetcode1143.txt"
)


splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)

# Ensure the directories exist
os.makedirs("./.cache", exist_ok=True)
os.makedirs("./.cache/files/", exist_ok=True)
os.makedirs("./.cache/embeddings/", exist_ok=True)


# embed textfile
# @st.cache_resource()
@st.cache_data()
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever


@st.cache_data()
def extract_audio_from_video(video_path, audio_path):
    # audio_path = video_path.replace("mp4", "mp3")
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)


# chunk_size <= minutes to cut
@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]

        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    # --- for development purpose --- #
    # if has_transcript:
    #     return
    # ------------------------------- #
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            # add it to the destination
            text_file.write(transcript["text"])


def text_to_bytes(text):
    return text.encode("utf-8")


class SimpleMemory:
    def __init__(self):
        self.context = ""

    def update_memory(self, new_context):
        self.context += new_context

    def get_context(self):
        return self.context


if "video_memory" not in st.session_state:
    st.session_state.video_memory = SimpleMemory()

# if "summary" not in st.session_state:
#     st.session_state.summary = ""


# ----------- save message for chat -------- #
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


# ---- save chat as a doc, use this context for the next chat ----
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


# ----------- save summary -------- #
# def save_summary(summary):
#     st.session_state["summary"] += summary


###-------- page start ----------###

st.set_page_config(
    page_title="Video Summay AI",
    page_icon="🤓",
)
st.title("🤓 Video Summay AI")
st.markdown(
    """
Welcome to Summay AI!

If you don't have a time to watch all the content / lecture / record file?
Here is the solution. I will summary all the contents for you.
Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "avi", "mkv", "mov"])


if video:
    os.makedirs("./.cache", exist_ok=True)
    os.makedirs(f"./.cache/chunks_{os.path.splitext(video.name)[0]}", exist_ok=True)

    chunks_folder = f"./.cache/chunks_{os.path.splitext(video.name)[0]}"
    transcript_path = f"./.cache/{os.path.splitext(video.name)[0]}.txt"
    # if not os.path.exists(transcript_path):
    if not os.path.exists(transcript_path):
        with st.status("Loading video...") as status:
            video_content = video.read()
            os.makedirs("./.cache", exist_ok=True)
            video_path = f"./.cache/{video.name}"
            audio_path = (
                video_path.replace(".mp4", ".mp3")
                .replace(".avi", ".mp3")
                .replace(".mkv", ".mp3")
                .replace(".mov", ".mp3")
            )

            # wb > write in binary
            with open(video_path, "wb") as f:
                # save the video that user upload to cache file "f"
                f.write(video_content)
            status.update(label="Extracting audio...")
            extract_audio_from_video(video_path, audio_path)
            status.update(label="cutting audio segments...")
            os.makedirs(chunks_folder, exist_ok=True)
            cut_audio_in_chunks(audio_path, 10, chunks_folder)
            status.update(label="Transcribing audio...")
            transcribe_chunks(chunks_folder, transcript_path)

    # --- Transcription download"
    if os.path.exists(transcript_path):
        # QA with summary -------------

        loader = TextLoader(transcript_path)
        docs = loader.load_and_split(text_splitter=splitter)

        # summarize one chain
        first_summary_prompt = ChatPromptTemplate.from_template(
            """
            Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY: 
            """
        )

        first_summary_chain = (
            first_summary_prompt | llm | StrOutputParser()
        )  # StrOutputParser make us don't need to use .content

        # --------------------------------------------------------------------
        # --------------------------- refine chain ---------------------------
        # The refine documents chain constructs a response by looping over the input documents and iteratively updating its answer. (https://js.langchain.com/v0.1/docs/modules/chains/document/refine/)
        # ---------------------------------------------------------------------
        # Summarize all document

        refine_prompt = ChatPromptTemplate.from_template(
            """
            Your job is to produce a final summary.
            We have provided an existing summary up to a certain point: {existing_summary}
            We have the opportunity to refine the existing summary (only if needed) with some more context below.
            ------------
            {context}
            ------------
            Given the new context, refine the original summary.
            If the context isn't useful, RETURN the original summary.
            """
        )

        refine_chain = refine_prompt | llm | StrOutputParser()

        with st.status("Summarizing...") as status:
            #     for i, doc in enumerate(docs[1:]):
            #         st.info(f"Processing document {i+1}/{len(docs)-1}")
            #         summary = refine_chain.invoke(
            #             {
            #                 "existing_summary": summary,
            #                 "context": doc.page_content,
            #             }
            #         )
            # st.write(summary)

            # if st.session_state["summary"]:
            #     summary = st.session_state["summary"]

            # else:

            summary = first_summary_chain.invoke({"text": docs[0].page_content})
            for i, doc in enumerate(docs[1:]):
                status.update(label=f"processing doc{i+1}/{len(docs)-1}")
                summary = refine_chain.invoke(
                    {
                        "existing_summary": summary,
                        "context": doc.page_content,
                    }
                )

            # save_summary(summary)
        st.write(summary)
