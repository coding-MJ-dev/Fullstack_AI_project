import sys
import asyncio
import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader

st.set_page_config(
    page_title="Investigate webpage AI",
    page_icon="🔍",
)
st.title("🔍 Investigate webpage AI")
st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions about a webiste!!

Investigate webpage AI designed to analyze and extract information from webpages.

Please upload your file on the sidebar!
"""
)

if "win32" in sys.platform:
    # Windows specific event-loop policy & cmd
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    cmds = [["C:/Windows/system32/HOSTNAME.EXE"]]
else:
    # Unix default event-loop policy & cmds
    cmds = [
        ["du", "-sh", "/Users/fredrik/Desktop"],
        ["du", "-sh", "/Users/fredrik"],
        ["du", "-sh", "/Users/fredrik/Pictures"],
    ]

with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

if url:
    # asyn chrominum loader
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    st.write(docs)
