import streamlit as st

st.set_page_config(
    page_title="Video AI",
    page_icon="🎬",
)
st.title("🎬 Video AI")
st.markdown(
    """
Welcome to Video AI!

Do you need a transcript of a video? Or don't have enough time to watch a video but need a summary? Or perhaps you want to ask something about the video's content?

##### What you can get
- video transcription
- summary
- ask question about a video

You've come to the right place!

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "avi", "mkv", "mov"])
