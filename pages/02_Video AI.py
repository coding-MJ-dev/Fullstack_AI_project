import streamlit as st
import subprocess


def extract_audio_from_video(video_path):
    audio_path = video_path.replace("mp4", "mp3")
    command = [
        "ffmpeg",
        # input
        "-i",
        video_path,
        # ignore the vedio => vedio nope => vn
        "-vn",
        audio_path,
    ]
    subprocess.run(command)


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
    if video:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        # wb > write in binary
        with open(video_path, "wb") as f:
            # save the video that user upload to cache file "f"
            f.write(video_content)

        extract_audio_from_video(video_path)
