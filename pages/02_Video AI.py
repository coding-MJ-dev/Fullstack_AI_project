import streamlit as st
import subprocess
import math
import glob
import openai
from pydub import AudioSegment


@st.cache_data()
def extract_audio_from_video(video_path):
    audio_path = video_path.replace("mp4", "mp3")
    command = [
        "ffmpeg",
        # Are you overwrite? Yes/no
        "-y",
        # input
        "-i",
        video_path,
        # ignore the vedio => vedio nope => vn
        "-vn",
        audio_path,
    ]
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
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    # find mp3 files in the folder
    for file in files:
        # get transcript // append transcribe to destination
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            # add it to the destination
            text_file.write(transcript["text"])


###-------- page start ----------###

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
    chunks_folder = "./.cache/chunks"

    with st.status("Loading video..."):
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        # wb > write in binary
        with open(video_path, "wb") as f:
            # save the video that user upload to cache file "f"
            f.write(video_content)
    with st.status("Extracting audio..."):
        extract_audio_from_video(video_path)
    with st.status("cutting audio segments..."):
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
    with st.status("Transcribing audio..."):
        transcribe_chunks(chunks_folder, transcript_path)
