import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline
import subprocess, json, os

# --------------------------
# Helper Functions
# --------------------------

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    video_ids = qs.get('v')
    if not video_ids:
        raise ValueError(f"No video id found in URL: {url}")
    return video_ids[0]

def fetch_transcript_api(video_id: str, lang="en"):
    """Try youtube-transcript-api first."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
        return " ".join([t["text"] for t in transcript])
    except (TranscriptsDisabled, NoTranscriptFound, Exception):
        return None

def fetch_transcript_ytdlp(video_url: str):
    """Fallback: use yt-dlp to download captions."""
    video_id = extract_video_id(video_url)
    filename = f"{video_id}.en.json3"

    subprocess.run(
        ["yt-dlp", "--skip-download", "--write-auto-subs",
         "--sub-lang", "en", "--sub-format", "json3",
         "-o", f"{video_id}.%(ext)s", video_url],
        capture_output=True
    )

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = " ".join([e["segs"][0]["utf8"] for e in data["events"] if "segs" in e])
        os.remove(filename)  # clean up
        return text
    return None

# --------------------------
# Streamlit App
# --------------------------

st.title("🎥 YouTube Video Summarizer")
st.write("Paste a YouTube link and get an automatic summary of the transcript.")

url = st.text_input("Enter YouTube Video URL:")

if url:
    try:
        st.info("⏳ Fetching transcript...")
        video_id = extract_video_id(url)

        # Try transcript API first
        text = fetch_transcript_api(video_id, "en")

        # If failed, fallback to yt-dlp
        if not text:
            text = fetch_transcript_ytdlp(url)

        if not text:
            st.error("❌ No transcript available for this video.")
        else:
            # Load summarizer
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

            with st.spinner("Summarizing... Please wait..."):
                summary = summarizer(text, max_length=120, min_length=40, do_sample=False)

            st.subheader("Summary:")
            st.write(summary[0]["summary_text"])

    except Exception as e:
        st.error(f"Error: {e}")
