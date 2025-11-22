import os
import io
import tempfile
from typing import List, Tuple

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import ffmpeg
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


# ===========================
# CONFIG
# ===========================

HF_REPO_ID = "RishiPTrial/Hornbill_detector"      # <-- change if needed
HF_WEIGHTS_FILENAME = "best.pt"                   # <-- change if needed

HORN_BILL_CLASS_NAME = "hornbill"                 # <-- class name in your model
HORN_BILL_CLASS_ID = 0                            # <-- class ID if needed


# ===========================
# UTILITIES
# ===========================

@st.cache_resource(show_spinner=True)
def load_model():
    """Download YOLO weights from HF and load model."""
    token = os.getenv("HF_TOKEN", None)

    weight_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_WEIGHTS_FILENAME,
        use_auth_token=token,
    )

    model = YOLO(weight_path)
    return model


def detect_hornbill_in_frame(frame: np.ndarray, model: YOLO, conf_threshold=0.25):
    """True if hornbill exists in the frame."""
    results = model(frame, verbose=False)[0]
    names = results.names

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < conf_threshold:
            continue

        class_name = names.get(cls_id, "")

        if class_name.lower() == HORN_BILL_CLASS_NAME.lower():
            return True
        if cls_id == HORN_BILL_CLASS_ID:
            return True

    return False


def extract_frames(video_path, fps=2):
    """
    Extract frames via ffmpeg at fixed FPS.
    Returns: ([(timestamp(sec), frame_array)], video_duration_sec)
    """
    # Get video metadata
    probe = ffmpeg.probe(video_path)
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    duration = float(video_stream["duration"])

    # Run ffmpeg to output PNG frames as a byte stream
    out, _ = (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=fps)
        .output('pipe:', format='image2pipe', vcodec='png')
        .run(capture_stdout=True, capture_stderr=True)
    )

    frames = []
    pointer = 0
    index = 0

    while pointer < len(out):
        try:
            # Read one PNG frame from remaining bytes
            byte_stream = io.BytesIO(out[pointer:])
            img = Image.open(byte_stream)
            arr = np.array(img)

            timestamp = index / fps
            frames.append((timestamp, arr))

            # Advance pointer by the size of this PNG file in bytes
            img_bytes = byte_stream.getbuffer().nbytes
            pointer += img_bytes
            index += 1

        except Exception:
            break

    return frames, duration


def merge_times_to_segments(times: List[float], gap=1.0):
    """Merge close timestamps into continuous intervals."""
    if not times:
        return []

    times.sort()
    segments = []
    start = times[0]
    prev = times[0]

    for t in times[1:]:
        if t - prev <= gap:
            prev = t
        else:
            segments.append((start, prev))
            start = t
            prev = t

    segments.append((start, prev))
    return segments


def format_time(seconds: float):
    """Convert seconds → mm:ss.mmm"""
    ms = int((seconds - int(seconds)) * 1000)
    total = int(seconds)
    m = total // 60
    s = total % 60
    return f"{m:02d}:{s:02d}.{ms:03d}"


# ===========================
# STREAMLIT UI
# ===========================

st.set_page_config(page_title="Hornbill Detector", page_icon="🦜")

st.title("🦜 Hornbill Presence Detector")
st.write(
    "Upload a video to detect when hornbill appears, the time intervals, "
    "and the total duration it stays in frame."
)

uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "mkv", "avi"])

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    fps_extract = st.slider("FPS Sampling Rate", 1, 4, 2)
    gap = st.slider("Gap tolerance (seconds)", 0.0, 5.0, 1.0, 0.5)

if uploaded:
    st.video(uploaded)

    if st.button("Analyze Video", type="primary"):
        # Save to temp file
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name

        st.info("Loading model...")
        model = load_model()

        st.info("Extracting frames (ffmpeg)...")
        frames, duration = extract_frames(video_path, fps=fps_extract)

        st.info(f"Running hornbill detection on {len(frames)} frames...")

        hornbill_times = []
        progress = st.progress(0)

        for idx, (timestamp, frame) in enumerate(frames):
            if detect_hornbill_in_frame(frame, model, conf_threshold=conf):
                hornbill_times.append(timestamp)

            progress.progress((idx + 1) / len(frames))

        if not hornbill_times:
            st.warning("No hornbill found.")
        else:
            segments = merge_times_to_segments(hornbill_times, gap)
            total_secs = sum((end - start) for start, end in segments)
            total_minutes = total_secs / 60

            st.success("Hornbill detected!")
            st.metric("Total Presence (minutes)", f"{total_minutes:.2f} min")

            # Show intervals
            st.write("### Presence Intervals")
            rows = [{
                "Segment": i + 1,
                "Start (sec)": round(start, 3),
                "End (sec)": round(end, 3),
                "Start (time)": format_time(start),
                "End (time)": format_time(end),
                "Duration (sec)": round(end - start, 3),
                "Duration": format_time(end - start),
            } for i, (start, end) in enumerate(segments)]

            df = pd.DataFrame(rows)
            st.dataframe(df)

            # Download CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                csv,
                "hornbill_intervals.csv",
                "text/csv"
            )

        # cleanup
        try:
            os.remove(video_path)
        except Exception:
            pass

else:
    st.info("Upload a video to begin.")
