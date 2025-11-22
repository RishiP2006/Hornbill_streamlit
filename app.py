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
# CONFIGURATION
# ===========================

# HuggingFace repo (must be repo ID, not URL)
HF_REPO_ID = "RishiPTrial/Hornbill_detector"

# Weight file inside the repo
HF_WEIGHTS_FILENAME = "hornbill_nesthole_y11s_best-2.pt"

# Class information
HORN_BILL_CLASS_NAME = "hornbill"
HORN_BILL_CLASS_ID = 0     # (optional fallback)


# ===========================
# LOAD MODEL
# ===========================

@st.cache_resource(show_spinner=True)
def load_model():
    """Download YOLO weights from HuggingFace and load the model."""
    token = os.getenv("HF_TOKEN", None)

    weight_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_WEIGHTS_FILENAME,
        use_auth_token=token,   # if private, token must be set in Streamlit Secrets
    )

    model = YOLO(weight_path)
    return model


# ===========================
# DETECTION LOGIC
# ===========================

def detect_hornbill_in_frame(frame: np.ndarray, model: YOLO, conf_threshold=0.25):
    """Return True if hornbill is detected inside a frame."""
    results = model(frame, verbose=False)[0]
    names = results.names

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < conf_threshold:
            continue

        class_name = names.get(cls_id, "").lower()

        # Check by class name
        if class_name == HORN_BILL_CLASS_NAME.lower():
            return True

        # Optional fallback: class ID
        if cls_id == HORN_BILL_CLASS_ID:
            return True

    return False


# ===========================
# FRAME EXTRACTION (NO OPENCV)
# ===========================

def extract_frames(video_path, fps=2):
    """
    Extract frames using ffmpeg at fixed FPS.
    Returns: ([(timestamp, frame_array)], video_duration_seconds)
    """
    # Probe video metadata
    probe = ffmpeg.probe(video_path)
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    duration = float(video_stream["duration"])

    # Extract frames as PNG bytes
    out, _ = (
        ffmpeg
        .input(video_path)
        .filter("fps", fps=fps)
        .output("pipe:", format="image2pipe", vcodec="png")
        .run(capture_stdout=True, capture_stderr=True)
    )

    frames = []
    pointer = 0
    index = 0

    while pointer < len(out):
        try:
            data = io.BytesIO(out[pointer:])
            img = Image.open(data)
            frame_arr = np.array(img)

            timestamp = index / fps
            frames.append((timestamp, frame_arr))

            # Advance pointer by reading actual PNG size
            img.load()
            img_bytes = data.getbuffer().nbytes
            pointer += img_bytes
            index += 1

        except Exception:
            break

    return frames, duration


# ===========================
# TIME SEGMENT MERGING
# ===========================

def merge_times_to_segments(times: List[float], gap=1.0):
    """Merge timestamps into continuous presence intervals."""
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
    """Convert seconds → mm:ss.mmm format."""
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
st.write("Upload a video to detect hornbill presence, intervals, and total duration.")


uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
    fps_extract = st.slider("FPS sampling rate", 1, 5, 2)
    gap = st.slider("Gap tolerance (seconds)", 0.0, 5.0, 1.0, 0.5)


if uploaded:
    st.video(uploaded)

    if st.button("Analyze video", type="primary"):
        # Save uploaded file to temp dir
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name

        st.info("Loading YOLO model…")
        model = load_model()

        st.info("Extracting frames using ffmpeg…")
        frames, duration = extract_frames(video_path, fps=fps_extract)

        st.info(f"Processing {len(frames)} frames…")
        progress = st.progress(0)
        hornbill_times = []

        for idx, (timestamp, frame) in enumerate(frames):
            if detect_hornbill_in_frame(frame, model, conf_threshold=conf):
                hornbill_times.append(timestamp)

            progress.progress((idx + 1) / len(frames))

        if not hornbill_times:
            st.warning("No hornbill detected.")
        else:
            segments = merge_times_to_segments(hornbill_times, gap)
            total_secs = sum((end - start) for start, end in segments)
            total_mins = total_secs / 60

            st.success("Hornbill detected in the video!")
            st.metric("Total Presence (minutes)", f"{total_mins:.2f} min")

            st.write("### Presence Intervals")
            rows = []

            for i, (start, end) in enumerate(segments):
                duration_seg = end - start
                rows.append({
                    "Segment": i + 1,
                    "Start (sec)": round(start, 3),
                    "End (sec)": round(end, 3),
                    "Start (time)": format_time(start),
                    "End (time)": format_time(end),
                    "Duration (sec)": round(duration_seg, 3),
                    "Duration": format_time(duration_seg),
                })

            df = pd.DataFrame(rows)
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="hornbill_intervals.csv",
                mime="text/csv",
            )

        # remove temp file
        try:
            os.remove(video_path)
        except:
            pass

else:
    st.info("Upload a video to begin analysis.")
