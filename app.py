import os
import tempfile
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


# =========================
# CONFIG: MODEL PARAMETERS
# =========================

# Hugging Face repo where your hornbill detector lives
HF_REPO_ID = "https://huggingface.co/RishiPTrial/Hornbill_detector"   # <-- keep / change as needed

# The filename of your YOLO weights inside that repo
HF_WEIGHTS_FILENAME = "hornbill_nesthole_y11s_best-2.pt"                # <-- change if your file is named differently

# The expected hornbill class name or class ID in your model
HORN_BILL_CLASS_NAME = "hornbill"              # <-- change if your class label is different
HORN_BILL_CLASS_ID = 0                         # <-- set to None if you want to rely only on name


# =========================
# HELPER FUNCTIONS
# =========================

@st.cache_resource(show_spinner=True)
def load_model():
    """
    Downloads the YOLO weights from Hugging Face (if needed) and loads the model.
    If the repo is private, set HF_TOKEN as an environment variable in Streamlit Cloud.
    """
    token = os.getenv("HF_TOKEN", None)

    weight_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_WEIGHTS_FILENAME,
        use_auth_token=token,
    )

    model = YOLO(weight_path)
    return model


def detect_hornbill_in_frame(
    frame: np.ndarray,
    model: YOLO,
    conf_threshold: float = 0.25,
) -> bool:
    """
    Runs the model on a single frame and returns True if hornbill is detected.
    """
    results = model(frame, verbose=False)[0]

    names = results.names  # dict: class_id -> class_name

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < conf_threshold:
            continue

        class_name = names.get(cls_id, "")

        # Check by class name (preferred) or by class id fallback
        name_match = (
            HORN_BILL_CLASS_NAME is not None
            and class_name.lower() == HORN_BILL_CLASS_NAME.lower()
        )
        id_match = HORN_BILL_CLASS_ID is not None and cls_id == HORN_BILL_CLASS_ID

        if name_match or id_match:
            return True

    return False


def merge_times_to_segments(times: List[float], gap_tolerance: float) -> List[Tuple[float, float]]:
    """
    Given a sorted list of timestamps (seconds) where hornbill is present,
    merges them into continuous segments. A break > gap_tolerance creates a new segment.
    """
    if not times:
        return []

    segments = []
    seg_start = times[0]
    prev = times[0]

    for t in times[1:]:
        if t - prev <= gap_tolerance:
            # same segment
            prev = t
        else:
            # close previous segment
            segments.append((seg_start, prev))
            seg_start = t
            prev = t

    segments.append((seg_start, prev))
    return segments


def format_time(seconds: float) -> str:
    """
    Format seconds as H:MM:SS.mmm (milliseconds optional).
    """
    if seconds < 0:
        seconds = 0.0

    ms = int((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60

    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}.{ms:03d}"
    else:
        return f"{m:02d}:{s:02d}.{ms:03d}"


# =========================
# STREAMLIT APP
# =========================

st.set_page_config(
    page_title="Hornbill Presence Detector",
    page_icon="🦜",
    layout="centered",
)

st.title("🦜 Hornbill Presence in Video")
st.write(
    """
Upload a video, and this app will:

1. Run your hornbill detection model on each frame  
2. Find **when** the hornbill appears  
3. Report:
   - Time intervals where hornbill is present  
   - Total duration (in minutes & seconds) the hornbill is in the frame
"""
)

with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider(
        "Detection confidence threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Only detections above this confidence are counted.",
    )
    gap_tolerance = st.slider(
        "Gap tolerance between detections (seconds)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.5,
        help="Small gaps shorter than this are merged into one continuous presence.",
    )


uploaded_video = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_video is not None:
    st.video(uploaded_video)

    if st.button("Analyze video for hornbill presence", type="primary"):
        # Save the uploaded file to a temporary location
        suffix = os.path.splitext(uploaded_video.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_video.read())
            tmp_video_path = tmp_file.name

        st.info("Loading model… (first time may take a bit)")
        model = load_model()

        st.info("Reading video and running detection…")

        cap = cv2.VideoCapture(tmp_video_path)
        if not cap.isOpened():
            st.error("Could not open the video. Please try another file.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_secs = frame_count / fps if fps > 0 else None

            if fps <= 0:
                st.error("Could not read FPS from video; timing may not be accurate.")
                fps = 1.0  # fallback to avoid division by zero

            progress_text = "Analyzing frames…"
            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            hornbill_times = []  # timestamps (seconds) where hornbill is present

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / fps  # seconds

                # Run detection
                hornbill_present = detect_hornbill_in_frame(
                    frame, model, conf_threshold=conf_threshold
                )

                if hornbill_present:
                    hornbill_times.append(timestamp)

                frame_idx += 1

                # Update progress
                if frame_count > 0:
                    progress_bar.progress(min(frame_idx / frame_count, 1.0))
                    status_placeholder.write(
                        f"{progress_text} Frame {frame_idx}/{frame_count} "
                        f"({timestamp:.2f} s)"
                    )

            cap.release()

            st.success("Analysis complete ✅")

            # Compute presence segments
            hornbill_times = sorted(hornbill_times)
            segments = merge_times_to_segments(hornbill_times, gap_tolerance)

            if not segments:
                st.warning("No hornbill detected in this video.")
            else:
                total_presence_sec = sum(end - start for start, end in segments)
                total_presence_min = total_presence_sec / 60.0

                st.subheader("Results")

                st.metric(
                    "Total hornbill presence (minutes)",
                    f"{total_presence_min:.2f} min",
                    help=f"Equivalent to {total_presence_sec:.2f} seconds.",
                )

                st.write("### Time intervals where hornbill is in the frame")

                rows = []
                for i, (start, end) in enumerate(segments, start=1):
                    duration = end - start
                    rows.append(
                        {
                            "Segment #": i,
                            "Start (s)": round(start, 3),
                            "End (s)": round(end, 3),
                            "Start (formatted)": format_time(start),
                            "End (formatted)": format_time(end),
                            "Duration (s)": round(duration, 3),
                            "Duration (formatted)": format_time(duration),
                        }
                    )

                import pandas as pd

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                # Allow download as CSV
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download intervals as CSV",
                    data=csv,
                    file_name="hornbill_presence_intervals.csv",
                    mime="text/csv",
                )

        # Clean up temp file
        try:
            os.remove(tmp_video_path)
        except Exception:
            pass
else:
    st.info("Upload a video file to get started.")
