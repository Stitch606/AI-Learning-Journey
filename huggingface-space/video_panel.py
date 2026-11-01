
import os


# ==========================================
# 1. RESOLVE VIDEO PATH
# ==========================================

def resolve_video_path(video_input):

    """
    Extract the actual video path from Gradio input.

    Supports:
    - File path strings
    - Gradio file dictionaries
    - Video tuples
    """

    if video_input is None:
        return None

    # Gradio Video may return (video, subtitles)
    if isinstance(video_input, tuple):
        video_input = video_input[0]

    # Gradio File may return a dictionary
    if isinstance(video_input, dict):
        video_input = (
            video_input.get("path")
            or video_input.get("name")
        )

    if not isinstance(video_input, str):
        return None

    if not os.path.isfile(video_input):
        return None

    return video_input


# ==========================================
# 2. INSPECT VIDEO
# ==========================================

def inspect_video(video_path):

    """
    Read basic video metadata using OpenCV.

    Returns a dictionary containing:
    - filename
    - width
    - height
    - fps
    - frames
    - duration
    """

    video_path = resolve_video_path(video_path)

    if video_path is None:
        return None

    import cv2

    capture = cv2.VideoCapture(video_path)

    try:

        if not capture.isOpened():
            return None

        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        duration = (
            frames / fps
            if fps > 0
            else 0
        )

        return {
            "filename": os.path.basename(video_path),
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "frames": frames,
            "duration": round(duration, 2),
            "resolution": f"{width}x{height}"
        }

    finally:
        capture.release()


# ==========================================
# 3. FORMAT VIDEO DURATION
# ==========================================

def format_duration(seconds):

    seconds = max(0, int(seconds or 0))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining = seconds % 60

    if hours > 0:
        return f"{hours:02}:{minutes:02}:{remaining:02}"

    return f"{minutes:02}:{remaining:02}"


# ==========================================
# 4. FORMAT VIDEO INFORMATION
# ==========================================

def format_video_info(metadata):

    if not metadata:
        return "No video selected"

    return (
        f"🎥 {metadata['filename']} | "
        f"{metadata['resolution']} | "
        f"{metadata['fps']} FPS | "
        f"{format_duration(metadata['duration'])}"
    )


# ==========================================
# 5. PREPARE VIDEO OUTPUT
# ==========================================

def prepare_video_output(video_path):

    """
    Validate a processed video before displaying it
    through a Gradio Video output component.
    """

    video_path = resolve_video_path(video_path)

    if video_path is None:
        return None

    return video_path


# ==========================================
# 6. CREATE EMPTY VIDEO STATE
# ==========================================

def empty_video_state():

    return {
        "video_path": None,
        "is_processing": False,
        "current_frame": 0,
        "total_frames": 0,
        "fps": 0,
        "duration": 0,
        "metadata": None
    }
