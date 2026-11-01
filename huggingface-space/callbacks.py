from video_panel import (
    resolve_video_path,
    inspect_video
)

from charts import create_activity_chart


# ==========================================
# 1. EMPTY ANALYSIS RESULT
# ==========================================

def empty_result():

    return {
        "status": "Ready",

        "people": 0,
        "actions": 0,
        "falls": 0,
        "collisions": 0,

        "walking": 0,
        "standing": 0,
        "lifting": 0,
        "falling": 0,

        "timestamps": [],

        "walking_history": [],
        "standing_history": [],
        "lifting_history": [],
        "falling_history": [],

        "events": []
    }


# ==========================================
# 2. NORMALIZE AI RESULT
# ==========================================

def normalize_result(result):

    """
    Ensure that AI output follows the dashboard schema.
    """

    if not isinstance(result, dict):
        raise ValueError("AI processor must return a dictionary")

    data = empty_result()
    data.update(result)

    numeric_fields = [
        "people",
        "actions",
        "falls",
        "collisions",
        "walking",
        "standing",
        "lifting",
        "falling"
    ]

    for field in numeric_fields:

        try:
            data[field] = float(data[field])
        except (ValueError, TypeError):
            data[field] = 0

    percentage_fields = [
        "walking",
        "standing",
        "lifting",
        "falling"
    ]

    for field in percentage_fields:
        data[field] = max(0, min(100, data[field]))

    if not isinstance(data["events"], list):
        data["events"] = []

    return data


# ==========================================
# 3. PREPARE CHART
# ==========================================

def prepare_chart(data):

    return create_activity_chart(
        timestamps=data["timestamps"],

        walking=data["walking_history"],
        standing=data["standing_history"],
        lifting=data["lifting_history"],
        falling=data["falling_history"]
    )


# ==========================================
# 4. PREPARE EVENTS TABLE
# ==========================================

def prepare_events(data):

    rows = []

    for event in data["events"]:

        if not isinstance(event, dict):
            continue

        action = str(
            event.get("action", "Unknown")
        )

        confidence = event.get(
            "confidence", 0
        )

        timestamp = str(
            event.get("time", "--:--:--")
        )

        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0

        rows.append([
            action,
            f"{confidence:.2f}",
            timestamp
        ])

    return rows


# ==========================================
# 5. BUILD DASHBOARD OUTPUTS
# ==========================================

def build_dashboard_outputs(result):

    """
    Convert AI results into Gradio component outputs.

    Output order must match main.py bindings.
    """

    data = normalize_result(result)

    return (

        data["status"],

        data["people"],
        data["actions"],
        data["falls"],
        data["collisions"],

        data["walking"],
        data["standing"],
        data["lifting"],
        data["falling"],

        prepare_chart(data),

        prepare_events(data)
    )


# ==========================================
# 6. RUN ANALYSIS
# ==========================================

def run_analysis(
    video_input,
    threshold,
    model,
    device,
    processor=None
):

    """
    Connect uploaded video to an external AI processor.

    processor must be supplied by main.py.
    """

    video_path = resolve_video_path(video_input)

    if video_path is None:

        result = empty_result()
        result["status"] = "Please upload a valid video"

        return build_dashboard_outputs(result)

    metadata = inspect_video(video_path)

    if metadata is None:

        result = empty_result()
        result["status"] = "Unable to read video"

        return build_dashboard_outputs(result)

    if processor is None:

        result = empty_result()
        result["status"] = "AI processor not connected"

        return build_dashboard_outputs(result)

    try:

        # Send video information to AI engine.

        ai_result = processor(
            video_path=video_path,
            threshold=threshold,
            model=model,
            device=device
        )

        result = normalize_result(ai_result)

        result["status"] = "Analysis completed"

        return build_dashboard_outputs(result)

    except Exception as error:

        result = empty_result()

        result["status"] = (
            f"Analysis error: {str(error)}"
        )

        return build_dashboard_outputs(result)


# ==========================================
# 7. RESET DASHBOARD
# ==========================================

def reset_dashboard():

    result = empty_result()

    return build_dashboard_outputs(result)
    