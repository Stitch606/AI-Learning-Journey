
import html


# ==========================================
# HELPERS
# ==========================================

def safe(value):
    """Protect HTML from unsafe dynamic text."""
    return html.escape(str(value))


# ==========================================
# STATISTICS CARDS
# ==========================================

def statistics_html(
    people=0,
    actions=0,
    falls=0,
    collisions=0
):

    cards = [
        ("👤", people, "Total People", "stat-blue"),
        ("🏃", actions, "Actions Detected", "stat-purple"),
        ("⚠️", falls, "Fall Detected", "stat-red"),
        ("🛡️", collisions, "Collisions", "stat-green"),
    ]

    result = '<div class="stats-grid">'

    for icon, value, label, color in cards:

        result += f"""
        <div class="stat-card {color}">

            <div class="stat-icon">{icon}</div>

            <div>
                <div class="stat-value">{safe(value)}</div>
                <div class="stat-label">{safe(label)}</div>
            </div>

        </div>
        """

    result += "</div>"

    return result


# ==========================================
# DETECTED ACTIONS
# ==========================================

def actions_html(
    walking=0,
    standing=0,
    lifting=0,
    falling=0
):

    actions = [
        ("🚶", "Walking", walking, "progress-blue"),
        ("🧍", "Standing", standing, "progress-green"),
        ("🏋️", "Lifting", lifting, "progress-orange"),
        ("🏃", "Fall", falling, "progress-red"),
    ]

    result = '<div class="action-list">'

    for icon, name, percentage, color in actions:

        percentage = max(0, min(100, float(percentage)))

        result += f"""
        <div class="action-item">

            <div class="action-icon">{icon}</div>

            <div class="action-info">

                <div class="action-info-header">
                    <span>{name}</span>
                    <span>{percentage:.1f}%</span>
                </div>

                <div class="progress-track">
                    <div
                        class="progress-fill {color}"
                        style="width:{percentage}%"
                    ></div>
                </div>

            </div>

        </div>
        """

    result += "</div>"

    return result


# ==========================================
# RECENT EVENTS
# ==========================================

def events_html(events=None):

    if events is None:
        events = []

    if not events:

        return """
        <div class="event-list">
            <div style="
                text-align:center;
                padding:25px;
                color:#91a9ce;
            ">
                No events detected yet.
            </div>
        </div>
        """

    result = '<div class="event-list">'

    event_styles = {
        "Fall": ("🚨", "#ff2870"),
        "Walking": ("🚶", "#00d9ff"),
        "Standing": ("🧍", "#00e6a0"),
        "Lifting": ("🏋️", "#ffaa00"),
        "Collision": ("⚠️", "#ff2870"),
    }

    for event in events:

        # Each event is a dictionary.
        name = safe(event.get("action", "Unknown"))
        confidence = event.get("confidence", 0)
        timestamp = safe(event.get("time", "--:--:--"))

        try:
            confidence = max(0, min(1, float(confidence)))
        except (ValueError, TypeError):
            confidence = 0

        icon, color = event_styles.get(
            name,
            ("📌", "#00d9ff")
        )

        result += f"""
        <div class="event-item">

            <div style="
                display:flex;
                align-items:center;
                gap:12px;
                min-width:0;
            ">

                <div style="
                    width:38px;
                    height:38px;
                    flex-shrink:0;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    border:1px solid {color};
                    border-radius:50%;
                    color:{color};
                ">
                    {icon}
                </div>

                <div>
                    <div class="event-name"
                         style="color:{color}">
                        {name}
                    </div>

                    <div class="event-time">
                        Conf: {confidence:.2f}
                    </div>
                </div>

            </div>

            <div class="event-time">
                {timestamp}
            </div>

        </div>
        """

    result += "</div>"

    return result


# ==========================================
# SYSTEM STATUS
# ==========================================

def status_html(
    status="Ready",
    online=True
):

    color = "#00e6a0" if online else "#ff2870"
    indicator = "●" if online else "●"

    return f"""
    <div style="
        display:flex;
        align-items:center;
        gap:8px;
        color:{color};
        font-size:13px;
        font-weight:600;
    ">
        <span>{indicator}</span>
        <span>{safe(status)}</span>
    </div>
    """


# ==========================================
# VIDEO INFORMATION
# ==========================================

def video_info_html(
    filename="No video selected",
    resolution="--",
    fps=0
):

    return f"""
    <div style="
        display:flex;
        align-items:center;
        justify-content:space-between;
        flex-wrap:wrap;
        gap:10px;
        color:#dceaff;
        font-size:13px;
    ">

        <div>
            🎥 {safe(filename)}
        </div>

        <div style="
            display:flex;
            gap:8px;
        ">

            <span style="
                background:#091a3b;
                border:1px solid #164b91;
                padding:6px 10px;
                border-radius:20px;
            ">
                {safe(resolution)}
            </span>

            <span style="
                background:#091a3b;
                border:1px solid #164b91;
                padding:6px 10px;
                border-radius:20px;
            ">
                {safe(fps)} FPS
            </span>

        </div>

    </div>
    """
