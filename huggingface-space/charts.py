import plotly.graph_objects as go


COLORS = {
    "Walking": "#55E9FF",
    "Standing": "#B52CFF",
    "Lifting": "#FF4FD8",
    "Fall": "#FF5D91",
}


def apply_chart_style(fig, title):
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": "#F7E9FF"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#CDB8DE", "family": "DM Sans, Arial"},
        margin=dict(l=35, r=15, t=55, b=35),
        legend=dict(
            orientation="h",
            y=-0.25,
            font=dict(size=11, color="#EAD8F8"),
        ),
        hovermode="x unified",
        xaxis=dict(showgrid=False, zeroline=False, color="#CDB8DE"),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(181,44,255,0.15)",
            zeroline=False,
            color="#CDB8DE",
        ),
        height=280,
    )
    return fig


def create_activity_chart(
    timestamps=None,
    walking=None,
    standing=None,
    lifting=None,
    falling=None,
):
    timestamps = timestamps or []
    walking = walking or []
    standing = standing or []
    lifting = lifting or []
    falling = falling or []

    fig = go.Figure()
    datasets = {
        "Walking": walking,
        "Standing": standing,
        "Lifting": lifting,
        "Fall": falling,
    }

    for action, values in datasets.items():
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                name=action,
                mode="lines",
                line=dict(color=COLORS[action], width=2.5),
                fill="tozeroy",
                fillcolor=COLORS[action] + "12",
            )
        )

    fig.update_layout(yaxis_title="Detected Actions", xaxis_title="Time")
    return apply_chart_style(fig, "ACTIONS OVER TIME")


def create_distribution_chart(counts=None):
    counts = counts or {}
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [COLORS.get(name, "#B52CFF") for name in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.65,
                marker=dict(
                    colors=colors,
                    line=dict(color="#160D24", width=3),
                ),
                textinfo="label+percent",
                textfont=dict(color="#F7E9FF", size=11),
                hovertemplate="%{label}<br>Count: %{value}<br>%{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(showlegend=True, height=280)
    return apply_chart_style(fig, "ACTION DISTRIBUTION")


def empty_chart():
    return create_activity_chart(
        timestamps=[],
        walking=[],
        standing=[],
        lifting=[],
        falling=[],
    )
