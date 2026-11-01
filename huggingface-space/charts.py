
import plotly.graph_objects as go


# ==========================================
# COLORS
# ==========================================

COLORS = {
    "Walking": "#00d9ff",
    "Standing": "#00e6a0",
    "Lifting": "#ffaa00",
    "Fall": "#ff2870",
}


# ==========================================
# COMMON CHART STYLE
# ==========================================

def apply_chart_style(fig, title):

    fig.update_layout(
        title={
            "text": title,
            "font": {
                "size": 16,
                "color": "#e8f2ff"
            }
        },

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        font={
            "color": "#91a9ce",
            "family": "Arial"
        },

        margin=dict(
            l=35,
            r=15,
            t=55,
            b=35
        ),

        legend=dict(
            orientation="h",
            y=-0.25,
            font=dict(size=11)
        ),

        hovermode="x unified",

        xaxis=dict(
            showgrid=False,
            zeroline=False,
            color="#91a9ce"
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(50,100,180,0.15)",
            zeroline=False,
            color="#91a9ce"
        ),

        height=280
    )

    return fig


# ==========================================
# ACTIONS OVER TIME
# ==========================================

def create_activity_chart(
    timestamps=None,
    walking=None,
    standing=None,
    lifting=None,
    falling=None
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
        "Fall": falling
    }

    for action, values in datasets.items():

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                name=action,
                mode="lines",
                line=dict(
                    color=COLORS[action],
                    width=2
                ),
                fill="tozeroy",
                fillcolor=COLORS[action] + "12"
            )
        )

    fig.update_layout(
        yaxis_title="Detected Actions",
        xaxis_title="Time"
    )

    return apply_chart_style(
        fig,
        "ACTIONS OVER TIME"
    )


# ==========================================
# ACTION DISTRIBUTION
# ==========================================

def create_distribution_chart(
    counts=None
):

    counts = counts or {}

    labels = list(counts.keys())
    values = list(counts.values())

    colors = [
        COLORS.get(name, "#087bff")
        for name in labels
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.65,

                marker=dict(
                    colors=colors,
                    line=dict(
                        color="#07132d",
                        width=3
                    )
                ),

                textinfo="label+percent",
                textfont=dict(
                    color="#e8f2ff",
                    size=11
                ),

                hovertemplate=(
                    "%{label}<br>"
                    "Count: %{value}<br>"
                    "%{percent}"
                    "<extra></extra>"
                )
            )
        ]
    )

    fig.update_layout(
        showlegend=True,
        height=280
    )

    fig = apply_chart_style(
        fig,
        "ACTION DISTRIBUTION"
    )

    return fig


# ==========================================
# EMPTY CHART
# ==========================================

def empty_chart():

    return create_activity_chart(
        timestamps=[],
        walking=[],
        standing=[],
        lifting=[],
        falling=[]
    )
