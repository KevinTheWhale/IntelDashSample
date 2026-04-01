import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── Load & Prep ───────────────────────────────────────────────────────────────
df_raw = pd.read_csv("uci-secom.csv")

label_col = df_raw.columns[-1]
df_raw = df_raw.rename(columns={label_col: "label"})

# Separate features from metadata to avoid fragmentation
feature_cols = [c for c in df_raw.columns if c != "label"]
numeric_df = df_raw[feature_cols].apply(pd.to_numeric, errors="coerce")

# Build clean master df via concat — no fragmentation
df = pd.concat([
    numeric_df,
    df_raw["label"].reset_index(drop=True),
    df_raw["label"].map({-1: "Pass", 1: "Fail"}).rename("result").reset_index(drop=True),
    pd.Series(range(len(df_raw)), name="time_index"),
], axis=1)

# Null rates across all features
null_rates = (numeric_df.isnull().sum() / len(numeric_df) * 100).sort_values(ascending=False)

# For null rate chart: top 20 features with highest null rate
top_null = null_rates.head(20)

# For dropdowns + control chart: top 20 most complete features (lowest null rate, non-zero variance)
complete_features = null_rates.sort_values(ascending=True).index.tolist()
# Filter to features that have actual variance
valid_features = [
    f for f in complete_features
    if numeric_df[f].std() > 0 and numeric_df[f].notna().sum() > 100
][:20]

# Fallback if filtering too aggressive
if len(valid_features) < 5:
    valid_features = complete_features[:20]

# Filled df for control chart + distribution
filled_data = numeric_df[valid_features].apply(lambda col: col.fillna(col.median()))
filled = pd.concat([
    filled_data.reset_index(drop=True),
    df["result"].reset_index(drop=True),
    df["time_index"].reset_index(drop=True),
], axis=1)

# KPIs
total      = len(df)
fail_count = int((df["label"] == 1).sum())
pass_count = int((df["label"] == -1).sum())
yield_rate = round(pass_count / total * 100, 2)

# ── App Init ──────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "SECOM Process Intelligence"

# ── Color Palette ─────────────────────────────────────────────────────────────
BG       = "#0d1117"
CARD     = "#161b22"
BORDER   = "#30363d"
ACCENT   = "#58a6ff"
PASS_CLR = "#3fb950"
FAIL_CLR = "#f85149"
TEXT     = "#e6edf3"
SUBTEXT  = "#8b949e"

def kpi_card(title, value, subtitle, color=ACCENT):
    return html.Div([
        html.P(title, style={"color": SUBTEXT, "fontSize": "11px",
                             "textTransform": "uppercase", "letterSpacing": "1.5px",
                             "marginBottom": "6px"}),
        html.H2(str(value), style={"color": color, "fontSize": "36px",
                                   "fontWeight": "700", "margin": "0",
                                   "fontFamily": "monospace"}),
        html.P(subtitle, style={"color": SUBTEXT, "fontSize": "12px", "marginTop": "4px"}),
    ], style={
        "background": CARD, "border": f"1px solid {BORDER}",
        "borderTop": f"3px solid {color}", "borderRadius": "8px",
        "padding": "20px 24px", "flex": "1", "minWidth": "160px"
    })

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(style={
    "background": BG, "minHeight": "100vh",
    "fontFamily": "'Segoe UI', sans-serif", "color": TEXT,
    "padding": "32px 40px"
}, children=[

    # Header
    html.Div([
        html.Div([
            html.H1("SECOM Process Intelligence",
                    style={"margin": "0", "fontSize": "22px", "fontWeight": "600",
                           "color": TEXT, "letterSpacing": "0.5px"}),
            html.P("Semiconductor Manufacturing · Yield & Process Analytics",
                   style={"margin": "4px 0 0", "color": SUBTEXT, "fontSize": "13px"}),
        ]),
        html.Span("● LIVE", style={"color": PASS_CLR, "fontSize": "12px",
                                   "fontWeight": "600", "letterSpacing": "1px",
                                   "alignSelf": "center"})
    ], style={"display": "flex", "justifyContent": "space-between",
              "marginBottom": "28px", "paddingBottom": "20px",
              "borderBottom": f"1px solid {BORDER}"}),

    # KPI Row
    html.Div([
        kpi_card("Total Wafers",  f"{total:,}",      "Production run"),
        kpi_card("Yield Rate",    f"{yield_rate}%",  "Pass / Total",           PASS_CLR),
        kpi_card("Pass",          f"{pass_count:,}", "Label: −1",              PASS_CLR),
        kpi_card("Fail",          f"{fail_count:,}", "Label: +1",              FAIL_CLR),
        kpi_card("Features",      "591",             "Process parameters"),
        kpi_card("Avg Null Rate", f"{round(null_rates.mean(), 1)}%",
                 "Across all features", "#d29922"),
    ], style={"display": "flex", "gap": "16px", "marginBottom": "24px", "flexWrap": "wrap"}),

    # Row 1: Yield Trend + Donut
    html.Div([
        html.Div([
            html.P("Yield Trend (Rolling 50-Wafer Window)",
                   style={"color": SUBTEXT, "fontSize": "12px", "marginBottom": "8px",
                          "textTransform": "uppercase", "letterSpacing": "1px"}),
            dcc.Graph(id="yield-trend", config={"displayModeBar": False})
        ], style={"background": CARD, "border": f"1px solid {BORDER}",
                  "borderRadius": "8px", "padding": "20px", "flex": "2"}),

        html.Div([
            html.P("Pass / Fail Distribution",
                   style={"color": SUBTEXT, "fontSize": "12px", "marginBottom": "8px",
                          "textTransform": "uppercase", "letterSpacing": "1px"}),
            dcc.Graph(id="donut-chart", config={"displayModeBar": False})
        ], style={"background": CARD, "border": f"1px solid {BORDER}",
                  "borderRadius": "8px", "padding": "20px", "flex": "1"}),
    ], style={"display": "flex", "gap": "16px", "marginBottom": "24px"}),

    # Row 2: Control Chart + Null Rate
    html.Div([
        html.Div([
            html.Div([
                html.P("Process Control Chart",
                       style={"color": SUBTEXT, "fontSize": "12px",
                              "textTransform": "uppercase", "letterSpacing": "1px",
                              "margin": "0"}),
                dcc.Dropdown(
                            id="feature-select",
                            options=[{"label": f, "value": f} for f in valid_features],
                            value=valid_features[0],
                            clearable=False,
                            style={"width": "260px", "fontSize": "13px", "color": "#000000"}
                            )
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "center", "marginBottom": "12px"}),
            dcc.Graph(id="control-chart", config={"displayModeBar": False})
        ], style={"background": CARD, "border": f"1px solid {BORDER}",
                  "borderRadius": "8px", "padding": "20px", "flex": "2"}),

        html.Div([
            html.P("Top 20 Features by Null Rate",
                   style={"color": SUBTEXT, "fontSize": "12px", "marginBottom": "8px",
                          "textTransform": "uppercase", "letterSpacing": "1px"}),
            dcc.Graph(id="null-rate-chart", config={"displayModeBar": False})
        ], style={"background": CARD, "border": f"1px solid {BORDER}",
                  "borderRadius": "8px", "padding": "20px", "flex": "1"}),
    ], style={"display": "flex", "gap": "16px", "marginBottom": "24px"}),

    # Row 3: Feature Distribution
    html.Div([
        html.Div([
            html.Div([
                html.P("Feature Distribution by Outcome",
                       style={"color": SUBTEXT, "fontSize": "12px",
                              "textTransform": "uppercase", "letterSpacing": "1px",
                              "margin": "0"}),
                dcc.Dropdown(
                        id="dist-feature-select",
                        options=[{"label": f, "value": f} for f in valid_features],
                        value=valid_features[1] if len(valid_features) > 1 else valid_features[0],
                        clearable=False,
                        style={"width": "260px", "fontSize": "13px", "color": "#000000"}
                            )
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "center", "marginBottom": "12px"}),
            dcc.Graph(id="dist-chart", config={"displayModeBar": False})
        ], style={"background": CARD, "border": f"1px solid {BORDER}",
                  "borderRadius": "8px", "padding": "20px", "flex": "1"}),
    ], style={"display": "flex", "gap": "16px"}),

    # Footer
    html.Div(
        html.P("SECOM UCI Dataset · Semiconductor Process Intelligence Dashboard · K",
               style={"color": SUBTEXT, "fontSize": "11px",
                      "textAlign": "center", "marginTop": "32px"})
    )
])

# ── Shared plot layout ────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, size=11),
    margin=dict(l=40, r=20, t=20, b=40),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
)

# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(Output("yield-trend", "figure"), Input("feature-select", "value"))
def update_yield_trend(_):
    roll = (df["label"] == -1).astype(int).rolling(50).mean() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_index"], y=roll, mode="lines", name="Rolling Yield",
        line=dict(color=PASS_CLR, width=2),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.08)"
    ))
    fig.add_hline(y=yield_rate, line_dash="dash", line_color=ACCENT,
                  annotation_text=f"Avg {yield_rate}%", annotation_font_color=ACCENT)
    fig.update_layout(**PLOT_LAYOUT, height=260, yaxis_range=[0, 105],
                      yaxis_title="Yield %", xaxis_title="Wafer Index")
    return fig

@app.callback(Output("donut-chart", "figure"), Input("feature-select", "value"))
def update_donut(_):
    fig = go.Figure(go.Pie(
        labels=["Pass", "Fail"], values=[pass_count, fail_count],
        hole=0.65,
        marker=dict(colors=[PASS_CLR, FAIL_CLR], line=dict(color=BG, width=3)),
        textinfo="percent", textfont=dict(size=13, color=TEXT),
    ))
    fig.update_layout(**PLOT_LAYOUT, height=260, showlegend=True,
                      legend=dict(orientation="h", y=-0.1),
                      annotations=[dict(text=f"{yield_rate}%<br>Yield",
                                        x=0.5, y=0.5, font_size=16,
                                        font_color=PASS_CLR, showarrow=False)])
    return fig

@app.callback(Output("control-chart", "figure"), Input("feature-select", "value"))
def update_control_chart(feature):
    series = filled[feature]
    mean = series.mean()
    std  = series.std()
    ucl  = mean + 3 * std
    lcl  = mean - 3 * std
    ooc  = (series > ucl) | (series < lcl)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filled["time_index"], y=series, mode="lines",
                             name=feature, line=dict(color=ACCENT, width=1)))
    fig.add_trace(go.Scatter(x=filled["time_index"][ooc], y=series[ooc],
                             mode="markers", name="Out of Control",
                             marker=dict(color=FAIL_CLR, size=6)))
    fig.add_hline(y=mean, line_color=PASS_CLR, line_dash="dot",
                  annotation_text="Mean", annotation_font_color=PASS_CLR)
    fig.add_hline(y=ucl, line_color=FAIL_CLR, line_dash="dash",
                  annotation_text="UCL +3σ", annotation_font_color=FAIL_CLR)
    fig.add_hline(y=lcl, line_color=FAIL_CLR, line_dash="dash",
                  annotation_text="LCL −3σ", annotation_font_color=FAIL_CLR)
    fig.update_layout(**PLOT_LAYOUT, height=260,
                      xaxis_title="Wafer Index", yaxis_title=feature,
                      legend=dict(orientation="h", y=1.1))
    return fig

@app.callback(Output("null-rate-chart", "figure"), Input("feature-select", "value"))
def update_null_rate(_):
    data = top_null[top_null > 0]
    colors = [FAIL_CLR if v > 50 else "#d29922" if v > 20 else PASS_CLR for v in data.values]
    fig = go.Figure(go.Bar(
        x=data.values,
        y=data.index,
        orientation="h",
        marker=dict(color=colors)
    ))
    fig.update_layout(**PLOT_LAYOUT, height=260)
    fig.update_xaxes(title_text="Null Rate (%)", range=[0, 105])
    fig.update_yaxes(tickfont=dict(size=9))
    return fig

@app.callback(Output("dist-chart", "figure"), Input("dist-feature-select", "value"))
def update_dist(feature):
    pass_vals = filled[filled["result"] == "Pass"][feature].dropna()
    fail_vals = filled[filled["result"] == "Fail"][feature].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pass_vals, name="Pass",
                               marker_color=PASS_CLR, opacity=0.7, nbinsx=40))
    fig.add_trace(go.Histogram(x=fail_vals, name="Fail",
                               marker_color=FAIL_CLR, opacity=0.7, nbinsx=40))
    fig.update_layout(**PLOT_LAYOUT, height=260, barmode="overlay",
                      xaxis_title=feature, yaxis_title="Count",
                      legend=dict(orientation="h", y=1.1))
    return fig

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)