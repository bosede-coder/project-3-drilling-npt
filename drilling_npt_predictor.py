"""
Drilling Non-Productive Time (NPT) Predictor
AI-Powered Drilling Risk Detection for Oil and Gas
Author: Bose Abubakre, PhD MBA
Portfolio: Oil and Gas AI | github.com/boseabubakre

Data: Synthetic drilling parameters modeled after
      BOEM Gulf of Mexico offshore drilling reports
      and publicly available IADC drilling datasets

Features: WOB, ROP, RPM, TORQUE, SPP, FLOW_IN, ECD, HOOKLOAD, TEMP
Target: NPT events - Stuck Pipe, Lost Circulation, Kick, Washout, Normal

Run: streamlit run drilling_npt_predictor.py
Install: pip install streamlit pandas numpy plotly scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drilling NPT Predictor",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #ffffff;
    color: #111827;
  }
  .main, .stApp { background-color: #ffffff; }
  [data-testid="metric-container"] {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 12px 16px;
  }
  [data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #ffffff;
    letter-spacing: 0.06em;
  }
  [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 24px;
    color: #f0a500;
    font-weight: 500;
  }
  [data-testid="stSidebar"] {
    background-color: #f0f4ff;
    border-right: 1px solid #3a7bd5;
  }
  h2 { color: #111827; font-size: 15px !important;
       border-bottom: 1px solid #bfdbfe; padding-bottom: 6px; }
  hr { border: none; border-top: 1px solid #bfdbfe; margin: 8px 0; }
  .info-box {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-left: 3px solid #3b82f6; border-radius: 4px;
    padding: 10px 14px; font-size: 13px; color: #1e3a8a; margin: 8px 0;
  }
  .risk-critical {
    background: #4a0a0a; border: 2px solid #f85149;
    border-radius: 8px; padding: 12px; text-align: center;
  }
  .risk-warning {
    background: #3a2800; border: 2px solid #f0a500;
    border-radius: 8px; padding: 12px; text-align: center;
  }
  .risk-normal {
    background: #0a3a1a; border: 2px solid #3fb950;
    border-radius: 8px; padding: 12px; text-align: center;
  }
  #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
EVENT_COLORS = {
    "Normal":           "#3fb950",
    "Stuck Pipe":       "#f85149",
    "Lost Circulation": "#f0a500",
    "Kick":             "#bc8cff",
    "Washout":          "#58a6ff",
}

EVENT_COST = {
    "Normal":           0,
    "Stuck Pipe":       450000,
    "Lost Circulation": 180000,
    "Kick":             320000,
    "Washout":          95000,
}

EVENT_DESCRIPTIONS = {
    "Stuck Pipe":       "Drillstring unable to rotate or move. Average cost $450K/event.",
    "Lost Circulation": "Drilling fluid lost to formation. Average cost $180K/event.",
    "Kick":             "Formation fluid influx into wellbore. Average cost $320K/event.",
    "Washout":          "Bit or string erosion from fluid flow. Average cost $95K/event.",
    "Normal":           "Drilling within normal operating parameters.",
}

# ── DATA GENERATION ───────────────────────────────────────────────────────────
@st.cache_data
def generate_drilling_data(n_samples=8000, seed=42):
    """
    Synthetic drilling data modeled after BOEM GOM offshore wells.
    Realistic parameter ranges for 8.5-inch hole section, 8-12 ppg mud weight.

    In production: download BOEM drilling reports from
    https://www.boem.gov/oil-gas-energy/drilling
    """
    np.random.seed(seed)
    records = []
    depths = np.linspace(2000, 12000, n_samples)

    for i, depth in enumerate(depths):
        depth_factor = (depth - 2000) / 10000
        event_roll = np.random.random()

        # Normal drilling baseline (85% of time)
        wob       = np.random.normal(25, 5)
        rop       = np.random.normal(80 - 30 * depth_factor, 15)
        rpm       = np.random.normal(120, 15)
        torque    = np.random.normal(8000 + 2000 * depth_factor, 800)
        spp       = np.random.normal(2800 + 1500 * depth_factor, 200)
        flow_in   = np.random.normal(650, 30)
        ecd       = np.random.normal(9.2 + 1.5 * depth_factor, 0.2)
        hookload  = np.random.normal(200 + 50 * depth_factor, 15)
        temp      = np.random.normal(80 + 120 * depth_factor, 10)
        event     = "Normal"

        # Stuck pipe (6%): high torque, low ROP, variable WOB
        if event_roll > 0.94:
            torque   *= np.random.uniform(1.8, 3.5)
            rop      *= np.random.uniform(0.0, 0.15)
            wob      += np.random.uniform(10, 25)
            hookload *= np.random.uniform(1.2, 1.6)
            event     = "Stuck Pipe"

        # Lost circulation (4%): sudden drop in flow, low SPP
        elif event_roll > 0.90:
            flow_in  *= np.random.uniform(0.3, 0.7)
            spp      *= np.random.uniform(0.4, 0.7)
            ecd      -= np.random.uniform(0.3, 0.8)
            event     = "Lost Circulation"

        # Kick (3%): increased flow out, pit gain, low ECD
        elif event_roll > 0.87:
            flow_in  *= np.random.uniform(1.2, 1.6)
            ecd      -= np.random.uniform(0.4, 1.0)
            spp      *= np.random.uniform(1.1, 1.4)
            event     = "Kick"

        # Washout (2%): high flow, low torque, low ROP
        elif event_roll > 0.85:
            flow_in  *= np.random.uniform(1.1, 1.3)
            torque   *= np.random.uniform(0.5, 0.7)
            rop      *= np.random.uniform(0.2, 0.5)
            event     = "Washout"

        records.append({
            "DEPTH_FT":  round(depth, 1),
            "WOB":       round(float(np.clip(wob, 0, 60)), 2),
            "ROP":       round(float(np.clip(rop, 0, 200)), 2),
            "RPM":       round(float(np.clip(rpm, 40, 200)), 1),
            "TORQUE":    round(float(np.clip(torque, 500, 35000)), 0),
            "SPP":       round(float(np.clip(spp, 500, 5500)), 0),
            "FLOW_IN":   round(float(np.clip(flow_in, 100, 900)), 1),
            "ECD":       round(float(np.clip(ecd, 7.5, 13.0)), 3),
            "HOOKLOAD":  round(float(np.clip(hookload, 50, 450)), 1),
            "TEMP":      round(float(np.clip(temp, 60, 350)), 1),
            "EVENT":     event,
        })

    return pd.DataFrame(records)

@st.cache_resource
def train_npt_model(df):
    features = ["WOB","ROP","RPM","TORQUE","SPP","FLOW_IN","ECD","HOOKLOAD","TEMP"]
    X = df[features]
    y = df["EVENT"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.1, random_state=42, subsample=0.8)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test.values).mean()
    importance = pd.DataFrame({
        "Feature":    features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
    return model, X_test, y_test, y_pred, accuracy, importance

@st.cache_data
def generate_realtime_stream(seed=7):
    """Simulate real-time drilling feed for the live monitoring panel."""
    np.random.seed(seed)
    n = 200
    depths = np.linspace(8500, 9000, n)
    records = []
    for i, depth in enumerate(depths):
        df_factor = (depth - 2000) / 10000
        # Inject a stuck pipe event around point 120
        if 115 < i < 140:
            wob      = np.random.normal(45, 3)
            rop      = np.random.normal(2, 1)
            torque   = np.random.normal(24000, 1500)
            hookload = np.random.normal(340, 10)
        else:
            wob      = np.random.normal(25, 3)
            rop      = np.random.normal(55, 8)
            torque   = np.random.normal(10000, 600)
            hookload = np.random.normal(250, 10)
        spp     = np.random.normal(3200, 150)
        flow_in = np.random.normal(650, 20)
        ecd     = np.random.normal(9.5, 0.15)
        rpm     = np.random.normal(118, 8)
        temp    = np.random.normal(180 + 50 * df_factor, 5)

        row = {
            "DEPTH_FT": round(depth, 1),
            "WOB":      float(np.clip(wob, 0, 60)),
            "ROP":      float(np.clip(rop, 0, 200)),
            "RPM":      float(np.clip(rpm, 40, 200)),
            "TORQUE":   float(np.clip(torque, 500, 35000)),
            "SPP":      float(np.clip(spp, 500, 5500)),
            "FLOW_IN":  float(np.clip(flow_in, 100, 900)),
            "ECD":      float(np.clip(ecd, 7.5, 13.0)),
            "HOOKLOAD": float(np.clip(hookload, 50, 450)),
            "TEMP":     float(np.clip(temp, 60, 350)),
        }
        records.append(row)

    return pd.DataFrame(records)

def chart_layout(**overrides):
    base = dict(
        paper_bgcolor="#f0f4ff",
        plot_bgcolor="#ffffff",
        font=dict(family="IBM Plex Mono", color="#1e3a8a", size=10),
        xaxis=dict(gridcolor="#dbeafe", linecolor="#dbeafe"),
        yaxis=dict(gridcolor="#dbeafe", linecolor="#dbeafe"),
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(bgcolor="#f0f4ff", bordercolor="#dbeafe", borderwidth=1)
    )
    base.update(overrides)
    return base

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with st.spinner("Loading drilling data and training NPT detection model..."):
    df = generate_drilling_data()
    model, X_test, y_test, y_pred, accuracy, importance = train_npt_model(df)
    rt_df = generate_realtime_stream()
    # Apply model predictions outside cached function (model not hashable)
    features = ["WOB","ROP","RPM","TORQUE","SPP","FLOW_IN","ECD","HOOKLOAD","TEMP"]
    rt_df["PREDICTED_EVENT"] = model.predict(rt_df[features])
    proba = model.predict_proba(rt_df[features])
    for j, cls in enumerate(model.classes_):
        rt_df[f"PROB_{cls.replace(' ','_')}"] = proba[:, j]
    rt_df["MAX_RISK"] = 1 - rt_df["PROB_Normal"]

# ── HEADER ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([4, 2])
with c1:
    st.markdown("## 🛢️ Drilling NPT Early Warning System")
    st.markdown(
        '<div class="info-box">Real-time anomaly detection for drilling non-productive time events. '
        'Trained on synthetic parameters modeled after <b>BOEM Gulf of Mexico</b> offshore drilling '
        'reports. Detects Stuck Pipe, Lost Circulation, Kick, and Washout events before they escalate. '
        'Connect to real WITSML or CSV drilling feeds for production deployment.</div>',
        unsafe_allow_html=True)
with c2:
    st.markdown('<div style="margin-top:12px"></div>', unsafe_allow_html=True)
    st.metric("Model Accuracy", f"{accuracy:.1%}")

# ── KPI STRIP ─────────────────────────────────────────────────────────────────
st.markdown("#### Current Well Status — 8.5-in Hole Section")
npt_events = df[df["EVENT"] != "Normal"]
total_npt = len(npt_events)
npt_pct = total_npt / len(df) * 100
est_cost = sum(EVENT_COST[e] for e in npt_events["EVENT"])
stuck_n = (df["EVENT"] == "Stuck Pipe").sum()
kick_n  = (df["EVENT"] == "Kick").sum()
circ_n  = (df["EVENT"] == "Lost Circulation").sum()

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("NPT Rate", f"{npt_pct:.1f}%", delta="Industry avg 15%")
with k2:
    st.metric("Est. NPT Cost", f"${est_cost/1e6:.1f}M", delta="Per well section")
with k3:
    st.metric("Stuck Pipe Events", str(stuck_n), delta="Highest cost NPT")
with k4:
    st.metric("Kick Events", str(kick_n), delta="Safety critical")
with k5:
    st.metric("Lost Circ Events", str(circ_n), delta="Fluid losses")

st.markdown("---")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Well Controls")
    depth_range = st.slider(
        "Depth Range (ft)",
        float(df["DEPTH_FT"].min()), float(df["DEPTH_FT"].max()),
        (4000.0, 8000.0)
    )
    show_events = st.multiselect(
        "Show Events",
        list(EVENT_COLORS.keys()),
        default=list(EVENT_COLORS.keys())
    )
    st.markdown("---")
    st.markdown("### Real-Time Risk Assessment")
    st.markdown('<div style="font-family:IBM Plex Mono;font-size:11px;color:#1e3a8a">Enter current drilling parameters:</div>',
                unsafe_allow_html=True)
    wob_in      = st.slider("WOB (klbs)",      0.0, 60.0,  25.0)
    rop_in      = st.slider("ROP (ft/hr)",     0.0, 200.0, 65.0)
    rpm_in      = st.slider("RPM",             40.0,200.0, 120.0)
    torque_in   = st.slider("Torque (ft·lbs)", 500.0,35000.0, 8500.0, step=100.0)
    spp_in      = st.slider("SPP (psi)",       500.0,5500.0, 2900.0, step=50.0)
    flow_in_v   = st.slider("Flow In (gpm)",   100.0,900.0, 650.0)
    ecd_in      = st.slider("ECD (ppg)",       7.5,  13.0,  9.3)
    hookload_in = st.slider("Hookload (klbs)", 50.0, 450.0, 210.0)
    temp_in     = st.slider("Temp (°F)",       60.0, 350.0, 185.0)

    if st.button("⚡ Assess Risk Now", type="primary"):
        inp = [[wob_in, rop_in, rpm_in, torque_in, spp_in,
                flow_in_v, ecd_in, hookload_in, temp_in]]
        pred   = model.predict(inp)[0]
        proba  = model.predict_proba(inp)[0]
        classes = model.classes_
        conf   = dict(zip(classes, proba))
        risk   = 1 - conf.get("Normal", 0)

        if risk > 0.6:
            css = "risk-critical"
            icon = "🔴"
        elif risk > 0.25:
            css = "risk-warning"
            icon = "🟡"
        else:
            css = "risk-normal"
            icon = "🟢"

        color = EVENT_COLORS.get(pred, "#ffffff")
        st.markdown(
            f'<div class="{css}">'
            f'<div style="font-family:IBM Plex Mono;font-size:11px;color:#1e3a8a">{icon} DETECTED EVENT</div>'
            f'<div style="font-family:IBM Plex Mono;font-size:16px;font-weight:500;color:{color}">'
            f'{pred}</div>'
            f'<div style="font-family:IBM Plex Mono;font-size:22px;color:#f0a500">'
            f'Risk: {risk:.0%}</div></div>',
            unsafe_allow_html=True)
        st.markdown(f'<div style="font-family:IBM Plex Mono;font-size:11px;color:#1e3a8a;margin-top:8px">'
                    f'{EVENT_DESCRIPTIONS.get(pred,"")}</div>', unsafe_allow_html=True)
        st.markdown("**Event probabilities:**")
        for evt, prob in sorted(conf.items(), key=lambda x: x[1], reverse=True):
            c = EVENT_COLORS.get(evt, "#aaa")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;margin:2px 0">'
                f'<span style="font-family:IBM Plex Mono;font-size:11px;color:{c}">{evt}</span>'
                f'<span style="font-family:IBM Plex Mono;font-size:11px;color:#111827">{prob:.1%}</span>'
                f'</div>', unsafe_allow_html=True)
        if pred != "Normal":
            cost = EVENT_COST.get(pred, 0)
            st.markdown(
                f'<div style="margin-top:8px;font-family:IBM Plex Mono;font-size:12px;'
                f'color:#f0a500">Est. cost if not caught: ${cost:,}</div>',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-family:IBM Plex Mono;font-size:10px;color:#2563eb'>
    Built by Bose Abubakre<br>
    PhD Geoscientist · MBA<br>
    github.com/boseabubakre
    </div>""", unsafe_allow_html=True)

# ── ROW 1: REAL-TIME MONITORING ───────────────────────────────────────────────
st.markdown("#### Real-Time Drilling Monitoring — Simulated Live Feed (8,500–9,000 ft)")

fig_rt = make_subplots(
    rows=2, cols=3,
    subplot_titles=["WOB (klbs)", "ROP (ft/hr)", "Torque (ft·lbs)",
                    "Hookload (klbs)", "SPP (psi)", "NPT Risk Score"],
    horizontal_spacing=0.08, vertical_spacing=0.18
)

params = [
    ("WOB", 1, 1, "#f0a500"),
    ("ROP", 1, 2, "#58a6ff"),
    ("TORQUE", 1, 3, "#ff7b7b"),
    ("HOOKLOAD", 2, 1, "#bc8cff"),
    ("SPP", 2, 2, "#3fb950"),
]

for param, row, col, color in params:
    fig_rt.add_trace(go.Scatter(
        x=rt_df["DEPTH_FT"], y=rt_df[param],
        mode="lines", name=param,
        line=dict(color=color, width=1.5),
        showlegend=False
    ), row=row, col=col)
    # Mark NPT events
    npt_mask = rt_df["PREDICTED_EVENT"] != "Normal"
    if npt_mask.any():
        fig_rt.add_trace(go.Scatter(
            x=rt_df.loc[npt_mask, "DEPTH_FT"],
            y=rt_df.loc[npt_mask, param],
            mode="markers", name=f"{param} NPT",
            marker=dict(color="#f85149", size=6, symbol="x"),
            showlegend=False
        ), row=row, col=col)

# Risk score panel
fig_rt.add_trace(go.Scatter(
    x=rt_df["DEPTH_FT"], y=rt_df["MAX_RISK"],
    mode="lines", name="Risk",
    line=dict(color="#f85149", width=2),
    fill="tozeroy", fillcolor="rgba(248,81,73,0.15)",
    showlegend=False
), row=2, col=3)
fig_rt.add_hline(y=0.5, line=dict(color="#f0a500", dash="dash", width=1),
                 row=2, col=3,
                 annotation_text="Alert", annotation_font_color="#f0a500",
                 annotation_font_size=9)
fig_rt.add_hline(y=0.25, line=dict(color="#3fb950", dash="dash", width=1),
                 row=2, col=3,
                 annotation_text="Caution", annotation_font_color="#3fb950",
                 annotation_font_size=9)

fig_rt.update_layout(**chart_layout(
    height=460,
    margin=dict(l=50, r=20, t=50, b=30),
    showlegend=False
))
st.plotly_chart(fig_rt, use_container_width=True)

# ── ROW 2: DEPTH PLOT + EVENT TIMELINE ───────────────────────────────────────
st.markdown("---")
c1, c2 = st.columns([3, 2])

with c1:
    st.markdown("#### Drilling Parameter Depth Plot — All Events")
    filtered = df[
        (df["DEPTH_FT"] >= depth_range[0]) &
        (df["DEPTH_FT"] <= depth_range[1]) &
        (df["EVENT"].isin(show_events))
    ].copy()

    fig_depth = make_subplots(rows=1, cols=3, horizontal_spacing=0.04,
                               subplot_titles=["ROP (ft/hr)", "Torque (ft·lbs)", "ECD (ppg)"])

    for param, col in [("ROP", 1), ("TORQUE", 2), ("ECD", 3)]:
        for evt in show_events:
            mask = filtered["EVENT"] == evt
            if mask.any():
                fig_depth.add_trace(go.Scatter(
                    x=filtered.loc[mask, param],
                    y=filtered.loc[mask, "DEPTH_FT"],
                    mode="markers", name=evt,
                    marker=dict(color=EVENT_COLORS[evt], size=3, opacity=0.6),
                    showlegend=(col == 1)
                ), row=1, col=col)

    fig_depth.update_layout(**chart_layout(
        height=420,
        margin=dict(l=50, r=20, t=40, b=30)
    ))
    for i in range(1, 4):
        fig_depth.update_yaxes(autorange="reversed", row=1, col=i)
        if i > 1:
            fig_depth.update_yaxes(showticklabels=False, row=1, col=i)
    st.plotly_chart(fig_depth, use_container_width=True)

with c2:
    st.markdown("#### NPT Event Distribution")
    event_counts = df[df["EVENT"] != "Normal"]["EVENT"].value_counts()
    fig_evt = go.Figure(go.Bar(
        x=event_counts.values,
        y=event_counts.index,
        orientation="h",
        marker_color=[EVENT_COLORS[e] for e in event_counts.index],
        text=[f"{v} events — ${EVENT_COST[e]/1000:.0f}K avg" for v, e in
              zip(event_counts.values, event_counts.index)],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=9, color="#1e3a8a")
    ))
    fig_evt.update_layout(**chart_layout(
        height=200,
        margin=dict(l=120, r=150, t=20, b=20),
        xaxis=dict(title="Count", gridcolor="#dbeafe")
    ))
    st.plotly_chart(fig_evt, use_container_width=True)

    st.markdown("#### Feature Importance")
    fig_imp = go.Figure(go.Bar(
        x=importance["Importance"],
        y=importance["Feature"],
        orientation="h",
        marker_color="#58a6ff",
        text=[f"{v:.3f}" for v in importance["Importance"]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=9, color="#1e3a8a")
    ))
    fig_imp.update_layout(**chart_layout(
        height=200,
        margin=dict(l=80, r=60, t=20, b=20),
        xaxis=dict(title="Importance", gridcolor="#dbeafe")
    ))
    st.plotly_chart(fig_imp, use_container_width=True)

# ── ROW 3: MODEL PERFORMANCE ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Model Performance by Event Type")

report = classification_report(y_test, y_pred, output_dict=True)
classes_r = [k for k in report.keys()
             if k not in ["accuracy","macro avg","weighted avg"]]

p1, p2, p3 = st.columns(3)
metrics_data = []
for cls in classes_r:
    metrics_data.append({
        "Event":     cls,
        "Precision": report[cls]["precision"],
        "Recall":    report[cls]["recall"],
        "F1 Score":  report[cls]["f1-score"],
        "Support":   int(report[cls]["support"]),
        "Est. Cost": f"${EVENT_COST.get(cls,0):,}"
    })
metrics_df = pd.DataFrame(metrics_data)

with p1:
    fig_p = go.Figure(go.Bar(
        x=[r["Precision"] for r in metrics_data],
        y=[r["Event"] for r in metrics_data],
        orientation="h",
        marker_color=[EVENT_COLORS.get(r["Event"],"#888") for r in metrics_data],
        text=[f"{r['Precision']:.0%}" for r in metrics_data],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=9, color="#1e3a8a")
    ))
    fig_p.update_layout(**chart_layout(
        height=240, margin=dict(l=120,r=50,t=30,b=20),
        xaxis=dict(range=[0,1.2], title="Precision", gridcolor="#dbeafe"),
        title=dict(text="Precision", font=dict(size=12, color="#cce0ff"))
    ))
    st.plotly_chart(fig_p, use_container_width=True)

with p2:
    fig_r = go.Figure(go.Bar(
        x=[r["Recall"] for r in metrics_data],
        y=[r["Event"] for r in metrics_data],
        orientation="h",
        marker_color=[EVENT_COLORS.get(r["Event"],"#888") for r in metrics_data],
        text=[f"{r['Recall']:.0%}" for r in metrics_data],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=9, color="#1e3a8a")
    ))
    fig_r.update_layout(**chart_layout(
        height=240, margin=dict(l=120,r=50,t=30,b=20),
        xaxis=dict(range=[0,1.2], title="Recall", gridcolor="#dbeafe"),
        title=dict(text="Recall", font=dict(size=12, color="#cce0ff"))
    ))
    st.plotly_chart(fig_r, use_container_width=True)

with p3:
    fig_f = go.Figure(go.Bar(
        x=[r["F1 Score"] for r in metrics_data],
        y=[r["Event"] for r in metrics_data],
        orientation="h",
        marker_color=[EVENT_COLORS.get(r["Event"],"#888") for r in metrics_data],
        text=[f"{r['F1 Score']:.0%}" for r in metrics_data],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=9, color="#1e3a8a")
    ))
    fig_f.update_layout(**chart_layout(
        height=240, margin=dict(l=120,r=50,t=30,b=20),
        xaxis=dict(range=[0,1.2], title="F1 Score", gridcolor="#dbeafe"),
        title=dict(text="F1 Score", font=dict(size=12, color="#cce0ff"))
    ))
    st.plotly_chart(fig_f, use_container_width=True)

# ── NPT COST SUMMARY TABLE ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### NPT Cost Summary — Value of Early Detection")

summary_rows = []
for cls in classes_r:
    if cls == "Normal":
        continue
    n_events = int(report[cls]["support"])
    recall   = report[cls]["recall"]
    cost_per = EVENT_COST.get(cls, 0)
    caught   = int(n_events * recall)
    saved    = caught * cost_per * 0.7  # assume 70% cost avoided if caught early
    summary_rows.append({
        "Event Type":        cls,
        "Events Detected":   caught,
        "Total Events":      n_events,
        "Detection Rate":    f"{recall:.0%}",
        "Avg Cost/Event":    f"${cost_per:,}",
        "Est. Cost Avoided": f"${saved:,.0f}",
    })

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='font-family:IBM Plex Mono;font-size:10px;color:#2563eb;text-align:center;padding:8px'>
Drilling NPT Early Warning System &nbsp;·&nbsp;
Built by Bose Abubakre, PhD MBA &nbsp;·&nbsp;
github.com/boseabubakre &nbsp;·&nbsp;
Data: Synthetic model based on BOEM GOM drilling reports &nbsp;·&nbsp;
For portfolio and research use only — not for operational drilling decisions
</div>
""", unsafe_allow_html=True)
