"""
Hybrid NIDS Dashboard
=====================
Professional SOC-grade Streamlit dashboard for the
Hybrid Network Intrusion Detection System.
Connects to the FastAPI backend at localhost:8000.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# ── Config ──────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Hybrid NIDS — Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Refined Color Palette ───────────────────────────────────────
C = {
    "bg":          "#0a0e17",
    "bg2":         "#111827",
    "card":        "#1a1f2e",
    "card_hover":  "#232a3b",
    "border":      "#2a3045",
    "border_hi":   "#3d4663",
    "cyan":        "#00d4ff",
    "cyan_dim":    "#0099bb",
    "green":       "#00e68a",
    "green_bg":    "rgba(0,230,138,0.08)",
    "red":         "#ff4d6a",
    "red_bg":      "rgba(255,77,106,0.08)",
    "amber":       "#ffb347",
    "amber_bg":    "rgba(255,179,71,0.08)",
    "purple":      "#a78bfa",
    "text":        "#e8edf5",
    "text2":       "#8892a8",
    "text3":       "#5c6478",
}

# ── Inject CSS ──────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
    --bg: {C['bg']};
    --card: {C['card']};
    --cyan: {C['cyan']};
    --green: {C['green']};
    --red: {C['red']};
    --amber: {C['amber']};
}}

/* ── Global ─────────────────────────────── */
.stApp {{
    background: {C['bg']};
    font-family: 'Inter', -apple-system, sans-serif;
}}
html, body, .stApp, .main .block-container {{
    color: {C['text']};
}}
.block-container {{
    padding-top: 2rem !important;
    max-width: 1400px;
}}

/* ── Sidebar ────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0d1321 0%, #0a0e17 100%);
    border-right: 1px solid {C['border']};
}}
section[data-testid="stSidebar"] * {{
    color: {C['text']} !important;
}}
section[data-testid="stSidebar"] .stRadio > div {{
    gap: 2px;
}}
section[data-testid="stSidebar"] .stRadio label {{
    padding: 10px 16px !important;
    border-radius: 8px;
    transition: background 0.15s;
    font-size: 0.9rem !important;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(0,212,255,0.06);
}}
section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
section[data-testid="stSidebar"] [aria-checked="true"] {{
    background: rgba(0,212,255,0.1) !important;
    border-left: 3px solid {C['cyan']} !important;
}}

/* ── Hide streamlit chrome ──────────────── */
#MainMenu, footer, header {{visibility: hidden;}}

/* ── Typography ─────────────────────────── */
.hero-title {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {C['text']};
    letter-spacing: -0.02em;
    margin: 0 0 4px 0;
    line-height: 1.3;
}}
.hero-sub {{
    font-size: 0.88rem;
    color: {C['text2']};
    margin: 0 0 28px 0;
    line-height: 1.5;
}}
.section-label {{
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: {C['text3']};
    margin: 24px 0 12px 0;
    padding: 0;
}}

/* ── Cards ──────────────────────────────── */
.soc-card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
}}
.soc-card:hover {{
    border-color: {C['border_hi']};
}}

/* ── KPI tiles ──────────────────────────── */
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin: 16px 0 24px 0;
}}
.kpi {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    transition: border-color 0.2s, transform 0.15s;
}}
.kpi:hover {{
    border-color: {C['cyan']};
    transform: translateY(-1px);
}}
.kpi-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: {C['cyan']};
    line-height: 1.2;
}}
.kpi-label {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: {C['text3']};
    margin-top: 6px;
}}

/* ── Verdict banners ────────────────────── */
.verdict {{
    border-radius: 14px;
    padding: 28px 32px;
    margin: 16px 0;
    position: relative;
    overflow: hidden;
}}
.verdict::before {{
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
}}
.verdict-safe {{
    background: linear-gradient(135deg, rgba(0,230,138,0.08) 0%, rgba(0,180,110,0.04) 100%);
    border: 1px solid rgba(0,230,138,0.2);
}}
.verdict-safe::before {{ background: {C['green']}; }}
.verdict-attack {{
    background: linear-gradient(135deg, rgba(255,77,106,0.10) 0%, rgba(200,50,80,0.04) 100%);
    border: 1px solid rgba(255,77,106,0.25);
}}
.verdict-attack::before {{ background: {C['red']}; }}
.verdict-anomaly {{
    background: linear-gradient(135deg, rgba(255,179,71,0.10) 0%, rgba(200,140,50,0.04) 100%);
    border: 1px solid rgba(255,179,71,0.25);
}}
.verdict-anomaly::before {{ background: {C['amber']}; }}
.verdict h3 {{
    margin: 0 0 10px 0;
    font-size: 1.25rem;
    font-weight: 700;
}}
.verdict p {{
    margin: 4px 0;
    color: {C['text2']};
    font-size: 0.9rem;
    line-height: 1.6;
}}
.verdict .tag {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}}

/* ── Pipeline layers ────────────────────── */
.pipeline {{
    display: flex;
    flex-direction: column;
    gap: 0;
}}
.pipe-layer {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 18px 22px;
    display: flex;
    align-items: center;
    gap: 18px;
    transition: border-color 0.2s;
}}
.pipe-layer:hover {{
    border-color: {C['cyan']};
}}
.pipe-num {{
    width: 36px; height: 36px;
    border-radius: 50%;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    color: {C['cyan']};
    flex-shrink: 0;
}}
.pipe-text h4 {{
    margin: 0;
    font-size: 0.95rem;
    font-weight: 600;
    color: {C['text']};
}}
.pipe-text p {{
    margin: 2px 0 0;
    font-size: 0.8rem;
    color: {C['text2']};
}}
.pipe-arrow {{
    text-align: center;
    color: {C['border_hi']};
    font-size: 0.9rem;
    margin: 4px 0;
    padding-left: 17px;
}}

/* ── Decision matrix ────────────────────── */
.matrix-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.85rem;
}}
.matrix-table th {{
    background: rgba(0,212,255,0.06);
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: {C['text3']};
    border-bottom: 1px solid {C['border']};
}}
.matrix-table td {{
    padding: 12px 14px;
    border-bottom: 1px solid {C['border']};
    color: {C['text']};
}}
.matrix-table tr:last-child td {{ border-bottom: none; }}
.matrix-table tr:hover td {{ background: rgba(0,212,255,0.03); }}

/* ── Class pills ────────────────────────── */
.class-pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0;
}}
.pill {{
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid;
}}
.pill-safe {{ color: {C['green']}; border-color: rgba(0,230,138,0.3); background: rgba(0,230,138,0.08); }}
.pill-attack {{ color: {C['red']}; border-color: rgba(255,77,106,0.3); background: rgba(255,77,106,0.08); }}

/* ── Scrutiny cards ─────────────────────── */
.scru-card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 18px 20px;
    margin: 6px 0;
    cursor: pointer;
    transition: all 0.15s;
}}
.scru-card:hover {{ border-color: {C['cyan']}; }}
.scru-card.active {{
    border-color: {C['cyan']};
    background: rgba(0,212,255,0.05);
    box-shadow: 0 0 20px rgba(0,212,255,0.06);
}}

/* ── Status ─────────────────────────────── */
.status-dot {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}}
.status-on {{
    background: rgba(0,230,138,0.1);
    color: {C['green']};
    border: 1px solid rgba(0,230,138,0.25);
}}
.status-off {{
    background: rgba(255,77,106,0.1);
    color: {C['red']};
    border: 1px solid rgba(255,77,106,0.25);
}}

/* ── Plotly container fix ───────────────── */
.stPlotlyChart {{
    background: transparent !important;
}}

/* ── Quick-fill buttons ─────────────────── */
.stButton > button {{
    border-radius: 8px !important;
    border: 1px solid {C['border']} !important;
    background: {C['card']} !important;
    color: {C['text']} !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 8px 10px !important;
    transition: all 0.15s !important;
}}
.stButton > button:hover {{
    border-color: {C['cyan']} !important;
    background: rgba(0,212,255,0.06) !important;
    color: {C['cyan']} !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {C['cyan_dim']} 0%, {C['cyan']} 100%) !important;
    border: none !important;
    color: #000 !important;
    font-weight: 700 !important;
    padding: 12px !important;
}}
.stButton > button[kind="primary"]:hover {{
    opacity: 0.9 !important;
}}

/* ── Tabs ───────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 1px solid {C['border']};
}}
.stTabs [data-baseweb="tab"] {{
    color: {C['text2']} !important;
    font-size: 0.85rem;
    padding: 10px 20px;
}}
.stTabs [aria-selected="true"] {{
    color: {C['cyan']} !important;
    border-bottom-color: {C['cyan']} !important;
}}

/* ── Number inputs ──────────────────────── */
.stNumberInput input {{
    background: {C['bg2']} !important;
    border: 1px solid {C['border']} !important;
    color: {C['text']} !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}}

/* ── Expanders ──────────────────────────── */
.streamlit-expanderHeader {{
    background: {C['card']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    color: {C['text']} !important;
}}

/* ── Slider ─────────────────────────────── */
.stSlider > div > div > div {{
    color: {C['text2']} !important;
}}

/* ── Dataframe ──────────────────────────── */
.stDataFrame {{
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid {C['border']};
}}

/* ── Download button ────────────────────── */
.stDownloadButton > button {{
    background: {C['card']} !important;
    border: 1px solid {C['border']} !important;
    color: {C['cyan']} !important;
}}
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────
def api_get(path: str):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None


def api_post(path: str, data: dict):
    try:
        r = requests.post(f"{API_URL}{path}", json=data, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None


def get_model_info():
    if "model_info" not in st.session_state:
        info = api_get("/model/info")
        if info:
            st.session_state.model_info = info
    return st.session_state.get("model_info")


# ── Feature grouping for CICIDS2017 ────────────────────────────
FEATURE_GROUPS = {
    "🔌 Port & Flow Basics":    (0, 6),
    "📏 Packet Lengths":         (6, 14),
    "⏱️ Flow Rate":              (14, 16),
    "🕐 Inter-Arrival Times":    (16, 30),
    "🚩 Push/URG Flags":         (30, 34),
    "📊 Header & Throughput":    (34, 38),
    "📈 Packet Statistics":      (38, 43),
    "🏴 TCP Flag Counts":        (43, 51),
    "📐 Ratios & Bulk Stats":    (51, 62),
    "🔄 Subflow Metrics":        (62, 66),
    "🪟 Window & Activity":      (66, 78),
}


def generate_sample(attack_type: str, n_features: int) -> list[float]:
    """Generate a realistic-looking sample flow for demo."""
    np.random.seed(None)
    if n_features != 78:
        base = np.random.randn(n_features).tolist()
        if attack_type != "BENIGN":
            for i in range(min(4, n_features)):
                base[i] += 8.0
        return base

    base = np.zeros(n_features)
    base[0] = np.random.choice([80, 443, 22, 53, 8080])
    base[1] = np.random.exponential(50000)
    base[2] = np.random.poisson(5) + 1
    base[3] = np.random.poisson(3) + 1
    base[4] = np.random.exponential(500)
    base[5] = np.random.exponential(300)
    for i in range(6, 14):
        base[i] = abs(np.random.randn() * 100)
    base[14] = np.random.exponential(10000)
    base[15] = np.random.exponential(50)
    for i in range(16, 30):
        base[i] = abs(np.random.exponential(5000))
    for i in range(34, 38):
        base[i] = abs(np.random.randn() * 20)
    for i in range(38, 43):
        base[i] = abs(np.random.randn() * 100)
    base[47] = 1
    base[52] = np.random.exponential(200)
    for i in range(62, 66):
        base[i] = abs(np.random.poisson(3))
    base[66] = np.random.choice([8192, 16384, 32768, 65535])
    base[67] = np.random.choice([8192, 16384, 32768, 65535])
    base[69] = 20

    if "DoS" in attack_type or attack_type == "DDoS":
        base[0] = 80
        base[2] = np.random.poisson(50) + 20
        base[4] = np.random.exponential(5000) + 2000
        base[14] = np.random.exponential(100000) + 50000
        base[15] = np.random.exponential(500) + 200
        for i in range(16, 20):
            base[i] = abs(np.random.randn() * 50)
        base[44] = 1
        if attack_type == "DDoS":
            base[2] = np.random.poisson(100) + 50
            base[14] = np.random.exponential(200000) + 100000
    elif attack_type == "PortScan":
        base[0] = np.random.randint(1, 65535)
        base[2] = 1
        base[3] = 0
        base[1] = np.random.exponential(100)
        base[44] = 1
        for i in range(16, 20):
            base[i] = abs(np.random.randn() * 10)
    elif "Patator" in attack_type:
        base[0] = 21 if "FTP" in attack_type else 22
        base[2] = np.random.poisson(10) + 5
        base[1] = np.random.exponential(10000) + 5000
        base[8] = np.random.exponential(50) + 20
    elif "Web Attack" in attack_type:
        base[0] = 80
        base[2] = np.random.poisson(15) + 5
        base[4] = np.random.exponential(2000) + 500
        base[46] = 1
    elif attack_type == "Infiltration":
        base[1] = np.random.exponential(100000) + 50000
        base[4] = np.random.exponential(3000) + 1000
        base[70] = np.random.exponential(10000) + 5000
    elif attack_type == "Heartbleed":
        base[0] = 443
        base[5] = np.random.exponential(50000) + 10000
        base[12] = np.random.exponential(5000) + 2000
    elif attack_type == "Bot":
        base[2] = np.random.poisson(20) + 10
        base[3] = np.random.poisson(20) + 10
        base[52] = np.random.exponential(100) + 50

    return base.tolist()


PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=11, color=C['text']),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor=C['border'], zerolinecolor=C['border']),
    yaxis=dict(gridcolor=C['border'], zerolinecolor=C['border']),
)


def plot_shap(shap_data: list[dict]):
    if not shap_data:
        return
    features = [f["feature"] for f in reversed(shap_data)]
    values = [f["shap_value"] for f in reversed(shap_data)]
    colors = [C['red'] if v > 0 else C['green'] for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, family="JetBrains Mono"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="SHAP Feature Contributions", font=dict(size=13, color=C['text2'])),
        height=260,
        margin=dict(l=10, r=90, t=36, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_class_probs(probs: dict):
    sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top = sorted_p[:8]  # show top 8
    classes = [c for c, _ in top]
    values = [v for _, v in top]
    colors = [C['green'] if c == "BENIGN" else C['red'] for c in classes]

    fig = go.Figure(go.Bar(
        x=classes, y=values,
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="auto",
        textfont=dict(size=10, family="JetBrains Mono"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Class Probabilities", font=dict(size=13, color=C['text2'])),
        yaxis_range=[0, 1],
        height=280,
        xaxis=dict(tickangle=-45, gridcolor=C['border']),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 24px 0 8px;">
        <div style="font-size: 2.2rem; margin-bottom: 2px;">🛡️</div>
        <div style="font-size: 1.15rem; font-weight: 700; color: {C['text']};
                    letter-spacing: -0.01em;">Hybrid NIDS</div>
        <div style="font-size: 0.65rem; color: {C['text3']};
                    letter-spacing: 2.5px; text-transform: uppercase;
                    margin-top: 4px;">
            Intrusion Detection System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div style="height:1px; background:{C["border"]}; margin:16px 0;"></div>', unsafe_allow_html=True)

    health = api_get("/health")
    if health:
        uptime_min = health['uptime_seconds'] / 60
        st.markdown(f"""
        <div style="text-align:center; margin: 12px 0 8px;">
            <span class="status-dot status-on">● ONLINE</span>
        </div>
        <div style="text-align:center; color:{C['text3']}; font-size:0.72rem;
                    font-family:'JetBrains Mono',monospace; margin-bottom: 4px;">
            {uptime_min:.0f}m uptime · {health['total_alerts']} alerts
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align:center; margin: 12px 0 8px;">
            <span class="status-dot status-off">● OFFLINE</span>
        </div>
        <div style="text-align:center; color:{C['text3']}; font-size:0.72rem; margin-bottom:4px;">
            Start the API server
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<div style="height:1px; background:{C["border"]}; margin:16px 0;"></div>', unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🔍  Live Analysis", "📊  Batch Analysis",
         "🚨  Alert Log", "🧠  Model Info", "⚙️  Settings"],
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <div style="position:absolute; bottom:20px; left:0; right:0; text-align:center;
                color:{C['text3']}; font-size:0.65rem; line-height:1.8;">
        Hybrid NIDS v1.0<br>
        RF · Autoencoder · SHAP<br>
        VIT Bhopal · B.Tech CSE (Cyber)
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
#  PAGE 1 — LIVE ANALYSIS
# ═════════════════════════════════════════════════════════════════
if page == "🔍  Live Analysis":
    st.markdown('<p class="hero-title">Live Network Flow Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Analyze individual network flows through the 4-layer hybrid detection pipeline — Random Forest classification, Autoencoder anomaly scoring, and SHAP explainability.</p>', unsafe_allow_html=True)

    info = get_model_info()
    if not info:
        st.warning("⚠️ Cannot connect to the API. Start the backend with: `uvicorn src.api:app --port 8000`")
        st.stop()

    n_features = info["n_features"]
    feature_names = info["feature_names"]
    classes = info["classes"]

    # ── Quick-fill ──
    st.markdown('<p class="section-label">Quick-Fill Demo Samples</p>', unsafe_allow_html=True)

    main_attacks = ["BENIGN", "DDoS", "DoS Hulk", "PortScan", "FTP-Patator", "SSH-Patator", "Bot"]
    other_attacks = [c for c in classes if c not in main_attacks]
    fill_type = None

    cols = st.columns(min(len(main_attacks), 7))
    for i, cls in enumerate(main_attacks):
        if cls in classes or cls == "BENIGN":
            with cols[i % len(cols)]:
                label = f"{'🟢' if cls == 'BENIGN' else '🔴'} {cls}"
                if st.button(label, key=f"qf_{cls}", use_container_width=True):
                    fill_type = cls

    if other_attacks:
        cols2 = st.columns(min(len(other_attacks), 8))
        for i, cls in enumerate(other_attacks):
            with cols2[i % len(cols2)]:
                if st.button(f"🔴 {cls}", key=f"qf2_{cls}", use_container_width=True):
                    fill_type = cls

    if fill_type:
        st.session_state["sample_features"] = generate_sample(fill_type, n_features)

    st.markdown(f'<div style="height:1px; background:{C["border"]}; margin:20px 0;"></div>', unsafe_allow_html=True)

    # ── Two-column layout ──
    col_input, col_result = st.columns([1, 1.3], gap="large")

    with col_input:
        st.markdown('<p class="section-label">Input Features</p>', unsafe_allow_html=True)

        features = st.session_state.get("sample_features", [0.0] * n_features)
        edited = list(features)

        if n_features == 78:
            for group_name, (start, end) in FEATURE_GROUPS.items():
                if end <= n_features:
                    expanded = (start == 0)
                    with st.expander(f"{group_name}  ({end - start} features)", expanded=expanded):
                        for j in range(start, min(end, n_features)):
                            edited[j] = st.number_input(
                                feature_names[j], value=float(features[j]),
                                format="%.2f", key=f"f_{j}",
                            )
        else:
            for g in range(0, n_features, 5):
                end = min(g + 5, n_features)
                with st.expander(f"Features {g}–{end-1}", expanded=(g == 0)):
                    for j in range(g, end):
                        edited[j] = st.number_input(
                            feature_names[j], value=float(features[j]),
                            format="%.4f", key=f"f_{j}",
                        )

        st.markdown("")
        analyze = st.button("⚡ Analyze Flow", type="primary", use_container_width=True)

    with col_result:
        st.markdown('<p class="section-label">Detection Result</p>', unsafe_allow_html=True)

        if analyze:
            with st.spinner("Running detection pipeline..."):
                result = api_post("/predict", {"features": edited})

            if result:
                pred = result["prediction"]
                conf = result["confidence"]
                is_anom = result["is_anomaly"]
                anom_sc = result["anomaly_score"]
                thresh = result["threshold_used"]

                # Verdict banner
                if pred == "BENIGN" and not is_anom:
                    st.markdown(f"""
                    <div class="verdict verdict-safe">
                        <h3 style="color:{C['green']}">✅ BENIGN — No Threat Detected</h3>
                        <p>RF Confidence: <span class="tag" style="background:{C['green_bg']};color:{C['green']}">{conf:.1%}</span>
                        &nbsp; Anomaly Score: <span class="tag" style="background:{C['green_bg']};color:{C['green']}">{anom_sc:.4f}</span></p>
                        <p>Traffic pattern matches known-normal profiles. No action required.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif pred == "BENIGN" and is_anom:
                    st.markdown(f"""
                    <div class="verdict verdict-anomaly">
                        <h3 style="color:{C['amber']}">⚠️ ANOMALY — Possible Zero-Day</h3>
                        <p>RF says BENIGN, but anomaly score <span class="tag" style="background:{C['amber_bg']};color:{C['amber']}">{anom_sc:.4f}</span>
                        exceeds threshold <span class="tag" style="background:{C['amber_bg']};color:{C['amber']}">{thresh:.4f}</span></p>
                        <p>The autoencoder detected deviation from normal patterns — this could be an unknown attack the classifier hasn't seen before.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict verdict-attack">
                        <h3 style="color:{C['red']}">🚨 ATTACK — {pred}</h3>
                        <p>Confidence: <span class="tag" style="background:{C['red_bg']};color:{C['red']}">{conf:.1%}</span>
                        &nbsp; Anomaly Score: <span class="tag" style="background:{C['red_bg']};color:{C['red']}">{anom_sc:.4f}</span>
                        &nbsp; Scrutiny: <span class="tag" style="background:rgba(167,139,250,0.1);color:{C['purple']}">{result['scrutiny_level'].upper()}</span></p>
                    </div>
                    """, unsafe_allow_html=True)

                # KPI row
                st.markdown(f"""
                <div class="kpi-row">
                    <div class="kpi">
                        <div class="kpi-val" style="color:{C['green'] if conf > 0.8 else C['amber']}">{conf:.1%}</div>
                        <div class="kpi-label">Confidence</div>
                    </div>
                    <div class="kpi">
                        <div class="kpi-val" style="color:{C['red'] if is_anom else C['green']}">{anom_sc:.4f}</div>
                        <div class="kpi-label">Anomaly Score</div>
                    </div>
                    <div class="kpi">
                        <div class="kpi-val">{thresh:.4f}</div>
                        <div class="kpi-label">Threshold</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Charts
                plot_shap(result.get("shap_top_features", []))
                plot_class_probs(result.get("class_probabilities", {}))
        else:
            st.markdown(f"""
            <div class="soc-card" style="text-align:center; padding: 80px 20px;">
                <div style="font-size: 2.5rem; margin-bottom: 14px; opacity: 0.6;">⚡</div>
                <div style="font-size: 1rem; color: {C['text2']}; margin-bottom: 6px;">
                    Ready to Analyze
                </div>
                <div style="color: {C['text3']}; font-size: 0.82rem; line-height:1.7;">
                    Select a Quick-Fill sample above or enter feature values,<br>
                    then click <b style="color:{C['cyan']}">Analyze Flow</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
#  PAGE 2 — BATCH ANALYSIS
# ═════════════════════════════════════════════════════════════════
elif page == "📊  Batch Analysis":
    st.markdown('<p class="hero-title">Batch Flow Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Analyze multiple network flows at once — upload a CSV or generate test traffic.</p>', unsafe_allow_html=True)

    info = get_model_info()
    if not info:
        st.warning("⚠️ Cannot connect to API.")
        st.stop()
    n_features = info["n_features"]

    tab_upload, tab_gen = st.tabs(["📁  Upload CSV", "🎲  Generate Test Batch"])

    with tab_upload:
        uploaded = st.file_uploader("Upload CSV with flow features", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.markdown(f"""
            <div class="soc-card">
                <span style="color:{C['cyan']}; font-weight:600;">{len(df)}</span> flows ·
                <span style="color:{C['text2']}">{len(df.columns)} columns</span>
            </div>
            """, unsafe_allow_html=True)
            feature_df = df.drop("Label", axis=1) if "Label" in df.columns else df

            if len(feature_df.columns) != n_features:
                st.error(f"Expected {n_features} features, got {len(feature_df.columns)}")
            elif st.button("⚡ Analyze Batch", type="primary"):
                with st.spinner(f"Analyzing {len(df)} flows..."):
                    result = api_post("/predict/batch", {"flows": feature_df.values.tolist()})
                if result:
                    _render_batch_results(result)

    with tab_gen:
        c1, c2 = st.columns(2)
        with c1:
            n_samples = st.slider("Number of samples", 10, 200, 50)
        with c2:
            attack_ratio = st.slider("Attack ratio", 0.0, 1.0, 0.3)

        if st.button("🎲 Generate & Analyze", type="primary"):
            n_atk = int(n_samples * attack_ratio)
            atk_types = [c for c in info["classes"] if c != "BENIGN"]
            flows = [generate_sample("BENIGN", n_features) for _ in range(n_samples - n_atk)]
            flows += [generate_sample(np.random.choice(atk_types) if atk_types else "DoS", n_features) for _ in range(n_atk)]
            np.random.shuffle(flows)

            with st.spinner(f"Analyzing {n_samples} flows..."):
                result = api_post("/predict/batch", {"flows": flows})

            if result:
                rl = result["results"]
                rdf = pd.DataFrame({
                    "Prediction": [r.get("prediction", "error") for r in rl],
                    "Confidence": [r.get("confidence", 0) for r in rl],
                    "Anomaly Score": [r.get("anomaly_score", 0) for r in rl],
                    "Is Anomaly": [r.get("is_anomaly", False) for r in rl],
                })

                benign_pct = (rdf["Prediction"] == "BENIGN").mean()
                n_anom = int(rdf['Is Anomaly'].sum())
                st.markdown(f"""
                <div class="kpi-row">
                    <div class="kpi"><div class="kpi-val">{len(rdf)}</div><div class="kpi-label">Total Flows</div></div>
                    <div class="kpi"><div class="kpi-val" style="color:{C['green']}">{benign_pct:.0%}</div><div class="kpi-label">Benign</div></div>
                    <div class="kpi"><div class="kpi-val" style="color:{C['red']}">{1-benign_pct:.0%}</div><div class="kpi-label">Attacks</div></div>
                    <div class="kpi"><div class="kpi-val" style="color:{C['amber']}">{n_anom}</div><div class="kpi-label">Anomalies</div></div>
                </div>
                """, unsafe_allow_html=True)

                # Donut chart
                counts = rdf["Prediction"].value_counts()
                chart_colors = [C['green'] if c == "BENIGN" else C['red'] for c in counts.index]
                fig = go.Figure(go.Pie(
                    labels=counts.index, values=counts.values,
                    marker=dict(colors=chart_colors),
                    hole=0.5, textinfo="label+percent",
                    textfont=dict(size=11),
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=360, title=dict(text="Detection Breakdown", font=dict(size=13, color=C['text2'])))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(rdf, use_container_width=True, height=350)

                csv_out = rdf.to_csv(index=False)
                st.download_button("📥 Download Results", csv_out, "nids_results.csv", mime="text/csv")


# ═════════════════════════════════════════════════════════════════
#  PAGE 3 — ALERT LOG
# ═════════════════════════════════════════════════════════════════
elif page == "🚨  Alert Log":
    st.markdown('<p class="hero-title">Alert Log</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Real-time log of detected threats and anomalous traffic patterns.</p>', unsafe_allow_html=True)

    alerts_data = api_get("/alerts?limit=200")
    if not alerts_data:
        st.warning("⚠️ Cannot connect to API.")
        st.stop()

    alerts = alerts_data.get("alerts", [])
    total = alerts_data.get("total", 0)
    n_anom = len([a for a in alerts if a.get('is_anomaly')])
    n_types = len(set(a.get('prediction', '') for a in alerts))

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi"><div class="kpi-val" style="color:{C['red']}">{total}</div><div class="kpi-label">Total Alerts</div></div>
        <div class="kpi"><div class="kpi-val" style="color:{C['amber']}">{n_anom}</div><div class="kpi-label">Anomalies</div></div>
        <div class="kpi"><div class="kpi-val">{n_types}</div><div class="kpi-label">Attack Types</div></div>
    </div>
    """, unsafe_allow_html=True)

    if not alerts:
        st.markdown(f"""
        <div class="soc-card" style="text-align:center; padding: 70px 20px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px; opacity: 0.5;">📋</div>
            <div style="color: {C['text3']}; font-size: 0.88rem;">
                No alerts yet. Analyze some flows to see alerts appear here.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        adf = pd.DataFrame(alerts)
        adf["timestamp"] = pd.to_datetime(adf["timestamp"])

        attack_filter = st.multiselect(
            "Filter by attack type",
            options=sorted(adf["prediction"].unique().tolist()),
            default=adf["prediction"].unique().tolist(),
        )
        filtered = adf[adf["prediction"].isin(attack_filter)]

        if len(filtered) > 1:
            fig = px.histogram(filtered, x="timestamp", color="prediction", title="Alerts Over Time")
            fig.update_layout(**PLOTLY_LAYOUT, height=280, title=dict(font=dict(size=13, color=C['text2'])))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(filtered, use_container_width=True, height=400)


# ═════════════════════════════════════════════════════════════════
#  PAGE 4 — MODEL INFO
# ═════════════════════════════════════════════════════════════════
elif page == "🧠  Model Info":
    st.markdown('<p class="hero-title">Model Architecture</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Understanding the 4-layer hybrid detection pipeline trained on CICIDS2017.</p>', unsafe_allow_html=True)

    info = get_model_info()
    if not info:
        st.warning("⚠️ Cannot connect to API.")
        st.stop()

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi"><div class="kpi-val">4-Layer</div><div class="kpi-label">Architecture</div></div>
        <div class="kpi"><div class="kpi-val">{info['n_features']}</div><div class="kpi-label">Features</div></div>
        <div class="kpi"><div class="kpi-val">{info['n_classes']}</div><div class="kpi-label">Classes</div></div>
        <div class="kpi"><div class="kpi-val" style="font-size:1.2rem">{info['ae_base_threshold']:.4f}</div><div class="kpi-label">AE Threshold</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown('<p class="section-label">Detection Pipeline</p>', unsafe_allow_html=True)

        layers = [
            ("1", "Suricata IDS", "Signature-based detection — known CVEs and rule-based matching"),
            ("2", "Random Forest", f"Supervised ML — {info['n_classes']} attack classes, {info['n_features']} features, balanced weighting"),
            ("3", "Autoencoder", f"Unsupervised anomaly detection — trained on BENIGN-only, threshold: {info['ae_base_threshold']:.4f}"),
            ("4", "SHAP Explainer", "Per-prediction feature attribution — TreeExplainer for interpretable results"),
        ]

        st.markdown('<div class="pipeline">', unsafe_allow_html=True)
        for i, (num, title, desc) in enumerate(layers):
            st.markdown(f"""
            <div class="pipe-layer">
                <div class="pipe-num">{num}</div>
                <div class="pipe-text">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if i < len(layers) - 1:
                st.markdown('<div class="pipe-arrow">↓</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-label">Hybrid Decision Matrix</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="soc-card" style="padding:0; overflow:hidden;">
            <table class="matrix-table">
                <tr><th>Random Forest</th><th>Autoencoder</th><th>Verdict</th></tr>
                <tr>
                    <td style="color:{C['green']}">BENIGN</td>
                    <td style="color:{C['green']}">Normal</td>
                    <td>✅ Safe</td>
                </tr>
                <tr>
                    <td style="color:{C['green']}">BENIGN</td>
                    <td style="color:{C['amber']}">Anomaly</td>
                    <td>⚠️ Possible Zero-Day</td>
                </tr>
                <tr>
                    <td style="color:{C['red']}">Attack</td>
                    <td style="color:{C['amber']}">Anomaly</td>
                    <td>🚨 High Confidence Threat</td>
                </tr>
                <tr>
                    <td style="color:{C['red']}">Attack</td>
                    <td style="color:{C['green']}">Normal</td>
                    <td>🔴 Known Attack Pattern</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-label">Trained Classes</p>', unsafe_allow_html=True)
        pills = "".join(
            f'<span class="pill {"pill-safe" if c == "BENIGN" else "pill-attack"}">{c}</span>'
            for c in info['classes']
        )
        st.markdown(f"""
        <div class="soc-card">
            <div style="color:{C['text3']}; font-size:0.72rem; margin-bottom:10px;">
                {info['n_classes']} classes · CICIDS2017 dataset
            </div>
            <div class="class-pills">{pills}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-label">Component Status</p>', unsafe_allow_html=True)
        rf_ok = "✅" if info.get("has_autoencoder") is not None else "❌"
        ae_ok = "✅" if info.get("has_autoencoder") else "❌"
        sc_ok = "✅" if info.get("has_scaler") else "❌"
        st.markdown(f"""
        <div class="soc-card" style="font-size:0.88rem; line-height:2.2;">
            ✅ Random Forest Classifier<br>
            {ae_ok} Autoencoder (anomaly detection)<br>
            {sc_ok} StandardScaler (feature normalization)<br>
            ✅ SHAP TreeExplainer
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
#  PAGE 5 — SETTINGS
# ═════════════════════════════════════════════════════════════════
elif page == "⚙️  Settings":
    st.markdown('<p class="hero-title">Detection Settings</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Configure anomaly detection sensitivity for your deployment environment.</p>', unsafe_allow_html=True)

    settings = api_get("/settings")
    if not settings:
        st.warning("⚠️ Cannot connect to API.")
        st.stop()

    current = settings["scrutiny"]
    base_thresh = settings["base_threshold"]

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown('<p class="section-label">Scrutiny Level</p>', unsafe_allow_html=True)

        options = [
            ("low",    "🏠", "Home Network",
             "Relaxed detection — fewer false positives, flags only obvious threats.",
             f"Threshold: {base_thresh * 1.5:.4f} (1.5× base)"),
            ("medium", "🏢", "Office / Standard",
             "Balanced detection — good trade-off between sensitivity and accuracy.",
             f"Threshold: {base_thresh:.4f} (1.0× base — default)"),
            ("high",   "🏦", "Enterprise / SOC",
             "Aggressive detection — catches subtle threats, more alerts to triage.",
             f"Threshold: {base_thresh * 0.6:.4f} (0.6× base)"),
            ("custom", "🔧", "Custom Threshold",
             "Fine-grained control — set your own threshold value.",
             "User-defined"),
        ]

        new_scrutiny = st.radio(
            "Scrutiny",
            [o[0] for o in options],
            index=[o[0] for o in options].index(current),
            format_func=lambda x: next(f"{o[1]}  {o[2]}" for o in options if o[0] == x),
            label_visibility="collapsed",
        )

        sel = next(o for o in options if o[0] == new_scrutiny)
        active_class = "active" if new_scrutiny == current else ""
        st.markdown(f"""
        <div class="scru-card {active_class}">
            <div style="font-size:1rem; font-weight:600; color:{C['text']}; margin-bottom:6px;">
                {sel[1]} {sel[2]}
            </div>
            <div style="color:{C['text2']}; font-size:0.85rem; margin-bottom:8px; line-height:1.6;">
                {sel[3]}
            </div>
            <div style="color:{C['cyan']}; font-size:0.8rem; font-family:'JetBrains Mono',monospace;">
                {sel[4]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        custom_val = None
        if new_scrutiny == "custom":
            custom_val = st.number_input(
                "Custom threshold", min_value=0.0001, max_value=10.0,
                value=settings.get("custom_threshold") or base_thresh,
                step=0.01, format="%.4f",
            )

        if st.button("✅ Apply Settings", type="primary", use_container_width=True):
            payload = {"scrutiny": new_scrutiny}
            if new_scrutiny == "custom":
                payload["custom_threshold"] = custom_val
            result = api_post("/settings", payload)
            if result:
                st.success(f"Updated to **{new_scrutiny.upper()}** — effective threshold: {result['effective_threshold']:.4f}")
                st.rerun()

    with col2:
        st.markdown('<p class="section-label">Current Configuration</p>', unsafe_allow_html=True)

        eff = settings["effective_threshold"]
        st.markdown(f"""
        <div class="kpi-row" style="grid-template-columns: 1fr;">
            <div class="kpi">
                <div class="kpi-val">{settings['scrutiny'].upper()}</div>
                <div class="kpi-label">Scrutiny Level</div>
            </div>
        </div>
        <div class="kpi-row" style="grid-template-columns: 1fr 1fr;">
            <div class="kpi">
                <div class="kpi-val" style="color:{C['cyan']}">{eff:.4f}</div>
                <div class="kpi-label">Effective Threshold</div>
            </div>
            <div class="kpi">
                <div class="kpi-val">{base_thresh:.4f}</div>
                <div class="kpi-label">Base (95th %ile)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=eff,
            number=dict(font=dict(color=C['text'], family="JetBrains Mono", size=28)),
            title=dict(text="Anomaly Threshold", font=dict(size=12, color=C['text2'])),
            gauge=dict(
                axis=dict(range=[0, base_thresh * 2.5], tickfont=dict(color=C['text3'], size=9)),
                bar=dict(color=C['cyan']),
                bgcolor=C['card'],
                borderwidth=0,
                steps=[
                    dict(range=[0, base_thresh * 0.6], color="rgba(255,77,106,0.12)"),
                    dict(range=[base_thresh * 0.6, base_thresh], color="rgba(255,179,71,0.12)"),
                    dict(range=[base_thresh, base_thresh * 2.5], color="rgba(0,230,138,0.12)"),
                ],
                threshold=dict(line=dict(color=C['red'], width=2), thickness=0.8, value=base_thresh),
            ),
        ))
        fig.update_layout(height=230, margin=dict(l=20, r=20, t=45, b=5), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="soc-card" style="font-size:0.82rem; color:{C['text2']}; line-height:1.8;">
            <b style="color:{C['text']}">How it works</b><br><br>
            The autoencoder learns what <b>normal</b> traffic looks like.
            For each flow, it computes a <b>reconstruction error</b> —
            how different the flow is from learned patterns.<br><br>
            <span style="color:{C['red']}">■</span> Low threshold → more sensitive, more alerts<br>
            <span style="color:{C['amber']}">■</span> Medium → balanced detection<br>
            <span style="color:{C['green']}">■</span> High threshold → fewer alerts, may miss subtle attacks
        </div>
        """, unsafe_allow_html=True)
