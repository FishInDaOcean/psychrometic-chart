import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Psychrometric Chart & SHF Engine | Singapore Poly Style",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 1. ASHRAE / CIBSE THERMODYNAMIC EQUATIONS
# -----------------------------------------------------------------------------
P_ATM_DEFAULT = 101.325  # kPa (Sea Level)

def get_pws(t_c):
    """Saturation vapor pressure over liquid water (kPa)."""
    return 0.61078 * np.exp((17.27 * t_c) / (t_c + 237.3))

def calc_w(t_c, rh_pct, p_atm=P_ATM_DEFAULT):
    """Moisture Content W (kg/kg dry air)."""
    pws = get_pws(t_c)
    pw = (np.clip(rh_pct, 1e-3, 100.0) / 100.0) * pws
    return (0.62198 * pw) / np.maximum(p_atm - pw, 1e-5)

def calc_rh(t_c, w_kg, p_atm=P_ATM_DEFAULT):
    """Percentage Saturation / Relative Humidity (%)."""
    pw = (p_atm * w_kg) / (0.62198 + w_kg)
    pws = get_pws(t_c)
    return np.clip((pw / pws) * 100.0, 0.0, 100.0)

def calc_h(t_c, w_kg):
    """Specific Enthalpy h (kJ/kg dry air)."""
    return 1.006 * t_c + w_kg * (2501.0 + 1.86 * t_c)

def calc_v(t_c, w_kg, p_atm=P_ATM_DEFAULT):
    """Specific Volume v (m³/kg dry air)."""
    return (0.287058 * (t_c + 273.15) * (1.0 + 1.6078 * w_kg)) / p_atm

def calc_tdp(t_c, rh_pct):
    """Dew Point Temperature (°C)."""
    a, b = 17.27, 237.3
    alpha = ((a * t_c) / (b + t_c)) + np.log(np.clip(rh_pct / 100.0, 1e-4, 1.0))
    return (b * alpha) / (a - alpha)

def calc_twb(t_c, rh_pct):
    """Wet-Bulb Temperature (°C) via Stull formulation."""
    rh = float(rh_pct)
    return (
        t_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )

def get_w_from_twb(t_db, t_wb, p_atm=P_ATM_DEFAULT):
    """Moisture content along a constant wet-bulb sling line."""
    pws_wb = get_pws(t_wb)
    w_s_wb = (0.62198 * pws_wb) / (p_atm - pws_wb)
    return ((2501.0 - 2.381 * t_wb) * w_s_wb - 1.006 * (t_db - t_wb)) / (2501.0 + 1.86 * t_db - 4.186 * t_wb)

# -----------------------------------------------------------------------------
# 2. SIDEBAR CONFIGURATION & STATE MANAGEMENT
# -----------------------------------------------------------------------------
st.sidebar.title("📐 Psychrometric Studio")
st.sidebar.caption("Singapore Polytechnic / Carrier Standard Style")

theme_mode = st.sidebar.radio("Chart Theme Style", ["Classic Engineering Green", "Modern High-Contrast Dark"], index=0)
p_atm = st.sidebar.slider("Atmospheric Pressure (kPa)", 80.0, 110.0, 101.325, 0.25)
airflow = st.sidebar.slider("System Airflow (m³/s)", 0.2, 15.0, 3.0, 0.1)

preset = st.sidebar.selectbox(
    "Circuit / Process Preset",
    [
        "Summer Cooling & Dehumidification (ADP & SHF)",
        "Winter Pre-Heat & Humidification",
        "Direct Evaporative Cooling (DEC)",
        "Custom Multi-Point HVAC Cycle"
    ]
)

if "nodes" not in st.session_state or st.sidebar.button("Reset Circuit Points"):
    if "Summer" in preset:
        st.session_state.nodes = [
            {"id": "1 (OA)", "name": "Outdoor Air", "tdb": 34.0, "rh": 75.0, "color": "#dc2626"},
            {"id": "2 (RA)", "name": "Return Air (Room)", "tdb": 24.0, "rh": 50.0, "color": "#2563eb"},
            {"id": "3 (MA)", "name": "Mixed Air (Coil On)", "tdb": 26.5, "rh": 56.5, "color": "#d97706"},
            {"id": "4 (CC)", "name": "Off-Coil Supply", "tdb": 13.0, "rh": 90.0, "color": "#059669"}
        ]
    elif "Winter" in preset:
        st.session_state.nodes = [
            {"id": "1 (OA)", "name": "Outdoor Air", "tdb": 4.0, "rh": 85.0, "color": "#2563eb"},
            {"id": "2 (HC)", "name": "Heated Air", "tdb": 20.0, "rh": 30.0, "color": "#dc2626"},
            {"id": "3 (SA)", "name": "Humidified Supply", "tdb": 21.0, "rh": 55.0, "color": "#059669"}
        ]
    elif "Evaporative" in preset:
        st.session_state.nodes = [
            {"id": "1 (OA)", "name": "Hot Ambient", "tdb": 38.0, "rh": 20.0, "color": "#dc2626"},
            {"id": "2 (SA)", "name": "Evap Out", "tdb": 22.8, "rh": 78.0, "color": "#059669"}
        ]
    else:
        st.session_state.nodes = [
            {"id": "P1", "name": "State 1", "tdb": 24.0, "rh": 50.0, "color": "#2563eb"},
            {"id": "P2", "name": "State 2", "tdb": 15.0, "rh": 85.0, "color": "#059669"}
        ]

st.sidebar.markdown("---")
st.sidebar.subheader("State Points Coordinate Editor")

for i, node in enumerate(st.session_state.nodes):
    with st.sidebar.expander(f"Point {node['id']} - {node['name']}", expanded=(i < 2)):
        node["tdb"] = st.slider(f"Dry-Bulb (°C)", -5.0, 45.0, float(node["tdb"]), 0.5, key=f"tdb_{i}")
        node["rh"] = st.slider(f"Rel. Humidity (%)", 5.0, 100.0, float(node["rh"]), 1.0, key=f"rh_{i}")

# -----------------------------------------------------------------------------
# 3. BUILD THERMODYNAMIC DATA TABLE
# -----------------------------------------------------------------------------
summary_list = []
for n in st.session_state.nodes:
    w = calc_w(n["tdb"], n["rh"], p_atm)
    h = calc_h(n["tdb"], w)
    v = calc_v(n["tdb"], w, p_atm)
    tdp = calc_tdp(n["tdb"], n["rh"])
    twb = calc_twb(n["tdb"], n["rh"])
    summary_list.append({
        "Point": n["id"],
        "Name": n["name"],
        "Tdb (°C)": round(n["tdb"], 1),
        "RH (%)": round(n["rh"], 1),
        "W (kg/kg)": round(w, 5),
        "h (kJ/kg)": round(h, 2),
        "v (m³/kg)": round(v, 3),
        "Tdp (°C)": round(tdp, 1),
        "Twb (°C)": round(twb, 1),
        "_w": w,
        "_h": h,
        "_color": n["color"]
    })

df_states = pd.DataFrame(summary_list)

# -----------------------------------------------------------------------------
# 4. PLOTLY PSYCHROMETRIC CHART WITH SENSITIVE HEAT FACTOR (SHF) PROTRACTOR
# -----------------------------------------------------------------------------
is_classic = (theme_mode == "Classic Engineering Green")
GRID_COLOR = "#22c55e" if is_classic else "#1e293b"
SAT_COLOR  = "#15803d" if is_classic else "#38bdf8"
BG_COLOR   = "#ffffff" if is_classic else "#0b0f17"
TEXT_COLOR = "#166534" if is_classic else "#94a3b8"

fig = go.Figure()

# A. Dry-Bulb Lines (Vertical)
for t in range(-5, 46, 5):
    fig.add_trace(go.Scatter(
        x=[t, t], y=[0, calc_w(t, 100.0, p_atm)],
        mode="lines",
        line=dict(color=GRID_COLOR, width=0.7 if t % 10 != 0 else 1.2),
        hoverinfo="skip", showlegend=False
    ))

# B. Moisture Content Lines (Horizontal)
for w_val in np.arange(0.001, 0.026, 0.001):
    # Find intersection temperature with 100% RH
    t_start = -5.0
    for t_test in np.linspace(-5, 45, 200):
        if calc_w(t_test, 100.0, p_atm) >= w_val:
            t_start = t_test
            break
    fig.add_trace(go.Scatter(
        x=[t_start, 45.0], y=[w_val, w_val],
        mode="lines",
        line=dict(color=GRID_COLOR, width=0.6 if round(w_val * 1000) % 5 != 0 else 1.1),
        hoverinfo="skip", showlegend=False
    ))

# C. Percentage Saturation Curves (10% to 100%)
t_span = np.linspace(-5, 45, 120)
for rh in range(10, 101, 10):
    w_curve = [calc_w(t, rh, p_atm) for t in t_span]
    valid = [w <= 0.0255 for w in w_curve]
    fig.add_trace(go.Scatter(
        x=t_span[valid], y=np.array(w_curve)[valid],
        mode="lines",
        line=dict(
            color=SAT_COLOR if rh == 100 else GRID_COLOR,
            width=2.5 if rh == 100 else 1.0
        ),
        name=f"{rh}% Saturation",
        hoverinfo="text",
        hovertext=f"Percentage Saturation: {rh}%",
        showlegend=(rh == 100)
    ))

# D. Wet-Bulb Sling Temperature Lines (Sloping Downward-Right)
for twb in range(0, 31, 5):
    t_db_range = np.linspace(twb, 45, 50)
    w_wb_line = [get_w_from_twb(tdb, twb, p_atm) for tdb in t_db_range]
    valid_wb = [(w >= 0) and (w <= 0.0255) for w in w_wb_line]
    fig.add_trace(go.Scatter(
        x=t_db_range[valid_wb], y=np.array(w_wb_line)[valid_wb],
        mode="lines",
        line=dict(color=GRID_COLOR, width=0.9, dash="dot"),
        hoverinfo="skip", showlegend=False
    ))

# E. Specific Volume Lines (0.80 to 0.92 m³/kg, steep diagonal lines)
for v_target in [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92]:
    t_v = []
    w_v = []
    for t_step in np.linspace(-5, 45, 50):
        # v = 0.287058*(T+273.15)*(1 + 1.6078*W)/P -> solve for W
        w_val = ((p_atm * v_target) / (0.287058 * (t_step + 273.15)) - 1.0) / 1.6078
        if 0 <= w_val <= 0.0255:
            t_v.append(t_step)
            w_v.append(w_val)
    if len(t_v) > 1:
        fig.add_trace(go.Scatter(
            x=t_v, y=w_v,
            mode="lines",
            line=dict(color=GRID_COLOR, width=1.1, dash="dash"),
            hoverinfo="skip", showlegend=False
        ))

# F. Sensible Heat Factor (SHF) Semicircular Protractor (Embedded in Upper Region)
shf_center_x = 18.0   # Center of SHF Protractor in °C
shf_center_y = 0.021  # Center height in kg/kg
shf_radius_x = 12.0
shf_radius_y = 0.0035

# Semicircle arc
theta = np.linspace(0, np.pi, 100)
arc_x = shf_center_x + shf_radius_x * np.cos(theta)
arc_y = shf_center_y + shf_radius_y * np.sin(theta)

fig.add_trace(go.Scatter(
    x=arc_x, y=arc_y,
    mode="lines",
    line=dict(color=SAT_COLOR, width=2.0),
    hoverinfo="skip", showlegend=False
))

# SHF Base Reference Line & Axis Tick Marks
fig.add_trace(go.Scatter(
    x=[shf_center_x - shf_radius_x, shf_center_x + shf_radius_x],
    y=[shf_center_y, shf_center_y],
    mode="lines",
    line=dict(color=SAT_COLOR, width=1.5),
    hoverinfo="skip", showlegend=False
))

# SHF Scale Marks & Labels
for shf_val in [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]:
    # SHF angle mapping across quadrant
    ang = np.arccos(np.clip((shf_val - 0.5) * 2.0, -1.0, 1.0))
    tx = shf_center_x + shf_radius_x * np.cos(ang)
    ty = shf_center_y + shf_radius_y * np.sin(ang)
    fig.add_trace(go.Scatter(
        x=[shf_center_x, tx], y=[shf_center_y, ty],
        mode="lines",
        line=dict(color=SAT_COLOR, width=0.8, dash="dot"),
        hoverinfo="skip", showlegend=False
    ))
    fig.add_annotation(
        x=tx, y=ty, text=f"<b>{shf_val}</b>",
        showarrow=False, font=dict(size=9, color=TEXT_COLOR),
        yshift=8 if ty >= shf_center_y else -8
    )

fig.add_annotation(
    x=shf_center_x, y=shf_center_y + shf_radius_y * 0.4,
    text="<b>SENSIBLE HEAT FACTOR</b>",
    showarrow=False, font=dict(size=10, color=TEXT_COLOR)
)

# G. Circuit Process Vectors & Connecting Lines
if len(df_states) > 1:
    fig.add_trace(go.Scatter(
        x=df_states["Tdb (°C)"],
        y=df_states["_w"],
        mode="lines",
        line=dict(color="#2563eb" if is_classic else "#38bdf8", width=3.0, dash="dash"),
        name="Circuit Path"
    ))

# H. State Point Markers
for _, row in df_states.iterrows():
    fig.add_trace(go.Scatter(
        x=[row["Tdb (°C)"]],
        y=[row["_w"]],
        mode="markers+text",
        marker=dict(size=13, color=row["_color"], line=dict(color="#ffffff", width=2)),
        text=[row["Point"]],
        textposition="top right",
        name=f"{row['Point']}: {row['Name']}",
        hovertemplate=(
            f"<b>Point {row['Point']} ({row['Name']})</b><br>"
            + "Dry-Bulb: %{x:.1f} °C<br>"
            + "Moisture Content: %{y:.5f} kg/kg<br>"
            + f"Rel. Saturation: {row['RH (%)']:.1f}%<br>"
            + f"Specific Enthalpy: {row['h (kJ/kg)']:.2f} kJ/kg<br>"
            + f"Specific Volume: {row['v (m³/kg)']:.3f} m³/kg<br>"
            + f"Dew Point: {row['Tdp (°C)']:.1f} °C<br>"
            + f"Wet Bulb: {row['Twb (°C)']:.1f} °C<extra></extra>"
        )
    ))

# I. Chart Layout & Coordinate Calibration
fig.update_layout(
    title=dict(
        text="<b>SINGAPORE POLYTECHNIC PSYCHROMETRIC CHART</b>",
        font=dict(color=TEXT_COLOR, size=18, family="sans-serif"),
        x=0.5, xanchor="center"
    ),
    xaxis=dict(
        title="<b>DRY BULB TEMPERATURE °C</b>",
        range=[-5, 45],
        dtick=5,
        gridcolor=GRID_COLOR,
        color=TEXT_COLOR,
        zeroline=False,
        ticks="outside"
    ),
    yaxis=dict(
        title="<b>MOISTURE CONTENT kg/kg (DRY AIR)</b>",
        range=[0, 0.025],
        dtick=0.005,
        side="right",
        gridcolor=GRID_COLOR,
        color=TEXT_COLOR,
        zeroline=False,
        ticks="outside"
    ),
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=BG_COLOR,
    height=640,
    margin=dict(l=40, r=70, t=70, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5)
)

# -----------------------------------------------------------------------------
# 5. PROCESS COIL CAPACITY, SHF & ADP SOLVER
# -----------------------------------------------------------------------------
rho_air = 1.204
mass_flow = airflow * rho_air

p_in = df_states.iloc[-2] if len(df_states) >= 2 else df_states.iloc[0]
p_out = df_states.iloc[-1]

delta_h = abs(p_in["_h"] - p_out["_h"])
delta_t = abs(p_in["Tdb (°C)"] - p_out["Tdb (°C)"])
delta_w = abs(p_in["_w"] - p_out["_w"])

q_total = mass_flow * delta_h
q_sensible = mass_flow * 1.006 * delta_t
q_latent = max(0.0, q_total - q_sensible)
shf_val = (q_sensible / q_total) if q_total > 0 else 1.0
condensate_rate = mass_flow * delta_w * 3600.0  # L/hr or kg/hr

# Apparatus Dew Point (ADP) approximation: extrapolate process line to 100% Saturation
adp_temp = p_out["Tdp (°C)"]

# -----------------------------------------------------------------------------
# 6. RENDER STREAMLIT INTERFACE
# -----------------------------------------------------------------------------
st.plotly_chart(fig, use_container_width=True)

col_metrics, col_summary = st.columns([1, 2])

with col_metrics:
    st.markdown("### ⚙️ Coil & Load Performance")
    c1, c2 = st.columns(2)
    c1.metric("Sensible Heat Factor (SHF)", f"{shf_val:.2f}")
    c2.metric("Apparatus Dew Point (ADP)", f"{adp_temp:.1f} °C")
    
    c3, c4 = st.columns(2)
    c3.metric("Total Coil Load", f"{q_total:.1f} kW")
    c4.metric("Sensible Coil Load", f"{q_sensible:.1f} kW")
    
    c5, c6 = st.columns(2)
    c5.metric("Latent Load", f"{q_latent:.1f} kW")
    c6.metric("Condensate Rate", f"{condensate_rate:.2f} kg/h")

with col_summary:
    st.markdown("### 📊 Thermodynamic State Point Readout")
    cols_display = ["Point", "Name", "Tdb (°C)", "RH (%)", "W (kg/kg)", "h (kJ/kg)", "v (m³/kg)", "Tdp (°C)", "Twb (°C)"]
    st.dataframe(df_states[cols_display], use_container_width=True, hide_index=True)
    
    csv_bytes = df_states[cols_display].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export State Table (CSV)",
        data=csv_bytes,
        file_name="psychrometric_chart_states.csv",
        mime="text/csv"
    )
