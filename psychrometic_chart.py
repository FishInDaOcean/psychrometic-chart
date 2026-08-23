import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Singapore Poly Psychrometric Solver | AHU Reheat",
    page_icon="📐",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Thermodynamic Calculations (Standard ASHRAE / CIBSE Formulations)
# -----------------------------------------------------------------------------
P_ATM = 101.325  # Standard Atmospheric Pressure (kPa)

def get_pws(t_c):
    """Saturation vapor pressure over liquid water (kPa)."""
    return 0.61078 * np.exp((17.27 * t_c) / (t_c + 237.3))

def calc_w(t_c, rh_pct, p_atm=P_ATM):
    """Moisture Content W (kg/kg dry air)."""
    pws = get_pws(t_c)
    pw = (np.clip(rh_pct, 1e-4, 100.0) / 100.0) * pws
    return (0.62198 * pw) / np.maximum(p_atm - pw, 1e-5)

def calc_rh(t_c, w_kg, p_atm=P_ATM):
    """Relative Humidity / Percentage Saturation (%)."""
    pw = (p_atm * w_kg) / (0.62198 + w_kg)
    pws = get_pws(t_c)
    return np.clip((pw / pws) * 100.0, 0.0, 100.0)

def calc_h(t_c, w_kg):
    """Specific Enthalpy h (kJ/kg dry air)."""
    return 1.006 * t_c + w_kg * (2501.0 + 1.86 * t_c)

def calc_v(t_c, w_kg, p_atm=P_ATM):
    """Specific Volume v (m³/kg dry air)."""
    return (0.287058 * (t_c + 273.15) * (1.0 + 1.6078 * w_kg)) / p_atm

def get_w_from_twb(t_db, t_wb, p_atm=P_ATM):
    """Moisture content along a wet-bulb line."""
    pws_wb = get_pws(t_wb)
    w_s_wb = (0.62198 * pws_wb) / (p_atm - pws_wb)
    return ((2501.0 - 2.381 * t_wb) * w_s_wb - 1.006 * (t_db - t_wb)) / (2501.0 + 1.86 * t_db - 4.186 * t_wb)

# -----------------------------------------------------------------------------
# Sidebar Parameter Controls
# -----------------------------------------------------------------------------
st.sidebar.title("📐 Question 1 Parameters")
st.sidebar.caption("Singapore Polytechnic / AHU Reheat Circuit")

col_sb1, col_sb2 = st.sidebar.columns(2)
t_r = col_sb1.number_input("Room Tdb (°C)", value=21.0, step=0.5)
rh_r = col_sb2.number_input("Room RH (%)", value=50.0, step=1.0)

t_o = col_sb1.number_input("Outdoor Tdb (°C)", value=28.0, step=0.5)
rh_o = col_sb2.number_input("Outdoor RH (%)", value=60.0, step=1.0)

t_adp = st.sidebar.number_input("Cooling Coil ADP (°C)", value=4.0, step=0.5)
rshf_orig = st.sidebar.slider("Initial Room RSHF", 0.3, 1.0, 0.75, 0.05)
m_o = st.sidebar.number_input("Outdoor Air Mass Flow (kg/s)", value=0.5, step=0.1)
oa_to_ra_ratio = st.sidebar.number_input("RA to OA Ratio (OA:RA = 1:X)", value=3.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("Part 2: Modified Load")
rshf_new = st.sidebar.slider("New RSHF (with Reheat)", 0.3, 1.0, 0.60, 0.05)

# -----------------------------------------------------------------------------
# Step-by-Step Solver Logic
# -----------------------------------------------------------------------------
w_r = calc_w(t_r, rh_r)
h_r = calc_h(t_r, w_r)

w_o = calc_w(t_o, rh_o)
h_o = calc_h(t_o, w_o)

w_adp = calc_w(t_adp, 100.0)
h_adp = calc_h(t_adp, w_adp)

# Mass flow balance
m_r = m_o * oa_to_ra_ratio
m_s = m_o + m_r
oa_frac = m_o / m_s

# a) Mixed Air (Point M)
t_m = (oa_frac * t_o) + ((1.0 - oa_frac) * t_r)
w_m = (oa_frac * w_o) + ((1.0 - oa_frac) * w_r)
h_m = calc_h(t_m, w_m)

# b) Off-Coil Supply Point (Point S)
slope_coil = (w_m - w_adp) / (t_m - t_adp)
slope_rshf_orig = (1.006 / 2501.0) * ((1.0 - rshf_orig) / rshf_orig)
t_s = (w_r - slope_rshf_orig * t_r - w_adp + slope_coil * t_adp) / (slope_coil - slope_rshf_orig)
w_s = w_adp + slope_coil * (t_s - t_adp)
h_s = calc_h(t_s, w_s)
v_s = calc_v(t_s, w_s)

# c) Supply Air Volume Flow Rate
v_dot_s = m_s * v_s

# d) Bypass Factor
bf = (t_s - t_adp) / (t_m - t_adp)

# e) Cooling Coil Load & f) Room Total Load
q_coil = m_s * (h_m - h_s)
q_room = m_s * (h_r - h_s)

# g) Reheat Supply Point (Point S') & h) New Mass Flow Rate
slope_rshf_mod = (1.006 / 2501.0) * ((1.0 - rshf_new) / rshf_new)
w_s_prime = w_s  # Sensible reheat keeps moisture ratio constant
t_s_prime = t_r - (w_r - w_s_prime) / slope_rshf_mod
h_s_prime = calc_h(t_s_prime, w_s_prime)
m_s_new = q_room / (h_r - h_s_prime)

# -----------------------------------------------------------------------------
# Psychrometric Chart Generator (Classic Green Engineering Style)
# -----------------------------------------------------------------------------
GRID_COLOR = "#22c55e"
SAT_COLOR  = "#15803d"
BG_COLOR   = "#ffffff"
TEXT_COLOR = "#166534"

fig = go.Figure()

# Vertical Dry-Bulb grid lines
for t in range(-5, 46, 5):
    fig.add_trace(go.Scatter(
        x=[t, t], y=[0, calc_w(t, 100.0)],
        mode="lines",
        line=dict(color=GRID_COLOR, width=0.7 if t % 10 != 0 else 1.2),
        hoverinfo="skip", showlegend=False
    ))

# Horizontal Moisture Content grid lines
for w_val in np.arange(0.001, 0.026, 0.001):
    t_st = -5.0
    for t_test in np.linspace(-5, 45, 100):
        if calc_w(t_test, 100.0) >= w_val:
            t_st = t_test
            break
    fig.add_trace(go.Scatter(
        x=[t_st, 45.0], y=[w_val, w_val],
        mode="lines",
        line=dict(color=GRID_COLOR, width=0.6 if round(w_val * 1000) % 5 != 0 else 1.1),
        hoverinfo="skip", showlegend=False
    ))

# Relative Humidity / Saturation Curves
t_span = np.linspace(-5, 45, 120)
for rh in range(10, 101, 10):
    w_curve = [calc_w(t, rh) for t in t_span]
    valid = [w <= 0.0255 for w in w_curve]
    fig.add_trace(go.Scatter(
        x=t_span[valid], y=np.array(w_curve)[valid],
        mode="lines",
        line=dict(color=SAT_COLOR if rh == 100 else GRID_COLOR, width=2.5 if rh == 100 else 1.0),
        hoverinfo="text", hovertext=f"{rh}% Saturation", showlegend=False
    ))

# Wet-Bulb Sling lines
for twb in range(0, 31, 5):
    t_db_range = np.linspace(twb, 45, 50)
    w_wb = [get_w_from_twb(tdb, twb) for tdb in t_db_range]
    valid_wb = [(w >= 0) and (w <= 0.0255) for w in w_wb]
    fig.add_trace(go.Scatter(
        x=t_db_range[valid_wb], y=np.array(w_wb)[valid_wb],
        mode="lines", line=dict(color=GRID_COLOR, width=0.8, dash="dot"),
        hoverinfo="skip", showlegend=False
    ))

# Specific Volume lines
for v_val in [0.80, 0.82, 0.84, 0.86, 0.88, 0.90]:
    t_v, w_v = [], []
    for t_step in np.linspace(-5, 45, 40):
        w_res = ((P_ATM * v_val) / (0.287058 * (t_step + 273.15)) - 1.0) / 1.6078
        if 0 <= w_res <= 0.0255:
            t_v.append(t_step)
            w_v.append(w_res)
    if len(t_v) > 1:
        fig.add_trace(go.Scatter(
            x=t_v, y=w_v, mode="lines",
            line=dict(color=GRID_COLOR, width=1.0, dash="dash"),
            hoverinfo="skip", showlegend=False
        ))

# SHF Protractor Semicircle Overlay
shf_cx, shf_cy, shf_rx, shf_ry = 18.0, 0.021, 12.0, 0.0035
theta = np.linspace(0, np.pi, 60)
fig.add_trace(go.Scatter(
    x=shf_cx + shf_rx * np.cos(theta), y=shf_cy + shf_ry * np.sin(theta),
    mode="lines", line=dict(color=SAT_COLOR, width=2.0), hoverinfo="skip", showlegend=False
))
fig.add_trace(go.Scatter(
    x=[shf_cx - shf_rx, shf_cx + shf_rx], y=[shf_cy, shf_cy],
    mode="lines", line=dict(color=SAT_COLOR, width=1.5), hoverinfo="skip", showlegend=False
))
for shf_val in [0.4, 0.6, 0.75, 0.9, 1.0]:
    ang = np.arccos(np.clip((shf_val - 0.5) * 2.0, -1.0, 1.0))
    fig.add_trace(go.Scatter(
        x=[shf_cx, shf_cx + shf_rx * np.cos(ang)],
        y=[shf_cy, shf_cy + shf_ry * np.sin(ang)],
        mode="lines", line=dict(color=SAT_COLOR, width=0.8, dash="dot"),
        hoverinfo="skip", showlegend=False
    ))

# --- Process Vectors ---
# 1. Mixing Line (O - R)
fig.add_trace(go.Scatter(
    x=[t_o, t_r], y=[w_o, w_r], mode="lines",
    line=dict(color="#d97706", width=2.5), name="Mixing (O-R)"
))

# 2. Cooling Coil Line (M - ADP)
fig.add_trace(go.Scatter(
    x=[t_m, t_adp], y=[w_m, w_adp], mode="lines",
    line=dict(color="#0284c7", width=2.5), name="Cooling Coil (M-ADP)"
))

# 3. Initial Room Line (S - R)
fig.add_trace(go.Scatter(
    x=[t_s, t_r], y=[w_s, w_r], mode="lines",
    line=dict(color="#dc2626", width=2.5, dash="dash"), name=f"Room Line (RSHF={rshf_orig})"
))

# 4. Reheat Coil Line (S - S')
fig.add_trace(go.Scatter(
    x=[t_s, t_s_prime], y=[w_s, w_s_prime], mode="lines",
    line=dict(color="#9333ea", width=3.0), name="Reheat Coil (S -> S')"
))

# 5. Modified Room Line (S' - R)
fig.add_trace(go.Scatter(
    x=[t_s_prime, t_r], y=[w_s_prime, w_r], mode="lines",
    line=dict(color="#ea580c", width=2.5, dash="dot"), name=f"Modified Room (RSHF={rshf_new})"
))

# State Point Markers
points = [
    ("O", t_o, w_o, "#dc2626", "Outdoor Air"),
    ("R", t_r, w_r, "#2563eb", "Return Room Air"),
    ("M", t_m, w_m, "#d97706", "Mixed Air"),
    ("ADP", t_adp, w_adp, "#059669", "Apparatus Dew Point"),
    ("S", t_s, w_s, "#0284c7", "Off-Coil Supply"),
    ("S'", t_s_prime, w_s_prime, "#9333ea", "Reheat Supply")
]

for tag, px, py, pcol, pname in points:
    fig.add_trace(go.Scatter(
        x=[px], y=[py], mode="markers+text",
        marker=dict(size=12, color=pcol, line=dict(color="#ffffff", width=1.5)),
        text=[tag], textposition="top right",
        name=f"{tag} - {pname}", hoverinfo="text",
        hovertext=f"<b>Point {tag} ({pname})</b><br>Tdb: {px:.2f}°C<br>W: {py*1000:.2f} g/kg"
    ))

fig.update_layout(
    title=dict(
        text="<b>SINGAPORE POLYTECHNIC PSYCHROMETRIC CHART - AHU REHEAT PROCESS</b>",
        font=dict(color=TEXT_COLOR, size=16), x=0.5, xanchor="center"
    ),
    xaxis=dict(title="<b>DRY BULB TEMPERATURE °C</b>", range=[-5, 45], dtick=5, gridcolor=GRID_COLOR, color=TEXT_COLOR),
    yaxis=dict(title="<b>MOISTURE CONTENT kg/kg (DRY AIR)</b>", range=[0, 0.025], dtick=0.005, side="right", gridcolor=GRID_COLOR, color=TEXT_COLOR),
    paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR, height=620,
    margin=dict(l=40, r=70, t=60, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5)
)

# -----------------------------------------------------------------------------
# Interface Layout & Outputs
# -----------------------------------------------------------------------------
st.title("⚡ Singapore Poly AHU Exam Question Solver")

st.plotly_chart(fig, use_container_width=True)

st.markdown("### 📝 Exam Solution Output & Calculated Results")

# Formatted Metric Blocks (Clean string format to prevent escape sequence issues)
delta_temp_reheat = t_s_prime - t_s
delta_mass_flow = m_s_new - m_s

c1, c2, c3 = st.columns(3)
c1.metric("a) Mixed Air Tdb (T_M)", f"{t_m:.2f} °C", f"W_M = {w_m*1000:.2f} g/kg")
c2.metric("b) Supply Air Tdb (T_S)", f"{t_s:.2f} °C", f"W_S = {w_s*1000:.2f} g/kg")
c3.metric("c) Supply Volume Flow (V_S)", f"{v_dot_s:.2f} m³/s", f"v_S = {v_s:.3f} m³/kg")

c4, c5, c6 = st.columns(3)
c4.metric("d) Coil Bypass Factor (BF)", f"{bf:.3f}", f"{(bf*100):.1f}%")
c5.metric("e) Cooling Coil Load (Q_coil)", f"{q_coil:.2f} kW")
c6.metric("f) Room Total Heat Load (Q_room)", f"{q_room:.2f} kW")

c7, c8, c9 = st.columns(3)
c7.metric("g) New Supply Tdb (T_S')", f"{t_s_prime:.2f} °C", f"Reheat ΔT = +{delta_temp_reheat:.2f} °C")
c8.metric("h) New Supply Mass Flow (m_S,new)", f"{m_s_new:.3f} kg/s", f"Δm_dot = +{delta_mass_flow:.2f} kg/s")
c9.metric("i) Process State Summary", "All 6 points plotted")

# Full Tabular Summary
df_sol = pd.DataFrame([
    {"State": "O (Outdoor Air)", "Tdb (°C)": t_o, "RH (%)": rh_o, "W (g/kg)": w_o*1000, "h (kJ/kg)": h_o},
    {"State": "R (Return Air)", "Tdb (°C)": t_r, "RH (%)": rh_r, "W (g/kg)": w_r*1000, "h (kJ/kg)": h_r},
    {"State": "M (Mixed Air)", "Tdb (°C)": t_m, "RH (%)": calc_rh(t_m, w_m), "W (g/kg)": w_m*1000, "h (kJ/kg)": h_m},
    {"State": "ADP (Apparatus Dew Point)", "Tdb (°C)": t_adp, "RH (%)": 100.0, "W (g/kg)": w_adp*1000, "h (kJ/kg)": h_adp},
    {"State": "S (Off-Coil Supply)", "Tdb (°C)": t_s, "RH (%)": calc_rh(t_s, w_s), "W (g/kg)": w_s*1000, "h (kJ/kg)": h_s},
    {"State": "S' (Reheat Supply)", "Tdb (°C)": t_s_prime, "RH (%)": calc_rh(t_s_prime, w_s_prime), "W (g/kg)": w_s_prime*1000, "h (kJ/kg)": h_s_prime},
])
st.dataframe(df_sol.round(2), use_container_width=True, hide_index=True)
