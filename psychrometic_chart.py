import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Psychrometric Circuit Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# ASHRAE Thermodynamic Calculations (Standard Formulations)
# ---------------------------------------------------------
def get_pws(t_c):
    """Saturation vapor pressure over liquid water (kPa) via Magnus-Tetens."""
    return 0.61078 * np.exp((17.27 * t_c) / (t_c + 237.3))

def calc_humidity_ratio(t_c, rh_pct, p_atm):
    """Humidity ratio W (g/kg dry air)."""
    pws = get_pws(t_c)
    pw = (rh_pct / 100.0) * pws
    w_kg = (0.62198 * pw) / np.maximum(p_atm - pw, 1e-5)
    return w_kg * 1000.0

def calc_relative_humidity(t_c, w_g_kg, p_atm):
    """Relative humidity (%) from Dry-Bulb and W (g/kg)."""
    w_kg = w_g_kg / 1000.0
    pw = (p_atm * w_kg) / (0.62198 + w_kg)
    pws = get_pws(t_c)
    return np.clip((pw / pws) * 100.0, 0.0, 100.0)

def calc_enthalpy(t_c, w_g_kg):
    """Specific enthalpy h (kJ/kg dry air)."""
    w_kg = w_g_kg / 1000.0
    return 1.006 * t_c + w_kg * (2501.0 + 1.86 * t_c)

def calc_dew_point(t_c, rh_pct):
    """Dew point temperature (°C)."""
    a, b = 17.27, 237.3
    rh_clamped = np.clip(rh_pct / 100.0, 1e-4, 1.0)
    alpha = ((a * t_c) / (b + t_c)) + np.log(rh_clamped)
    return (b * alpha) / (a - alpha)

def calc_wet_bulb(t_c, rh_pct):
    """Wet-bulb temperature (°C) via Stull's empirical equation."""
    rh = rh_pct
    tw = (
        t_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return tw

# ---------------------------------------------------------
# Sidebar Controls & Circuit Presets
# ---------------------------------------------------------
st.sidebar.title("⚡ Circuit Controls")

p_atm = st.sidebar.slider("Barometric Pressure (kPa)", min_value=70.0, max_value=110.0, value=101.325, step=0.5)
airflow = st.sidebar.slider("Airflow Rate (m³/s)", min_value=0.2, max_value=20.0, value=2.5, step=0.1)

preset_option = st.sidebar.selectbox(
    "Active HVAC Circuit Preset",
    [
        "Summer Cooling & Dehumidification",
        "Winter Heating & Steam Humidification",
        "Direct Evaporative Cooling (DEC)",
        "Custom Multi-Point Circuit"
    ]
)

# Preset State Initialization
if "points" not in st.session_state or st.sidebar.button("Reset Circuit to Preset"):
    if preset_option == "Summer Cooling & Dehumidification":
        st.session_state.points = [
            {"Point": "OA", "Name": "Outdoor Air", "Tdb": 35.0, "RH": 60.0, "Color": "#ef4444"},
            {"Point": "RA", "Name": "Return Air", "Tdb": 24.0, "RH": 50.0, "Color": "#3b82f6"},
            {"Point": "MA", "Name": "Mixed Air", "Tdb": 27.3, "RH": 53.0, "Color": "#f59e0b"},
            {"Point": "CC", "Name": "Off-Coil", "Tdb": 12.5, "RH": 92.0, "Color": "#10b981"}
        ]
    elif preset_option == "Winter Heating & Steam Humidification":
        st.session_state.points = [
            {"Point": "OA", "Name": "Outdoor Air", "Tdb": 2.0, "RH": 80.0, "Color": "#3b82f6"},
            {"Point": "HC", "Name": "Pre-Heat Coil", "Tdb": 18.0, "RH": 28.0, "Color": "#ef4444"},
            {"Point": "HUM", "Name": "Humidified Air", "Tdb": 19.5, "RH": 55.0, "Color": "#10b981"}
        ]
    elif preset_option == "Direct Evaporative Cooling (DEC)":
        st.session_state.points = [
            {"Point": "IN", "Name": "Hot Ambient", "Tdb": 38.0, "RH": 20.0, "Color": "#ef4444"},
            {"Point": "OUT", "Name": "Evaporative Out", "Tdb": 22.5, "RH": 78.0, "Color": "#10b981"}
        ]
    else:
        st.session_state.points = [
            {"Point": "P1", "Name": "State 1", "Tdb": 20.0, "RH": 50.0, "Color": "#3b82f6"},
            {"Point": "P2", "Name": "State 2", "Tdb": 30.0, "RH": 40.0, "Color": "#ef4444"}
        ]

st.sidebar.subheader("State Node Coordinates")
for idx, p in enumerate(st.session_state.points):
    with st.sidebar.expander(f"{p['Point']}: {p['Name']}", expanded=(idx < 2)):
        p["Tdb"] = st.slider(f"Dry-Bulb (°C) [{p['Point']}]", -10.0, 50.0, float(p["Tdb"]), 0.5, key=f"tdb_{idx}")
        p["RH"] = st.slider(f"Rel. Humidity (%) [{p['Point']}]", 5.0, 100.0, float(p["RH"]), 1.0, key=f"rh_{idx}")

# ---------------------------------------------------------
# Thermodynamic Calculations for Table
# ---------------------------------------------------------
summary_records = []
for p in st.session_state.points:
    w = calc_humidity_ratio(p["Tdb"], p["RH"], p_atm)
    h = calc_enthalpy(p["Tdb"], w)
    tdp = calc_dew_point(p["Tdb"], p["RH"])
    twb = calc_wet_bulb(p["Tdb"], p["RH"])
    summary_records.append({
        "Point": p["Point"],
        "Name": p["Name"],
        "Tdb (°C)": round(p["Tdb"], 1),
        "RH (%)": round(p["RH"], 1),
        "W (g/kg)": round(w, 2),
        "h (kJ/kg)": round(h, 1),
        "Tdp (°C)": round(tdp, 1),
        "Twb (°C)": round(twb, 1),
        "_w": w,
        "_h": h,
        "_color": p["Color"]
    })

df_summary = pd.DataFrame(summary_records)

# ---------------------------------------------------------
# Plotly Psychrometric Chart Engine
# ---------------------------------------------------------
fig = go.Figure()

# Background RH Curves
t_axis = np.linspace(-10, 50, 120)
for rh in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    w_curve = [calc_humidity_ratio(t, rh, p_atm) for t in t_axis]
    valid = [w <= 30.0 for w in w_curve]
    t_valid = t_axis[valid]
    w_valid = np.array(w_curve)[valid]

    fig.add_trace(go.Scatter(
        x=t_valid,
        y=w_valid,
        mode="lines",
        line=dict(
            color="#38bdf8" if rh == 100 else "#262f3d",
            width=2.5 if rh == 100 else 1.0,
            dash="solid" if rh == 100 else "dash"
        ),
        hoverinfo="text",
        hovertext=f"RH = {rh}%",
        showlegend=False
    ))

# Constant Enthalpy / Wet-Bulb Guide Lines
for h_target in range(10, 130, 20):
    t_iso = []
    w_iso = []
    for t_step in np.linspace(-10, 50, 60):
        # h = 1.006*T + W/1000*(2501 + 1.86*T) -> solve for W
        w_val = 1000.0 * (h_target - 1.006 * t_step) / (2501.0 + 1.86 * t_step)
        if 0 <= w_val <= 30:
            rh_check = calc_relative_humidity(t_step, w_val, p_atm)
            if rh_check <= 100:
                t_iso.append(t_step)
                w_iso.append(w_val)
    if t_iso:
        fig.add_trace(go.Scatter(
            x=t_iso,
            y=w_iso,
            mode="lines",
            line=dict(color="#1f2937", width=1.0, dash="dot"),
            hoverinfo="skip",
            showlegend=False
        ))

# Circuit Connecting Vectors
if len(df_summary) > 1:
    fig.add_trace(go.Scatter(
        x=df_summary["Tdb (°C)"],
        y=df_summary["W (g/kg)"],
        mode="lines",
        line=dict(color="#38bdf8", width=2.5, dash="dash"),
        name="Process Vector",
        hoverinfo="skip"
    ))

# State Point Markers
for _, row in df_summary.iterrows():
    fig.add_trace(go.Scatter(
        x=[row["Tdb (°C)"]],
        y=[row["W (g/kg)"]],
        mode="markers+text",
        marker=dict(size=12, color=row["_color"], line=dict(color="#ffffff", width=1.5)),
        text=[row["Point"]],
        textposition="top right",
        name=f"{row['Point']}: {row['Name']}",
        hovertemplate=(
            f"<b>{row['Point']} - {row['Name']}</b><br>"
            + "Dry-Bulb: %{x:.1f} °C<br>"
            + "Humidity Ratio: %{y:.2f} g/kg<br>"
            + f"Rel. Humidity: {row['RH (%)']:.1f}%<br>"
            + f"Enthalpy: {row['h (kJ/kg)']:.1f} kJ/kg<br>"
            + f"Dew Point: {row['Tdp (°C)']:.1f} °C<extra></extra>"
        )
    ))

fig.update_layout(
    title=dict(
        text="<b>Psychrometric Process Circuit</b> (Sea Level, P = {:.3f} kPa)".format(p_atm),
        font=dict(color="#ffffff", size=16)
    ),
    xaxis=dict(
        title="Dry-Bulb Temperature (°C)",
        range=[-10, 50],
        dtick=5,
        gridcolor="#1f242c",
        zerolinecolor="#30363d"
    ),
    yaxis=dict(
        title="Humidity Ratio W (g/kg dry air)",
        range=[0, 30],
        dtick=5,
        side="right",
        gridcolor="#1f242c",
        zerolinecolor="#30363d"
    ),
    paper_bgcolor="#161b22",
    plot_bgcolor="#161b22",
    font=dict(color="#8b949e", family="monospace"),
    margin=dict(l=40, r=60, t=60, b=40),
    height=580,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# ---------------------------------------------------------
# Coil Load & SHR Calculation
# ---------------------------------------------------------
rho_air = 1.204  # Standard air density (kg/m3)
m_dot = airflow * rho_air  # Dry air mass flow (kg/s)

p_start = df_summary.iloc[-2] if len(df_summary) >= 2 else df_summary.iloc[0]
p_end = df_summary.iloc[-1]

delta_h = abs(p_start["_h"] - p_end["_h"])
delta_t = abs(p_start["Tdb (°C)"] - p_end["Tdb (°C)"])

q_total = m_dot * delta_h
q_sensible = m_dot * 1.006 * delta_t
q_latent = max(0.0, q_total - q_sensible)
shr = (q_sensible / q_total) if q_total > 0 else 1.0

# ---------------------------------------------------------
# Layout Rendering
# ---------------------------------------------------------
st.title("⚡ Psychrometric Circuit Studio")
st.caption("Interactive Carrier-style psychrometric chart and HVAC circuit solver.")

col_chart, col_loads = st.columns([3, 1])

with col_chart:
    st.plotly_chart(fig, use_container_width=True)

with col_loads:
    st.markdown("### Process Capacity")
    st.metric("Total Load", f"{q_total:.1f} kW")
    st.metric("Sensible Load", f"{q_sensible:.1f} kW")
    st.metric("Latent Load", f"{q_latent:.1f} kW")
    st.metric("Sensible Heat Ratio (SHR)", f"{shr:.2f}")

st.markdown("### 📊 Thermodynamic State Summary")
display_cols = ["Point", "Name", "Tdb (°C)", "RH (%)", "W (g/kg)", "h (kJ/kg)", "Tdp (°C)", "Twb (°C)"]
st.dataframe(df_summary[display_cols], use_container_width=True, hide_index=True)

csv_data = df_summary[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Circuit Data (CSV)",
    data=csv_data,
    file_name="psychrometric_circuit_data.csv",
    mime="text/csv"
)