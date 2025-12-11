import streamlit as st
import pandas as pd
import numpy as np
import json
from dataclasses import asdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. SETUP & DATA
# ==========================================
import simulation_engine as sim_engine

# ==========================================
# 1. SETUP & SESSION STATE
# ==========================================
st.set_page_config(page_title="Algae Plant Dashboard", layout="wide", initial_sidebar_state="expanded")

if 'sim_results' not in st.session_state:
    st.session_state['sim_results'] = None

# ==========================================
# 2. SIDEBAR - PARAMETERS
# ==========================================
st.sidebar.title("⚙️ Plant Configuration")

# A. Panel Geometry
st.sidebar.subheader("Panel Config")
p_width = st.sidebar.number_input("Width (m)", 0.5, 2.0, 1.0, 0.01)
p_length = st.sidebar.number_input("Length (m)", 0.5, 2.0, 1.0, 0.01)
p_thick = st.sidebar.number_input("Thickness (m)", 0.01, 0.5, 0.1, 0.01)
pivot_z = st.sidebar.slider("Pivot Offset Z (m)", -0.5, 0.5, 0.2, 0.01)

# B. Frame Layout
st.sidebar.subheader("Frame Layout")
pitch_x = st.sidebar.number_input("Panel Pitch X (m)", 0.5, 2.0, 1.05, 0.01)
pitch_y = st.sidebar.number_input("Panel Pitch Y (m)", 0.5, 2.0, 1.05, 0.01)
ball_dia = st.sidebar.slider("Vis: Pivot Ball Size", 0.01, 0.3, 0.1)

# C. Frame Tilts (Row Specific)
st.sidebar.subheader("Row Tilts (Degrees)")
tilt_r0 = st.sidebar.slider("Row 0 (North)", 0.0, 45.0, 0.0, 1.0)
tilt_r1 = st.sidebar.slider("Row 1 (Center)", 0.0, 45.0, 0.0, 1.0)
tilt_r2 = st.sidebar.slider("Row 2 (South)", 0.0, 45.0, 0.0, 1.0)
frame_tilts = [tilt_r0, tilt_r1, tilt_r2]

# D. Global
frame_rot = 5.0 # Fixed per spec

# E. Actions
run_sim = st.sidebar.button("🚀 Run Simulation (Full Year)", type="primary")

# ==========================================
# 3. RUN LOGIC
# ==========================================
if run_sim:
    with st.spinner("Running Physics Engine..."):
        # Create Config
        geo = sim_engine.PanelGeometry(
            width=p_width, length=p_length, thickness=p_thick,
            pitch_x=pitch_x, pitch_y=pitch_y,
            pivot_offset=(0.0, 0.0, pivot_z) 
        )
        
        # Init Simulator
        sim = sim_engine.ClashSimulator(
            geo=geo, 
            lat=-25.86, lon=26.90, 
            frame_rot=frame_rot,
            frame_tilts=frame_tilts
        )
        
        # Run
        # Check for DNI file
        dni_file = "solar irradiation_Koster.xlsx"
        df_res, _ = sim.run_year_power(excel_path=dni_file, frequency='6min')
        
        st.session_state['sim_results'] = df_res
        st.session_state['sim_geo'] = geo
        st.success("Simulation Complete!")

# Load Data
raw_df = st.session_state['sim_results']
if raw_df is None:
    # Try generic load or prompt
    st.info("👈 Configure parameters and click 'Run Simulation' to start.")
    # Attempt legacy load if available?
    try:
        raw_df = pd.read_csv("simulation_results.csv")
        # Ensure compatible columns exist
        if 'Power_Total_kW' not in raw_df.columns:
            st.warning("Old data detected (missing Power metrics). Please Re-Run.")
        raw_df['Time'] = pd.to_datetime(raw_df['Time'])
        st.session_state['sim_results'] = raw_df
        # Default mock geo
        st.session_state['sim_geo'] = sim_engine.PanelGeometry() 
    except:
        st.stop()
    
geo_conf = asdict(st.session_state.get('sim_geo', sim_engine.PanelGeometry()))
pivot_offset = geo_conf['pivot_offset']

# Clean up DNI / Power cols if missing (legacy file support)
if 'Power_Total_kW' not in raw_df.columns:
    raw_df['Power_Total_kW'] = 0.0


# ==========================================
# 2. GEOMETRY ENGINE (Optimized)
# ==========================================
def get_rotation_matrix(az_true, el, rot_deg):
    # Vectorized rotation calculation would be faster, but per-frame is okay for 100 frames
    az_rel = az_true - rot_deg
    theta = np.radians(90 - az_rel)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    tilt = np.radians(90 - el)
    ct, st = np.cos(tilt), np.sin(tilt)
    Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    return Rz @ Rx

def get_panel_traces(row, ball_dia, visible=True):
    """
    Returns a list of Trace objects for a single timestep.
    This is called for the 'Base' frame and every 'Animation' frame.
    """
    az, el = row['Azimuth'], row['Elevation']
    R = get_rotation_matrix(az, el, frame_rot)
    
    # Common Geometry
    w, l, t = geo_conf['width'], geo_conf['length'], geo_conf['thickness']
    ox, oy, oz = pivot_offset
    dx, dy, dz = w/2, l/2, t/2
    
    # Base Box (centered at geometric center)
    c_local = np.array([
        [dx, dy, dz], [dx, -dy, dz], [-dx, -dy, dz], [-dx, dy, dz],
        [dx, dy, -dz], [dx, -dy, -dz], [-dx, -dy, -dz], [-dx, dy, -dz]
    ])
    # Shift by pivot
    c_shifted = c_local + np.array([ox, oy, oz])
    
    # We merge all 4 panels into SINGLE meshes to reduce draw calls
    # This drastically improves performance compared to 4 separate mesh objects
    pitch_x, pitch_y = geo_conf['pitch_x'], geo_conf['pitch_y']
    indices = [(0,0), (0,1), (1,0), (1,1)]
    
    # Arrays to hold combined geometry
    all_x, all_y, all_z = [], [], [] # For Body
    all_xg, all_yg, all_zg = [], [], [] # For Glass
    all_xl, all_yl, all_zl = [], [], [] # For Wireframe
    
    for ix, iy in indices:
        cx, cy = ix * pitch_x, iy * pitch_y
        
        # Rotate & Translate
        c_world = (R @ c_shifted.T).T + np.array([cx, cy, 0])
        x, y, z = c_world[:,0], c_world[:,1], c_world[:,2]
        
        # Accumulate Body
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)
        
        # Accumulate Glass (Top face 0-3)
        all_xg.append(x[0:4])
        all_yg.append(y[0:4])
        all_zg.append(z[0:4])
        
        # Accumulate Wireframe (lines)
        lines_idx = [0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7]
        # Add None to break lines between panels
        wx = list(x[lines_idx]) + [None]
        wy = list(y[lines_idx]) + [None]
        wz = list(z[lines_idx]) + [None]
        all_xl.extend(wx)
        all_yl.extend(wy)
        all_zl.extend(wz)

    # Flatten arrays
    flat_x = np.concatenate(all_x)
    flat_y = np.concatenate(all_y)
    flat_z = np.concatenate(all_z)
    
    flat_xg = np.concatenate(all_xg)
    flat_yg = np.concatenate(all_yg)
    flat_zg = np.concatenate(all_zg)

    # Construct Indices for Mesh3d (This is tricky for combined meshes)
    # We have 4 panels. Each has 8 vertices.
    # Panel 0 verts: 0-7. Panel 1 verts: 8-15...
    # We must offset the standard cube indices by 8*p
    i_template = np.array([0, 0, 4, 4, 0, 0, 3, 3, 1, 1, 2, 2])
    j_template = np.array([1, 3, 5, 7, 4, 1, 7, 2, 5, 2, 6, 7])
    k_template = np.array([3, 2, 7, 6, 1, 5, 2, 6, 2, 6, 5, 6])
    
    i_body, j_body, k_body = [], [], []
    i_glass, j_glass, k_glass = [], [], []
    
    # Glass template (2 triangles)
    ig_temp = np.array([0, 0])
    jg_temp = np.array([1, 2])
    kg_temp = np.array([2, 3])

    for p in range(4):
        offset_8 = p * 8
        offset_4 = p * 4 # Glass only has 4 verts per panel in the flat_xg array
        
        i_body.append(i_template + offset_8)
        j_body.append(j_template + offset_8)
        k_body.append(k_template + offset_8)
        
        i_glass.append(ig_temp + offset_4)
        j_glass.append(jg_temp + offset_4)
        k_glass.append(kg_temp + offset_4)

    # VISUAL STATUS
    is_clash = row.get('Clash', False)
    body_color = 'red' if is_clash else 'lightgray'
    glass_color = 'darkred' if is_clash else 'dodgerblue'
    glass_opacity = 0.5 if is_clash else 0.9

    # 1. Body Mesh Trace
    trace_body = go.Mesh3d(
        x=flat_x, y=flat_y, z=flat_z,
        i=np.concatenate(i_body), j=np.concatenate(j_body), k=np.concatenate(k_body),
        color=body_color, opacity=1.0, flatshading=True, showlegend=False, visible=visible
    )
    
    # 2. Glass Mesh Trace
    trace_glass = go.Mesh3d(
        x=flat_xg, y=flat_yg, z=flat_zg,
        i=np.concatenate(i_glass), j=np.concatenate(j_glass), k=np.concatenate(k_glass),
        color=glass_color, opacity=glass_opacity, flatshading=True, showlegend=False, visible=visible
    )
    
    # 3. Wireframe Trace
    trace_wire = go.Scatter3d(
        x=all_xl, y=all_yl, z=all_zl,
        mode='lines', line=dict(color='black', width=3), showlegend=False, visible=visible
    )

    return [trace_body, trace_glass, trace_wire]

def build_animation_figure(day_df, ball_dia):
    # --- A. STATIC ELEMENTS (North Arrow, Pivot Balls) ---
    pitch_x, pitch_y = geo_conf['pitch_x'], geo_conf['pitch_y']
    
    # North Arrow
    # Plant Axis Y is 5 deg East of True North.
    # So True North is -5 deg relative to Y.
    # Angle from Y-axis (Up) is -5 deg (Left/Counter-Clockwise).
    # Sine is X (Left/Right), Cosine is Y (Up/Down).
    # sin(-5) = -0.08 (Left), cos(-5) = 0.99 (Up).
    rad_n = np.radians(-frame_rot)
    nv = np.array([np.sin(rad_n), np.cos(rad_n), 0]) * (pitch_y * 1.5)
    north_trace = go.Scatter3d(
        x=[0, nv[0]], y=[0, nv[1]], z=[0, 0],
        mode='lines+text', line=dict(color='darkred', width=6),
        text=["", "N"], textfont=dict(size=15, color='darkred'),
        name='North'
    )
    
    # Sun Path Trace (Validation)
    # Convert Az/El of the day to 3D arc
    sp_az = np.radians(day_df['Azimuth'] - frame_rot) # Plant-relative Az
    sp_el = np.radians(day_df['Elevation'])
    # Dome Radius
    r_dome = pitch_y * 2.5
    
    # Standard Conversion for Vis (Z up, Y North)
    # Using the same Azimuth convention as North Arrow (Angle from Y)
    # Azimuth 0 is North (Y). 90 is East (X).
    # X = R cosEl sinAz
    # Y = R cosEl cosAz
    # Z = R sinEl
    sp_x = r_dome * np.cos(sp_el) * np.sin(sp_az) 
    sp_y = r_dome * np.cos(sp_el) * np.cos(sp_az)
    sp_z = r_dome * np.sin(sp_el)
    
    # Filter visible sun
    mask = sp_z > 0
    sun_trace = go.Scatter3d(
        x=sp_x[mask], y=sp_y[mask], z=sp_z[mask],
        mode='lines', line=dict(color='orange', width=4, dash='dot'),
        name='Sun Path', hoverinfo='skip'
    )
    
    # Pivot Balls (Static)
    bx, by = [], []
    indices = [(0,0), (0,1), (1,0), (1,1)]
    for ix, iy in indices:
        bx.append(ix * pitch_x)
        by.append(iy * pitch_y)
        
    balls_trace = go.Scatter3d(
        x=bx, y=by, z=[0]*4,
        mode='markers',
        marker=dict(size=ball_dia*100, color='red', symbol='circle'),
        showlegend=False, hoverinfo='none'
    )
    
    # --- B. INITIAL FRAME DATA ---
    # Get data for the FIRST timestamp
    first_row = day_df.iloc[0]
    initial_panel_traces = get_panel_traces(first_row, ball_dia)
    
    # Combine all traces
    # Order: [North, Balls, Body, Glass, Wire]
    data = [north_trace, balls_trace, sun_trace] + initial_panel_traces

    # --- C. FRAMES (The Animation) ---
    frames = []
    # Loop through all times in the day
    for idx, row in day_df.iterrows():
        t_str = row['Time'].strftime("%H:%M")
        
        # Get new geometry for this time
        # We ONLY need to return the traces that change (Body, Glass, Wire)
        # These correspond to indices 2, 3, 4 in the 'data' list.
        new_traces = get_panel_traces(row, ball_dia)
        
        # Dynamic Frame Title: Power and Loss
        cur_pwr = row['Power_Actual_kW']
        is_clash = row['Clash']
        
        # HTML styled title for the frame
        if is_clash:
            title_text = f"Time: {t_str} | Power: {cur_pwr:.1f} kW <span style='color:red; font-weight:bold;'>(-50% Stow Loss)</span>"
        else:
            title_text = f"Time: {t_str} | Power: {cur_pwr:.1f} kW"

        frame = go.Frame(
            data=new_traces, # This replaces data[3], data[4], data[5]
            name=t_str,
            traces=[3, 4, 5], # Tell Plotly which traces to update
            layout=go.Layout(title_text=title_text)
        )
        frames.append(frame)

    # --- D. LAYOUT WITH SLIDER & BUTTONS ---
    center_x = pitch_x / 2
    center_y = pitch_y / 2
    
    fig = go.Figure(data=data, frames=frames)
    
    fig.update_layout(
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.6), # Matches approx range ratio (3m : 3m : 2m)
            xaxis=dict(range=[center_x-1.8, center_x+1.8], visible=False),
            yaxis=dict(range=[center_y-1.8, center_y+1.8], visible=False),
            zaxis=dict(range=[-1, 1], visible=False),
            bgcolor='white'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        showlegend=False,
        # ANIMATION CONTROLS
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.1, "y": 0,
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
                }
            ]
        }],
        sliders=[{
            "steps": [
                {
                    "method": "animate",
                    "args": [[f.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                    "label": f.name
                } for f in frames
            ],
            "currentvalue": {"prefix": "Time: ", "font": {"size": 20}},
            "y": 0, "x": 0.3, "len": 0.7
        }]
    )
    
    return fig

# ==========================================
# 3. APP LAYOUT
# ==========================================


# --- LEFT COLUMN: TABLE ---
col_list, col_main = st.columns([1, 3])

with col_list:
    st.subheader("📅 Select Date")
    raw_df['Date'] = raw_df['Time'].dt.date
    daily = raw_df.groupby('Date').agg(
        Clash_Count=('Clash', 'sum'),
        Energy_Potential=('Power_Potential_kW', 'sum'),
        Energy_Actual=('Power_Actual_kW', 'sum')
    ).reset_index()
    daily['Has Clash'] = daily['Clash_Count'] > 0
    
    # Calculate Loss %
    # Avoid division by zero
    daily['Loss_kWh'] = daily['Energy_Potential'] - daily['Energy_Actual']
    daily['Loss %'] = 0.0
    mask = daily['Energy_Potential'] > 0
    daily.loc[mask, 'Loss %'] = (daily.loc[mask, 'Loss_kWh'] / daily.loc[mask, 'Energy_Potential']) * 100
    
    # Format for display
    daily['Energy (kWh)'] = daily['Energy_Actual'].round(1)
    daily['Loss %'] = daily['Loss %'].round(1)
    
    show_clash_only = st.checkbox("Clash Days Only", value=True)
    table_df = daily[daily['Has Clash']] if show_clash_only else daily
        
    selected = st.dataframe(
        table_df[['Date', 'Has Clash', 'Energy (kWh)', 'Loss %']],
        selection_mode='single-row', on_select='rerun',
        hide_index=True, use_container_width=True, height=700
    )
    
    if not table_df.empty:
        if len(selected.selection['rows']) > 0:
            sel_date = table_df.iloc[selected.selection['rows'][0]]['Date']
        else:
            sel_date = table_df.iloc[0]['Date']
    else:
        st.warning("No data found matching criteria. If 'Clash Days Only' is checked, try unchecking it.")
        st.stop()

# --- RIGHT COLUMN: NATIVE ANIMATION ---
with col_main:
    # Get Data
    day_df = raw_df[raw_df['Date'] == sel_date].copy().sort_values('Time').reset_index(drop=True)
    
    # Get Daily Loss %
    loss_val = 0.0
    daily_rec = daily[daily['Date'] == sel_date]
    if not daily_rec.empty:
        loss_val = daily_rec.iloc[0]['Loss %']
    
    # Color red if significant loss
    loss_str = f":red[- {loss_val}% Loss]" if loss_val > 0 else f"- {loss_val}% Loss"
    st.subheader(f"Animation: {sel_date} | {loss_str}")
    
    with st.spinner("Building Smooth Animation... (Please Wait)"):
        # We build ONE figure with all frames embedded
        fig = build_animation_figure(day_df, ball_dia)
        st.plotly_chart(fig, use_container_width=True)
        
    # Stats for the Day
    clash_times = day_df[day_df['Clash']]
    if not clash_times.empty:
        st.error(f"⚠️ CLASH DETECTED between {clash_times['Time'].dt.time.min()} and {clash_times['Time'].dt.time.max()}")
    else:
        st.success("✅ No clashes on this day.")

    # 2D Chart (Static for context)
    fig_chart = make_subplots(specs=[[{"secondary_y": True}]])
    fig_chart.add_trace(go.Scatter(x=day_df['Time'], y=day_df['Elevation'], name="Elevation", fill='tozeroy', line=dict(color='orange')), secondary_y=True)
    fig_chart.add_trace(go.Scatter(x=day_df['Time'], y=day_df['Azimuth'], name="Azimuth", line=dict(color='gray', dash='dot')), secondary_y=False)
    
    clashes = day_df[day_df['Clash'] == True]
    if not clashes.empty:
        fig_chart.add_trace(go.Scatter(x=clashes['Time'], y=clashes['Elevation'], mode='markers', marker=dict(color='red'), name="Clash"), secondary_y=True)

    fig_chart.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_chart, use_container_width=True)

    # Power Chart with Loss
    st.subheader("⚡ Power Output")
    fig_p = go.Figure()
    # Potential (Background)
    fig_p.add_trace(go.Scatter(x=day_df['Time'], y=day_df['Power_Potential_kW'], fill='tozeroy', name="Potential (No Stow)", line=dict(color='lightgreen', width=1, dash='dot')))
    # Actual (Foreground)
    fig_p.add_trace(go.Scatter(x=day_df['Time'], y=day_df['Power_Actual_kW'], fill='tozeroy', name="Actual (Grid)", line=dict(color='green')))
    
    fig_p.update_layout(height=250, margin=dict(t=20, b=0, l=0, r=0), yaxis_title="Plant Power (kW)")
    st.plotly_chart(fig_p, use_container_width=True)