import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import sys
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# --- DYNAMIC PATH RESOLUTION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- IMPORT USER PACKAGE ---
try:
    from mcgillthz.twodim_analysis import THzExp, subtract_leak_NL, window_2d
    from mcgillthz.fft_utils import do_fft_all_taus
except ImportError as e:
    st.error(f"Import Error: {e}\nEnsure your folder structure matches: parent/mcgillthz/ and parent/GUIs/")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="2D THz Analysis", layout="wide")
st.title("2D THz Spectroscopy Analysis (Interactive)")

# --- INITIALIZE PARAMETER STATES ---
KEYS_TO_SAVE =[
    'fft_window_type', 'fft_min_time', 'fft_max_time', 'fft_pad_p2',
    'leak_tau_min', 'leak_tau_max', 'leak_t_min', 'leak_t_max',
    'win_w_type', 'win_tukey_alpha', 'win_min_time', 'win_max_time',
    'win_flat_tau_start', 'win_flat_tau_end', 'win_min_tau', 'win_max_tau',
    'win_sub_baseline', 'win_t_base_min', 'win_t_base_max',
    'fft2d_Nt', 'fft2d_Ntau'
]

static_defaults = {
    'fft_window_type': 'hann',
    'fft_min_time': -6.0,
    'fft_max_time': 6.0,
    'fft_pad_p2': 1,
    'leak_tau_min': -2.0,
    'leak_tau_max': 0.0,
    'leak_t_min': -2.0,
    'leak_t_max': 1.0,
    'win_w_type': 'tukey',
    'win_tukey_alpha': 0.5,
    'win_min_time': -9.0,
    'win_max_time': 10.0,
    'win_flat_tau_start': -0.5,
    'win_flat_tau_end': 1.5,
    'win_min_tau': -1.5,
    'win_max_tau': 2.5,
    'win_sub_baseline': False,
    'win_t_base_min': -2.0,
    'win_t_base_max': -1.0,
    'fft2d_Nt': 1024,
    'fft2d_Ntau': 128
}

for k, v in static_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if 'scan' not in st.session_state: st.session_state.scan = None
if 'base_name' not in st.session_state: st.session_state.base_name = ""
if 'fft_A' not in st.session_state: st.session_state.fft_A = None
if 'fft_B' not in st.session_state: st.session_state.fft_B = None
if 'cs_lines' not in st.session_state: st.session_state.cs_lines =[]


# --- HELPER FUNCTIONS ---
def select_file():
    script = """
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.attributes('-topmost', True)
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select a 2D Data File (e.g. 212K_AB.csv)",
    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
)
print(file_path)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    return result.stdout.strip()

def get_plotly_cmap(cmap_name):
    cmap_lower = cmap_name.lower()
    if cmap_lower == 'bwr': return 'RdBu_r'
    if cmap_lower == 'icefire': return 'IceFire'
    
    try:
        cmap = plt.get_cmap(cmap_lower)
        colors =[]
        for i in range(256):
            rgba = cmap(i / 255.0)
            colors.append([i / 255.0, f'rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})'])
        return colors
    except ValueError:
        mapping = {'hot': 'Hot', 'jet': 'Jet', 'blackbody': 'Blackbody'}
        return mapping.get(cmap_lower, cmap_lower)

def add_cs_line(m, b, name, is_vertical=False, x_val=0.0):
    colors =['cyan', 'lime', 'magenta', 'yellow', 'orange', 'white']
    color = colors[len(st.session_state.cs_lines) % len(colors)]
    st.session_state.cs_lines.append({
        'm': m, 'b': b, 'name': name, 'color': color, 
        'is_vertical': is_vertical, 'x_val': x_val
    })

# --- INTERACTIVE PLOTLY RENDERERS ---
def plotly_2x2_grid(scan, title, min_t=None, max_t=None, vmax=0.1):
    fig = make_subplots(rows=2, cols=2, subplot_titles=['AB', 'Nonlinear', 'A', 'B'], 
                        shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.1)
    dfs =[scan.AB, scan.NL, scan.A, scan.B]
    positions =[(1,1), (1,2), (2,1), (2,2)]
    t = dfs[0]['time'].values
    tau = dfs[0].columns[1:].astype(float)

    for i, df in enumerate(dfs):
        data = df.drop(columns='time').values.T
        mask = (t >= min_t) & (t <= max_t) if (min_t and max_t) else np.ones_like(t, dtype=bool)
        vmax_abs = np.max(np.abs(data[:, mask]))
        current_vmax = vmax if i == 1 else vmax_abs

        fig.add_trace(go.Heatmap(
            z=data, x=t, y=tau,
            colorscale='RdBu_r', zmin=-current_vmax, zmax=current_vmax,
            coloraxis="coloraxis" if i != 1 else "coloraxis2",
            hovertemplate="Time: %{x:.2f} ps<br>Delay: %{y:.2f} ps<br>Field: %{z:.4f}<extra></extra>"
        ), row=positions[i][0], col=positions[i][1])

    fig.update_layout(title_text=title, height=750, template="plotly_white",
        coloraxis=dict(colorscale='RdBu_r', colorbar_x=0.45, colorbar_title="A/B/AB (a.u.)"),
        coloraxis2=dict(colorscale='RdBu_r', cmin=-vmax, cmax=vmax, colorbar_x=1.0, colorbar_title="NL (a.u.)"))
    if min_t and max_t: fig.update_xaxes(range=[min_t, max_t])
    fig.update_xaxes(title_text="THz time t (ps)", row=2)
    fig.update_yaxes(title_text="Excitation delay τ (ps)", col=1)
    return fig

def plotly_spectrum(data_B, fft_B, data_A, fft_A, tau_val, uirevision, log_scale_y):
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Time Domain (τ = {tau_val} ps)", f"Spectrum (τ = {tau_val} ps)"])
    fig.add_trace(go.Scatter(x=data_B[0], y=data_B[1], name='B', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=fft_B[0], y=fft_B[1], name='B FFT', line=dict(color='blue'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=data_A[0], y=data_A[1], name='A', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=fft_A[0], y=fft_A[1], name='A FFT', line=dict(color='red'), showlegend=False), row=1, col=2)

    fig.update_xaxes(title_text="Time (ps)", row=1, col=1)
    fig.update_yaxes(title_text="Electric Field (a.u.)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (THz)", range=[0, 6], row=1, col=2)
    fig.update_yaxes(title_text="FFT Amplitude (a.u.)", type="log" if log_scale_y else "linear", row=1, col=2)
    fig.update_layout(height=500, hovermode="x unified", uirevision=uirevision, template="plotly_white")
    return fig

def plotly_2d_map_simple(df, title, min_t=None, max_t=None, vmin=None, vmax=None):
    t = df['time'].values
    tau = df.columns[1:].astype(float)
    data = df.drop(columns='time').values.T

    if vmin is None or vmax is None:
        mask = (t >= min_t) & (t <= max_t) if (min_t and max_t) else np.ones_like(t, dtype=bool)
        vmax_abs = np.max(np.abs(data[:, mask]))
        if vmin is None: vmin = -vmax_abs
        if vmax is None: vmax = vmax_abs

    fig = go.Figure(data=go.Heatmap(
        z=data, x=t, y=tau,
        colorscale='RdBu_r', zmin=vmin, zmax=vmax,
        hovertemplate="Time: %{x:.2f} ps<br>Delay: %{y:.2f} ps<br>Field: %{z:.4f}<extra></extra>"
    ))
    fig.update_layout(title=title, xaxis_title="THz time t (ps)", yaxis_title="Excitation delay τ (ps)", height=550, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
    if min_t and max_t: fig.update_xaxes(range=[min_t, max_t])
    return fig

def plotly_2d_fft(ft_data, freq_t, freq_tau, title, vmin=0, vmax=3, cmap='magma', log_scale=False, value_to_plot='Amplitude', plot_guidelines=True, cs_lines=None, x_max=3.0, y_max=None):
    if value_to_plot == 'Amplitude': z_data = np.abs(ft_data)
    elif value_to_plot == 'Real': z_data = np.real(ft_data)
    elif value_to_plot == 'Imaginary': z_data = np.imag(ft_data)
    elif value_to_plot == 'Phase': z_data = np.angle(ft_data)

    if log_scale and value_to_plot != 'Phase':
        z_data = np.where(np.abs(z_data) > 0, np.log10(np.abs(z_data)), vmin)

    fig = go.Figure(data=go.Heatmap(
        z=z_data, x=freq_t, y=freq_tau,
        colorscale=get_plotly_cmap(cmap), zmin=vmin, zmax=vmax,
        hovertemplate="ν_t: %{x:.2f} THz<br>ν_τ: %{y:.2f} THz<br>Value: %{z:.4f}<extra></extra>"
    ))
    
    if plot_guidelines:
        line_kws = dict(color='white', dash='dash', width=1.5)
        # Extended lines to cover negative range
        fig.add_trace(go.Scatter(x=[0, 10], y=[0, 0], mode='lines', line=line_kws, hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=[0, 10], y=[0, 10], mode='lines', line=line_kws, hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=[0, 10], y=[0, -10], mode='lines', line=line_kws, hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=[0, 10], y=[0, 20], mode='lines', line=line_kws, hoverinfo='skip', showlegend=False))

    if cs_lines is not None:
        t_min, t_max_data = 0.0, min(10.0, freq_t.max())
        tau_min, tau_max = freq_tau.min(), freq_tau.max()
        for line in cs_lines:
            if line.get('is_vertical', False):
                x_v = line['x_val']
                if t_min <= x_v <= t_max_data:
                    fig.add_trace(go.Scatter(x=[x_v, x_v], y=[tau_min, tau_max], mode='lines', line=dict(color=line['color'], width=3), name=line['name']))
            else:
                vt = np.linspace(t_min, t_max_data, 500)
                vtau = line['m'] * vt + line['b']
                mask = (vtau >= tau_min) & (vtau <= tau_max)
                if np.any(mask):
                    fig.add_trace(go.Scatter(x=vt[mask], y=vtau[mask], mode='lines', line=dict(color=line['color'], width=3), name=line['name']))

    fig.update_layout(title=title, xaxis_title="Probe frequency ν_t (THz)", yaxis_title="Excitation frequency ν_τ (THz)", height=650 if not cs_lines else 450, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
    
    # Dynamic Y-Axis limit based on data if not explicit
    if y_max is None:
        y_max = freq_tau.max()
        
    fig.update_xaxes(range=[0, x_max])
    fig.update_yaxes(range=[-y_max, y_max]) # Symmetrical range from -Max to +Max
    return fig

def plotly_time_freq_map(fft_df, title, cmap, log_scale=False, value_to_plot='Amplitude', vmin=None, vmax=None, x_max=3.0):
    freq_t = fft_df['freq'].values
    taus = fft_df.columns[1:].astype(float)
    z_data_complex = fft_df.drop(columns='freq').values.T

    if value_to_plot == 'Amplitude': z_data = np.abs(z_data_complex)
    elif value_to_plot == 'Real': z_data = np.real(z_data_complex)
    elif value_to_plot == 'Imaginary': z_data = np.imag(z_data_complex)
    elif value_to_plot == 'Phase': z_data = np.angle(z_data_complex)

    if log_scale and value_to_plot != 'Phase':
        z_data = np.where(np.abs(z_data) > 0, np.log10(np.abs(z_data)), np.nan)

    fig = go.Figure(data=go.Heatmap(
        z=z_data, x=freq_t, y=taus,
        colorscale=get_plotly_cmap(cmap), zmin=vmin, zmax=vmax,
        hovertemplate="ν_t: %{x:.2f} THz<br>τ: %{y:.2f} ps<br>Value: %{z:.4f}<extra></extra>"
    ))
    
    fig.update_layout(title=title, xaxis_title="Probe frequency ν_t (THz)", yaxis_title="Excitation delay τ (ps)", height=600, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
    fig.update_xaxes(range=[0, x_max])
    return fig

def plotly_cross_sections(scan, cs_lines, x_axis_type, y_axis_type):
    fig = go.Figure()
    if not cs_lines:
        fig.update_layout(title="No lines added. Use controls to add a cross section.", template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
        return fig
        
    interp_r = RegularGridInterpolator((scan.freq_tau, scan.freq_t), np.real(scan.ft_data), bounds_error=False, fill_value=np.nan)
    interp_i = RegularGridInterpolator((scan.freq_tau, scan.freq_t), np.imag(scan.ft_data), bounds_error=False, fill_value=np.nan)
    
    t_min, t_max = 0.0, min(3.0, scan.freq_t.max())
    tau_min, tau_max = scan.freq_tau.min(), scan.freq_tau.max()
    
    for line in cs_lines:
        is_vert = line.get('is_vertical', False)
        if is_vert and x_axis_type == "Probe Frequency (ν_t)": continue
            
        if is_vert:
            vtau_arr = np.linspace(tau_min, tau_max, 500)
            vt_arr = np.full_like(vtau_arr, line['x_val'])
        else:
            vt_full = np.linspace(t_min, t_max, 500)
            vtau_full = line['m'] * vt_full + line['b']
            mask = (vtau_full >= tau_min) & (vtau_full <= tau_max)
            if not np.any(mask): continue
            vt_arr = vt_full[mask]
            vtau_arr = vtau_full[mask]
        
        pts = np.column_stack((vtau_arr, vt_arr))
        z_r = interp_r(pts)
        z_i = interp_i(pts)
        z_comp = z_r + 1j * z_i
        
        if y_axis_type == 'Amplitude': y_vals = np.abs(z_comp)
        elif y_axis_type == 'Real': y_vals = np.real(z_comp)
        elif y_axis_type == 'Imaginary': y_vals = np.imag(z_comp)
        elif y_axis_type == 'Phase': y_vals = np.angle(z_comp)
        
        if x_axis_type == "Probe Frequency (ν_t)":
            x_vals, x_label = vt_arr, "Probe Frequency ν_t (THz)"
        else:
            x_vals, x_label = vtau_arr, "Excitation Frequency ν_τ (THz)"
            
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, name=line['name'], line=dict(color=line['color'], width=2)))
        
    fig.update_layout(title=f"1D Cross Sections ({y_axis_type})", xaxis_title=x_label, yaxis_title=f"{y_axis_type} (a.u.)", height=450, template="plotly_white", hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0))
    return fig

# --- LAYOUT: TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Import & 2x2 Grid", 
    "2. Individual FFTs", 
    "3. Subtract Leak NL", 
    "4. Windowing", 
    "5. 2D FFT",
    "6. Cross Sections"
])

# ==========================================
# TAB 1: Import & 2x2 Grid
# ==========================================
with tab1:
    st.header("1. Import Data")
    col_btn, col_path = st.columns([1, 4])
    with col_btn:
        if st.button("📁 Browse Data File", width="stretch"):
            chosen_path = select_file()
            if chosen_path:
                base_dir = os.path.dirname(chosen_path)
                filename = os.path.basename(chosen_path)
                for suffix in['_AB.csv', '_A.csv', '_B.csv', '_NL.csv']:
                    if filename.endswith(suffix):
                        filename = filename.replace(suffix, "")
                        break
                st.session_state.base_name = os.path.join(base_dir, filename)
                st.rerun()

    with col_path:
        st.text_input("Base Experiment Path", value=st.session_state.base_name, disabled=True)
    
    st.subheader("Grid Plot Parameters")
    c1, c2, c3 = st.columns(3)
    min_t = c1.number_input("Min t (ps)", value=-9.0, step=1.0)
    max_t = c2.number_input("Max t (ps)", value=10.0, step=1.0)
    vmax_grid = c3.number_input("vmax (NL Plot)", value=0.1, step=0.01)
    
    if st.button("Load and Plot Grid", type="primary", width="stretch"):
        if not st.session_state.base_name:
            st.error("Please select a valid base file path first.")
        else:
            base_dir = os.path.dirname(st.session_state.base_name)
            calib_path = os.path.join(base_dir, "calibration.csv")
            config_path = os.path.join(base_dir, "analysis_config.json")
            
            with st.spinner("Loading data..."):
                if os.path.exists(calib_path):
                    st.success(f"Calibration file found: `{calib_path}`")
                    scan = THzExp(st.session_state.base_name, calibration_file=calib_path)
                else:
                    st.warning("No calibration.csv found in directory. Proceeding without calibration.")
                    scan = THzExp(st.session_state.base_name, calibration_file=None)
                st.session_state.scan = scan

                with st.expander("📄 Data File Header (Metadata)", expanded=True):
                    st.json(scan.metadata)

                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            cfg = json.load(f)
                        for k, v in cfg.items():
                            st.session_state[k] = v
                        st.success("Loaded saved parameters from `analysis_config.json`.")
                    except Exception as e:
                        st.warning(f"Error loading config: {e}")
                else:
                    taus = scan.NL.columns[1:].astype(float)
                    ts = scan.NL['time'].values
                    max_tau, min_tau = float(taus.max()), float(taus.min())
                    max_t, min_t = float(ts.max()), float(ts.min())
                    
                    st.session_state['fft_min_time'] = min_t
                    st.session_state['fft_max_time'] = max_t
                    st.session_state['leak_tau_max'] = max_tau
                    st.session_state['leak_tau_min'] = max(min_tau, max_tau - 2.0)
                    
                    st.session_state['win_min_time'] = min_t
                    st.session_state['win_max_time'] = max_t
                    st.session_state['win_min_tau'] = min_tau
                    st.session_state['win_max_tau'] = max_tau
                    st.session_state['win_flat_tau_end'] = max_tau - 1.0
                    
                    st.info("No config found. Dynamic defaults applied from dataset.")
                
            with st.spinner("Rendering Interactive Grid..."):
                fig = plotly_2x2_grid(scan, title=os.path.basename(st.session_state.base_name), 
                                      min_t=min_t, max_t=max_t, vmax=vmax_grid)
                st.plotly_chart(fig, width="stretch")

# ==========================================
# TAB 2: Individual FFTs
# ==========================================
with tab2:
    st.header("2. Individual FFTs (A & B)")
    if st.session_state.scan is None:
        st.info("Please load data in Tab 1 first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        win_opts =['hann', 'hamming', 'tukey', 'boxcar']
        idx = win_opts.index(st.session_state['fft_window_type']) if st.session_state['fft_window_type'] in win_opts else 0
        st.session_state['fft_window_type'] = c1.selectbox("Window", options=win_opts, index=idx)
        
        st.session_state['fft_min_time'] = c2.number_input("Min Time (ps)", value=float(st.session_state['fft_min_time']))
        st.session_state['fft_max_time'] = c3.number_input("Max Time (ps)", value=float(st.session_state['fft_max_time']))
        st.session_state['fft_pad_p2'] = c4.number_input("Padding (Power of 2)", value=int(st.session_state['fft_pad_p2']), step=1)
        
        if st.button("Compute Individual FFTs", width="stretch"):
            with st.spinner("Computing..."):
                scan = st.session_state.scan
                fft_A, _ = do_fft_all_taus(scan.A, window=st.session_state['fft_window_type'], 
                                           min_time=st.session_state['fft_min_time'], 
                                           max_time=st.session_state['fft_max_time'], 
                                           pad_power2=st.session_state['fft_pad_p2'])
                fft_B, _ = do_fft_all_taus(scan.B, window=st.session_state['fft_window_type'], 
                                           min_time=st.session_state['fft_min_time'], 
                                           max_time=st.session_state['fft_max_time'], 
                                           pad_power2=st.session_state['fft_pad_p2'])
                st.session_state.fft_A = fft_A
                st.session_state.fft_B = fft_B
                st.success("FFTs Computed successfully!")

        if st.session_state.fft_A is not None and st.session_state.fft_B is not None:
            st.divider()
            scan = st.session_state.scan
            taus = scan.A.columns[1:].astype(float)
            
            c_opts1, c_opts2 = st.columns(2)
            fixed_zoom = c_opts1.checkbox("🔒 Keep Zoom/Pan fixed when sliding", value=False)
            log_scale_y = c_opts2.checkbox("Show FFT Spectra in Log Scale", value=True)
            
            selected_tau = st.select_slider("Select Excitation Delay τ (ps)", options=taus, value=taus[0])
            col_match =[c for c in scan.A.columns[1:] if abs(float(c) - selected_tau) < 1e-5][0]
            
            data_B_arr = np.array([scan.B['time'], scan.B[col_match]])
            fft_B_arr = np.array([st.session_state.fft_B['freq'], st.session_state.fft_B[col_match]])
            data_A_arr = np.array([scan.A['time'], scan.A[col_match]])
            fft_A_arr = np.array([st.session_state.fft_A['freq'], st.session_state.fft_A[col_match]])
            
            uirev_val = "fixed" if fixed_zoom else selected_tau
            fig = plotly_spectrum(data_B_arr, fft_B_arr, data_A_arr, fft_A_arr, selected_tau, uirev_val, log_scale_y)
            st.plotly_chart(fig, width="stretch")

# ==========================================
# TAB 3: Subtract Leak NL
# ==========================================
with tab3:
    st.header("3. Subtract Leaked Channel")
    if st.session_state.scan is None:
        st.info("Please load data in Tab 1 first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        st.session_state['leak_tau_min'] = c1.number_input("tau_min", value=float(st.session_state['leak_tau_min']), step=0.1)
        st.session_state['leak_tau_max'] = c2.number_input("tau_max", value=float(st.session_state['leak_tau_max']), step=0.1)
        st.session_state['leak_t_min'] = c3.number_input("t_min", value=float(st.session_state['leak_t_min']), step=0.1)
        st.session_state['leak_t_max'] = c4.number_input("t_max", value=float(st.session_state['leak_t_max']), step=0.1)
        
        if st.button("Subtract Leak", width="stretch"):
            scan = st.session_state.scan
            with st.spinner("Processing Leak Subtraction..."):
                scan.no_bg = subtract_leak_NL(scan.NL, st.session_state['leak_tau_min'], st.session_state['leak_tau_max'], 
                                              st.session_state['leak_t_min'], st.session_state['leak_t_max'])
                st.session_state.scan = scan
                st.success("Leak subtracted successfully!")
                
        colA, colB, colS = st.columns([4.5, 4.5, 1])
        scan_t3 = st.session_state.scan
        max_val_t3 = float(np.max(np.abs(scan_t3.NL.drop(columns='time').values)))
        if hasattr(scan_t3, 'no_bg'):
            max_val_t3 = max(max_val_t3, float(np.max(np.abs(scan_t3.no_bg.drop(columns='time').values))))
            
        with colS:
            t3_vmax = st.number_input("vmax", value=max_val_t3, step=0.01, format="%.4f", key='t3_vmax')
            t3_vmin = st.number_input("vmin", value=-t3_vmax, step=0.01, format="%.4f", key='t3_vmin', disabled=True)

        with colA:
            # vmin rigorously locked to -vmax
            fig1 = plotly_2d_map_simple(scan_t3.NL, "Before: Raw NL", vmin=-t3_vmax, vmax=t3_vmax)
            tm = st.session_state['leak_t_min']
            tM = st.session_state['leak_t_max']
            taum = st.session_state['leak_tau_min']
            tauM = st.session_state['leak_tau_max']
            x_corners =[tm - taum, tM - taum, tM - tauM, tm - tauM, tm - taum]
            y_corners =[taum, taum, tauM, tauM, taum]
            fig1.add_trace(go.Scatter(x=x_corners, y=y_corners, mode='lines', line=dict(color='black', width=2, dash='dash'), name='Averaging Region', hoverinfo='skip'))
            st.plotly_chart(fig1, width="stretch")
            
        with colB:
            if hasattr(scan_t3, 'no_bg'):
                fig2 = plotly_2d_map_simple(scan_t3.no_bg, "After: Leak Subtracted NL", vmin=-t3_vmax, vmax=t3_vmax)
                st.plotly_chart(fig2, width="stretch")
            else:
                st.info("Click 'Subtract Leak' to generate the corrected map.")

# ==========================================
# TAB 4: Windowing
# ==========================================
with tab4:
    st.header("4. Window the Data")
    if st.session_state.scan is None or not hasattr(st.session_state.scan, 'no_bg'):
        st.info("Please complete Tab 3 (Subtract Leak) first.")
    else:
        st.subheader("Probe Time (t) Parameters")
        c1, c2, c3, c4 = st.columns(4)
        win_opts =['tukey', 'hann', 'hamming']
        w_idx = win_opts.index(st.session_state['win_w_type']) if st.session_state['win_w_type'] in win_opts else 0
        st.session_state['win_w_type'] = c1.selectbox("Time Window Type", options=win_opts, index=w_idx)
        st.session_state['win_tukey_alpha'] = c2.number_input("Tukey Alpha", value=float(st.session_state['win_tukey_alpha']), step=0.1)
        st.session_state['win_min_time'] = c3.number_input("min_time (ps)", value=float(st.session_state['win_min_time']))
        st.session_state['win_max_time'] = c4.number_input("max_time (ps)", value=float(st.session_state['win_max_time']))
        
        t_window_param = (st.session_state['win_w_type'], st.session_state['win_tukey_alpha']) if st.session_state['win_w_type'] == 'tukey' else st.session_state['win_w_type']
        
        st.subheader("Excitation Delay (τ) Parameters")
        c5, c6, c7, c8 = st.columns(4)
        st.session_state['win_flat_tau_start'] = c5.number_input("flat_tau_start", value=float(st.session_state['win_flat_tau_start']))
        st.session_state['win_flat_tau_end'] = c6.number_input("flat_tau_end", value=float(st.session_state['win_flat_tau_end']))
        st.session_state['win_min_tau'] = c7.number_input("min_tau", value=float(st.session_state['win_min_tau']))
        st.session_state['win_max_tau'] = c8.number_input("max_tau", value=float(st.session_state['win_max_tau']))

        st.subheader("Baseline Subtraction")
        cb1, cb2, cb3 = st.columns([2, 1.5, 1.5])
        st.session_state['win_sub_baseline'] = cb1.checkbox("Subtract Baseline at each Delay", value=bool(st.session_state['win_sub_baseline']))
        st.session_state['win_t_base_min'] = cb2.number_input("t_base_min (ps)", value=float(st.session_state['win_t_base_min']), step=0.1)
        st.session_state['win_t_base_max'] = cb3.number_input("t_base_max (ps)", value=float(st.session_state['win_t_base_max']), step=0.1)

        if st.button("Apply Window", width="stretch"):
            scan = st.session_state.scan
            with st.spinner("Applying window and baseline correction..."):
                scan.windowed = window_2d(scan.no_bg, t_window=t_window_param, 
                                          flat_tau_start=st.session_state['win_flat_tau_start'], 
                                          flat_tau_end=st.session_state['win_flat_tau_end'], 
                                          min_tau=st.session_state['win_min_tau'], max_tau=st.session_state['win_max_tau'],
                                          min_time=st.session_state['win_min_time'], max_time=st.session_state['win_max_time'],
                                          subtract_baseline=st.session_state['win_sub_baseline'], 
                                          t_base_min=st.session_state['win_t_base_min'], t_base_max=st.session_state['win_t_base_max'])
                st.session_state.scan = scan
                st.success("Window and baseline operations applied successfully!")

        if hasattr(st.session_state.scan, 'windowed'):
            colA, colB, colS = st.columns([4.5, 4.5, 1])
            scan_t4 = st.session_state.scan
            max_val_t4 = float(np.max(np.abs(scan_t4.no_bg.drop(columns='time').values)))
            max_val_t4 = max(max_val_t4, float(np.max(np.abs(scan_t4.windowed.drop(columns='time').values))))
            
            with colS:
                t4_vmax = st.number_input("vmax", value=max_val_t4, step=0.01, format="%.4f", key='t4_vmax')
                t4_vmin = st.number_input("vmin", value=-t4_vmax, step=0.01, format="%.4f", key='t4_vmin', disabled=True)

            with colA:
                fig1 = plotly_2d_map_simple(scan_t4.no_bg, "Before: Leak Subtracted NL", vmin=-t4_vmax, vmax=t4_vmax)
                t_b_min = st.session_state['win_t_base_min']
                t_b_max = st.session_state['win_t_base_max']
                taus_t4 = scan_t4.no_bg.columns[1:].astype(float)
                tau_m, tau_M = taus_t4.min(), taus_t4.max()
                x_base =[t_b_min, t_b_max, t_b_max, t_b_min, t_b_min]
                y_base =[tau_m, tau_m, tau_M, tau_M, tau_m]
                fig1.add_trace(go.Scatter(x=x_base, y=y_base, mode='lines', 
                                          line=dict(color='black', width=1.5, dash='dash'), 
                                          name='Baseline Region', hoverinfo='skip'))
                st.plotly_chart(fig1, width="stretch")
            with colB:
                fig2 = plotly_2d_map_simple(scan_t4.windowed, "After: Windowed NL", vmin=-t4_vmax, vmax=t4_vmax)
                st.plotly_chart(fig2, width="stretch")

# ==========================================
# TAB 5: 2D FFT
# ==========================================
with tab5:
    st.header("5. 2D FFT")
    if st.session_state.scan is None or not hasattr(st.session_state.scan, 'windowed'):
        st.info("Please complete Tab 4 (Windowing) first.")
    else:
        col_controls, col_plot = st.columns([1, 3])
        
        with col_controls:
            st.subheader("Computation")
            st.session_state['fft2d_Nt'] = st.number_input("Nt (Probe Time padding)", value=int(st.session_state['fft2d_Nt']), step=256)
            st.session_state['fft2d_Ntau'] = st.number_input("Ntau (Delay padding)", value=int(st.session_state['fft2d_Ntau']), step=64)
            compute_plot_btn = st.button("Compute FFTs", type="primary", width="stretch")
            
            st.divider()
            st.subheader("Plotting Mode")
            value_to_plot = st.selectbox("Plot Value", options=['Amplitude', 'Real', 'Imaginary', 'Phase'], key="tab5_vtoplot")
            max_p_freq = st.number_input("Max Probe Freq (THz)", value=3.0, step=0.5, key="t5_x_max")
            
            cmap_list =['magma', 'jet', 'viridis', 'cividis', 'inferno', 'blackbody', 'bwr', 'icefire']
            if value_to_plot in['Real', 'Imaginary']: default_cmap = 'bwr'
            elif value_to_plot == 'Phase': default_cmap = 'icefire'
            else: default_cmap = 'magma'
            
            default_idx = cmap_list.index(default_cmap)
            cmap_2d = st.selectbox("Colormap", options=cmap_list, index=default_idx, key="tab5_cmap")
            log_scale = st.checkbox("Log Scale", value=False, key="tab5_log")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 Save Parameters to Config", width="stretch"):
                base_dir = os.path.dirname(st.session_state.base_name)
                config_path = os.path.join(base_dir, "analysis_config.json")
                config_data = {k: st.session_state[k] for k in KEYS_TO_SAVE if k in st.session_state}
                try:
                    with open(config_path, 'w') as f:
                        json.dump(config_data, f, indent=4)
                    st.success(f"Saved successfully to `{os.path.basename(config_path)}`!")
                except Exception as e:
                    st.error(f"Failed to save config: {e}")
            
        with col_plot:
            scan = st.session_state.scan
            
            if compute_plot_btn:
                with st.spinner("Computing 2D FFT & 1D Time-Freq Map..."):
                    # NOTE: subtract_baseline is fully removed from this step in backend/frontend
                    scan.do_2dfft(data=scan.windowed, window=None, 
                                  Nt=st.session_state['fft2d_Nt'], 
                                  Ntau=st.session_state['fft2d_Ntau'])
                    
                    pad_p2 = int(max(1, np.log2(st.session_state['fft2d_Nt'])))
                    fft_1d_df, _ = do_fft_all_taus(scan.windowed, window=None, min_time=-np.inf, max_time=np.inf, pad_power2=pad_p2)
                    scan.ft_1d_df = fft_1d_df
                    st.session_state.scan = scan
            
            if hasattr(scan, 'ft_data') and scan.ft_data is not None:
                # 1. Calculation logic for 2D FFT
                z_comp = scan.ft_data
                if value_to_plot == 'Amplitude': z_2d = np.abs(z_comp)
                elif value_to_plot == 'Real': z_2d = np.real(z_comp)
                elif value_to_plot == 'Imaginary': z_2d = np.imag(z_comp)
                elif value_to_plot == 'Phase': z_2d = np.angle(z_comp)
                
                if log_scale and value_to_plot != 'Phase':
                    z_2d = np.log10(np.where(np.abs(z_2d) > 0, np.abs(z_2d), 1e-10))
                
                max_2d = float(np.nanmax(np.abs(z_2d))) if not np.isnan(np.nanmax(z_2d)) else 1.0
                min_2d_def = -max_2d if (value_to_plot in['Real', 'Imaginary', 'Phase'] and not log_scale) else float(np.nanmin(z_2d))

                # Determine max freq from data
                max_exc_freq_val = float(scan.freq_tau.max())

                c_p1, c_s1 = st.columns([8.5, 1.5])
                with c_s1:
                    vmax_2d = st.number_input("vmax", value=max_2d, step=0.1, key=f"t5_vmax1_{value_to_plot}_{log_scale}")
                    vmin_2d = st.number_input("vmin", value=min_2d_def, step=0.1, key=f"t5_vmin1_{value_to_plot}_{log_scale}")
                with c_p1:
                    fig_2d = plotly_2d_fft(scan.ft_data, scan.freq_t, scan.freq_tau, title='2D Frequency-Frequency Map', 
                                        vmin=vmin_2d, vmax=vmax_2d, cmap=cmap_2d, 
                                        log_scale=log_scale, value_to_plot=value_to_plot, plot_guidelines=True,
                                        x_max=max_p_freq, y_max=max_exc_freq_val)
                    st.plotly_chart(fig_2d, width="stretch")
                
                # 2. Calculation logic for 1D Time-Freq FFT
                if hasattr(scan, 'ft_1d_df') and scan.ft_1d_df is not None:
                    z_comp_1d = scan.ft_1d_df.drop(columns='freq').values.T
                    if value_to_plot == 'Amplitude': z_1d = np.abs(z_comp_1d)
                    elif value_to_plot == 'Real': z_1d = np.real(z_comp_1d)
                    elif value_to_plot == 'Imaginary': z_1d = np.imag(z_comp_1d)
                    elif value_to_plot == 'Phase': z_1d = np.angle(z_comp_1d)
                    
                    if log_scale and value_to_plot != 'Phase':
                        z_1d = np.log10(np.where(np.abs(z_1d) > 0, np.abs(z_1d), 1e-10))
                        
                    max_1d = float(np.nanmax(np.abs(z_1d))) if not np.isnan(np.nanmax(z_1d)) else 1.0
                    min_1d_def = -max_1d if (value_to_plot in ['Real', 'Imaginary', 'Phase'] and not log_scale) else float(np.nanmin(z_1d))

                    c_p2, c_s2 = st.columns([8.5, 1.5])
                    with c_s2:
                        vmax_tf = st.number_input("vmax", value=max_1d, step=0.1, key=f"t5_vmax2_{value_to_plot}_{log_scale}")
                        vmin_tf = st.number_input("vmin", value=min_1d_def, step=0.1, key=f"t5_vmin2_{value_to_plot}_{log_scale}")
                    with c_p2:
                        fig_tf = plotly_time_freq_map(scan.ft_1d_df, title='Time-Frequency Map (Probe FFT)', 
                                                    cmap=cmap_2d, vmin=vmin_tf, vmax=vmax_tf, log_scale=log_scale, value_to_plot=value_to_plot,
                                                    x_max=max_p_freq)
                        st.plotly_chart(fig_tf, width="stretch")

# ==========================================
# TAB 6: Cross Sections
# ==========================================
with tab6:
    st.header("6. 2D FFT Cross Sections")
    if st.session_state.scan is None or getattr(st.session_state.scan, 'ft_data', None) is None:
        st.info("Please compute the 2D FFT in Tab 5 first.")
    else:
        col_ctrl_cs, col_plot_cs = st.columns([1, 3])
        
        with col_ctrl_cs:
            st.subheader("Add Standard Lines")
            b1, b2, b3, b4, b5 = st.columns(5)
            if b1.button("y=0"): add_cs_line(0, 0, "y=0")
            if b2.button("y=x"): add_cs_line(1, 0, "y=x")
            if b3.button("y=-x"): add_cs_line(-1, 0, "y=-x")
            if b4.button("y=2x"): add_cs_line(2, 0, "y=2x")
            if b5.button("x=0"): add_cs_line(0, 0, "x=0", is_vertical=True, x_val=0.0)
            
            st.subheader("Add Custom Line")
            is_vert_custom = st.checkbox("Vertical Line (x = c)", value=False)
            if not is_vert_custom:
                st.markdown(r"*Equation: ν_τ = m · ν_t + b*")
                line_m = st.number_input("Slope (m)", value=1.0, step=0.1)
                line_b = st.number_input("Intercept (b) [THz]", value=0.0, step=0.1)
            else:
                st.markdown(r"*Equation: ν_t = c*")
                line_x = st.number_input("x-intercept (c)[THz]", value=0.0, step=0.1)
                
            if st.button("➕ Add Line", width="stretch"):
                if is_vert_custom:
                    add_cs_line(0, 0, f"x={line_x}", is_vertical=True, x_val=line_x)
                else:
                    add_cs_line(line_m, line_b, f"y={line_m}x+{line_b}")

            st.divider()
            st.subheader("Current Lines")
            if not st.session_state.cs_lines:
                st.write("No lines added yet.")
            else:
                for i, line in enumerate(st.session_state.cs_lines):
                    c_name, c_btn = st.columns([3, 1])
                    c_name.markdown(f"<span style='color:{line['color']}; font-weight:bold;'>■</span> {line['name']}", unsafe_allow_html=True)
                    if c_btn.button("X", key=f"rm_line_{i}"):
                        st.session_state.cs_lines.pop(i)
                        st.rerun()

            st.divider()
            st.subheader("1D Plot Settings")
            cs_x_axis = st.selectbox("X-Axis",["Probe Frequency (ν_t)", "Excitation Frequency (ν_τ)"])
            cs_y_axis = st.selectbox("Y-Axis",["Amplitude", "Real", "Imaginary", "Phase"])

        with col_plot_cs:
            scan = st.session_state.scan
            
            z_comp_cs = scan.ft_data
            if cs_y_axis == 'Amplitude': z_cs = np.abs(z_comp_cs)
            elif cs_y_axis == 'Real': z_cs = np.real(z_comp_cs)
            elif cs_y_axis == 'Imaginary': z_cs = np.imag(z_comp_cs)
            elif cs_y_axis == 'Phase': z_cs = np.angle(z_comp_cs)
            
            max_cs = float(np.nanmax(np.abs(z_cs))) if not np.isnan(np.nanmax(z_cs)) else 1.0
            min_cs_def = -max_cs if cs_y_axis in['Real', 'Imaginary', 'Phase'] else float(np.nanmin(z_cs))

            c_p_map, c_s_map = st.columns([8.5, 1.5])
            with c_s_map:
                cs_cmap = st.selectbox("Colormap", options=['magma', 'jet', 'viridis', 'cividis', 'inferno', 'blackbody', 'bwr', 'icefire'], index=0)
                cs_vmax = st.number_input("vmax", value=max_cs, step=0.1, key=f"t6_vmax_{cs_y_axis}")
                cs_vmin = st.number_input("vmin", value=min_cs_def, step=0.1, key=f"t6_vmin_{cs_y_axis}")
            with c_p_map:
                fig_2d_cs = plotly_2d_fft(scan.ft_data, scan.freq_t, scan.freq_tau, title='2D Map Reference', 
                                        vmin=cs_vmin, vmax=cs_vmax, cmap=cs_cmap, 
                                        log_scale=False, value_to_plot=cs_y_axis, 
                                        plot_guidelines=False, cs_lines=st.session_state.cs_lines,
                                        x_max=3.0, y_max=float(scan.freq_tau.max()))
                st.plotly_chart(fig_2d_cs, width="stretch")
            
            st.divider()
            fig_1d_cs = plotly_cross_sections(scan, st.session_state.cs_lines, cs_x_axis, cs_y_axis)
            st.plotly_chart(fig_1d_cs, width="stretch")