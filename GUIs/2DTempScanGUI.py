import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import sys
import json
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pcolors
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

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
st.set_page_config(page_title="Temperature Scan Comparison", layout="wide")
st.title("2D THz Temperature Scan Comparison")
st.markdown("Batch process and compare temperature-dependent 2D THz scans.")

# --- STATE MANAGEMENT ---
if 'temp_scans' not in st.session_state:
    st.session_state.temp_scans = {}  
if 'sorted_temps' not in st.session_state:
    st.session_state.sorted_temps =[]
if 'config_loaded' not in st.session_state:
    st.session_state.config_loaded = {}
if 'global_fft_max' not in st.session_state:
    st.session_state.global_fft_max = 1.0

# Initialize the trend frequency manual state for robust clicking updates
if 'trend_freq_manual' not in st.session_state:
    st.session_state.trend_freq_manual = 1.0 

# --- HELPER FUNCTIONS ---
def select_multiple_files():
    script = """
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.attributes('-topmost', True)
root.withdraw()
file_paths = filedialog.askopenfilenames(
    title="Select 2D Data Files to Import",
    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
)
print("|".join(file_paths))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    output = result.stdout.strip()
    if output:
        return output.split("|")
    return[]

def extract_basenames(file_paths):
    basenames = set()
    for p in file_paths:
        base = p.replace('_AB.csv', '').replace('_A.csv', '').replace('_B.csv', '').replace('_NL.csv', '')
        basenames.add(base)
    return list(basenames)

def extract_temperature(filename):
    match = re.search(r'(\d+(?:\.\d+)?)\s*K', filename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

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
        return mapping.get(cmap_lower, cmap_name)

def get_continuous_colors(temps, cmap_name):
    if len(temps) <= 1:
        return['#d62728']
    
    t_min, t_max = min(temps), max(temps)
    norm_temps =[(t - t_min) / (t_max - t_min) for t in temps]
    
    cmap_lower = cmap_name.lower()
    if cmap_lower in['magma', 'inferno', 'cividis', 'hot', 'plasma', 'viridis']:
        try:
            cmap = plt.get_cmap(cmap_lower)
            colors =[]
            for norm in norm_temps:
                rgba = cmap(norm)
                colors.append(f'rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})')
            return colors
        except:
            pass
            
    cmap_map = {'thermal': 'Thermal', 'solar': 'Solar', 'sunset_r': 'Sunsetdark', 'agsunset': 'Agsunset'}
    plotly_cmap = cmap_map.get(cmap_lower, 'Magma')
    return pcolors.sample_colorscale(plotly_cmap, norm_temps)

# --- RENDERING FUNCTIONS ---
def plotly_2x2_grid(scan, title):
    fig = make_subplots(rows=2, cols=2, subplot_titles=['AB', 'Nonlinear', 'A', 'B'], 
                        shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.1)
    dfs =[scan.AB, scan.NL, scan.A, scan.B]
    positions =[(1,1), (1,2), (2,1), (2,2)]
    t = dfs[0]['time'].values
    tau = dfs[0].columns[1:].astype(float)

    for i, df in enumerate(dfs):
        data = df.drop(columns='time').values.T
        vmax_abs = np.max(np.abs(data))
        fig.add_trace(go.Heatmap(
            z=data, x=t, y=tau, colorscale='RdBu_r', zmin=-vmax_abs, zmax=vmax_abs,
            coloraxis="coloraxis" if i != 1 else "coloraxis2",
            hovertemplate="Time: %{x:.2f} ps<br>Delay: %{y:.2f} ps<br>Field: %{z:.4f}<extra></extra>"
        ), row=positions[i][0], col=positions[i][1])

    fig.update_layout(title_text=title, height=700, template="plotly_white",
        coloraxis=dict(colorscale='RdBu_r', colorbar_x=0.45, colorbar_title="A/B/AB (a.u.)"),
        coloraxis2=dict(colorscale='RdBu_r', colorbar_x=1.0, colorbar_title="NL (a.u.)"), margin=dict(t=40, b=0))
    fig.update_xaxes(title_text="THz time t (ps)", row=2)
    fig.update_yaxes(title_text="Excitation delay τ (ps)", col=1)
    return fig

def get_spectrum_traces(data_arr, fft_arr, color, label):
    tr_time = go.Scatter(x=data_arr[0], y=data_arr[1], name=label, line=dict(color=color))
    tr_spec = go.Scatter(x=fft_arr[0], y=fft_arr[1], name=label, line=dict(color=color), showlegend=False)
    return tr_time, tr_spec

def get_2d_plot_data(scan, value_to_plot, log_scale, vmin):
    z_comp = scan.ft_data
    if value_to_plot == 'Amplitude': z_2d = np.abs(z_comp)
    elif value_to_plot == 'Real': z_2d = np.real(z_comp)
    elif value_to_plot == 'Imaginary': z_2d = np.imag(z_comp)
    elif value_to_plot == 'Phase': z_2d = np.angle(z_comp)
    
    if log_scale and value_to_plot != 'Phase':
        z_2d = np.log10(np.where(np.abs(z_2d) > 0, np.abs(z_2d), 1e-10))
    return z_2d

def get_tf_plot_data(scan, value_to_plot, log_scale):
    z_comp = scan.ft_1d_df.drop(columns='freq').values.T
    if value_to_plot == 'Amplitude': z_tf = np.abs(z_comp)
    elif value_to_plot == 'Real': z_tf = np.real(z_comp)
    elif value_to_plot == 'Imaginary': z_tf = np.imag(z_comp)
    elif value_to_plot == 'Phase': z_tf = np.angle(z_comp)

    if log_scale and value_to_plot != 'Phase':
        z_tf = np.log10(np.where(np.abs(z_tf) > 0, np.abs(z_tf), np.nan))
    return z_tf



def get_cross_section(scan, line_def, x_axis_type, y_axis_type):
    interp_r = RegularGridInterpolator((scan.freq_tau, scan.freq_t), np.real(scan.ft_data), bounds_error=False, fill_value=np.nan)
    interp_i = RegularGridInterpolator((scan.freq_tau, scan.freq_t), np.imag(scan.ft_data), bounds_error=False, fill_value=np.nan)
    
    t_min, t_max = 0.0, min(10.0, scan.freq_t.max())
    tau_min, tau_max = scan.freq_tau.min(), scan.freq_tau.max()
    
    is_vert = line_def.get('is_vertical', False)
    if is_vert and x_axis_type == "Probe Frequency (ν_t)": return None, None, None
        
    if is_vert:
        vtau_arr = np.linspace(tau_min, tau_max, 500)
        vt_arr = np.full_like(vtau_arr, line_def['x_val'])
    else:
        vt_full = np.linspace(t_min, t_max, 500)
        vtau_full = line_def['m'] * vt_full + line_def['b']
        mask = (vtau_full >= tau_min) & (vtau_full <= tau_max)
        if not np.any(mask): return None, None, None
        vt_arr = vt_full[mask]
        vtau_arr = vtau_full[mask]
    
    pts = np.column_stack((vtau_arr, vt_arr))
    z_comp = interp_r(pts) + 1j * interp_i(pts)
    
    if y_axis_type == 'Amplitude': y_vals = np.abs(z_comp)
    elif y_axis_type == 'Real': y_vals = np.real(z_comp)
    elif y_axis_type == 'Imaginary': y_vals = np.imag(z_comp)
    elif y_axis_type == 'Phase': y_vals = np.angle(z_comp)
    
    if x_axis_type == "Probe Frequency (ν_t)":
        return vt_arr, y_vals, "Probe Frequency ν_t (THz)"
    return vtau_arr, y_vals, "Excitation Frequency ν_τ (THz)"


# --- LAYOUT: TABS ---
tab1, tab2, tab3 = st.tabs(["1. Import & Raw Data", "2. Individual Temperature Viewer", "3. Temperature Overlays"])

# ==========================================
# TAB 1: Import & Raw Data
# ==========================================
with tab1:
    st.header("1. Batch Import Files")
    col_btn, col_path = st.columns([1, 4])
    with col_btn:
        if st.button("📁 Select Files", width="stretch"):
            chosen_paths = select_multiple_files()
            if chosen_paths:
                bases = extract_basenames(chosen_paths)
                if not bases:
                    st.error("No valid dataset files found.")
                else:
                    folder_path = os.path.dirname(bases[0])
                    st.session_state.folder_path = folder_path
                    calib_path = os.path.join(folder_path, "calibration.csv")
                    config_path = os.path.join(folder_path, "analysis_config.json")
                    has_calib = os.path.exists(calib_path)
                    
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            cfg = json.load(f)
                        st.session_state.config_loaded = cfg
                    else:
                        st.warning("No `analysis_config.json` found. Using fallback default parameters.")
                        cfg = {
                            'fft_window_type': 'hann', 'fft_min_time': -6.0, 'fft_max_time': 6.0, 'fft_pad_p2': 1,
                            'leak_tau_min': -2.0, 'leak_tau_max': 0.0, 'leak_t_min': -2.0, 'leak_t_max': 1.0,
                            'win_w_type': 'tukey', 'win_tukey_alpha': 0.5, 'win_min_time': -9.0, 'win_max_time': 10.0,
                            'win_flat_tau_start': -0.5, 'win_flat_tau_end': 1.5, 'win_min_tau': -1.5, 'win_max_tau': 2.5,
                            'win_sub_baseline': False, 'win_t_base_min': -2.0, 'win_t_base_max': -1.0,
                            'fft2d_Nt': 1024, 'fft2d_Ntau': 128
                        }
                        st.session_state.config_loaded = cfg

                    temp_dict = {}
                    global_max_fft = 0.0
                    progress = st.progress(0)
                    status = st.empty()
                    
                    for idx, base in enumerate(bases):
                        b_name = os.path.basename(base)
                        T = extract_temperature(b_name)
                        if T is not None:
                            status.text(f"Processing: {b_name} ({T} K)...")
                            try:
                                scan = THzExp(base, calibration_file=calib_path if has_calib else None, shift_times=True)
                                
                                # 1D FFTs
                                scan.fft_A, _ = do_fft_all_taus(scan.A, window=cfg['fft_window_type'], min_time=cfg['fft_min_time'], max_time=cfg['fft_max_time'], pad_power2=cfg['fft_pad_p2'])
                                scan.fft_B, _ = do_fft_all_taus(scan.B, window=cfg['fft_window_type'], min_time=cfg['fft_min_time'], max_time=cfg['fft_max_time'], pad_power2=cfg['fft_pad_p2'])
                                
                                # Track Global FFT Maximum for fixed amplitude mapping
                                a_max = np.nanmax(np.abs(scan.fft_A.drop(columns='freq').values))
                                b_max = np.nanmax(np.abs(scan.fft_B.drop(columns='freq').values))
                                global_max_fft = max(global_max_fft, a_max, b_max)
                                
                                # Process Pipeline
                                scan.no_bg = subtract_leak_NL(scan.NL, cfg['leak_tau_min'], cfg['leak_tau_max'], cfg['leak_t_min'], cfg['leak_t_max'])
                                t_win = (cfg['win_w_type'], cfg['win_tukey_alpha']) if cfg['win_w_type'] == 'tukey' else cfg['win_w_type']
                                scan.windowed = window_2d(scan.no_bg, t_window=t_win, 
                                                          flat_tau_start=cfg['win_flat_tau_start'], flat_tau_end=cfg['win_flat_tau_end'], 
                                                          min_tau=cfg['win_min_tau'], max_tau=cfg['win_max_tau'],
                                                          min_time=cfg['win_min_time'], max_time=cfg['win_max_time'],
                                                          subtract_baseline=cfg['win_sub_baseline'], 
                                                          t_base_min=cfg['win_t_base_min'], t_base_max=cfg['win_t_base_max'])
                                
                                scan.do_2dfft(data=scan.windowed, window=None, Nt=cfg['fft2d_Nt'], Ntau=cfg['fft2d_Ntau'])
                                pad_p2 = int(max(1, np.log2(cfg['fft2d_Nt'])))
                                scan.ft_1d_df, _ = do_fft_all_taus(scan.windowed, window=None, min_time=-np.inf, max_time=np.inf, pad_power2=pad_p2)
                                
                                temp_dict[T] = scan
                            except Exception as e:
                                st.error(f"Error processing {b_name}: {e}")
                        
                        progress.progress((idx + 1) / len(bases))
                        
                    status.text("All datasets imported and processed successfully!")
                    st.session_state.global_fft_max = float(global_max_fft)
                    st.session_state.temp_scans = temp_dict
                    st.session_state.sorted_temps = sorted(list(temp_dict.keys()))
                    st.rerun()

    with col_path:
        st.text_input("Active Directory", value=st.session_state.get('folder_path', ''), disabled=True)

    if st.session_state.sorted_temps:
        st.divider()
        c_sel, c_info = st.columns([1, 2])
        selected_raw_temp = c_sel.select_slider("Inspect Raw Dataset (Temperature K):", options=st.session_state.sorted_temps, key="tab1_temp")
        c_info.success(f"Successfully loaded {len(st.session_state.sorted_temps)} temperatures: {st.session_state.sorted_temps}")
        
        scan_to_plot = st.session_state.temp_scans[selected_raw_temp]
        fig = plotly_2x2_grid(scan_to_plot, title=f"Raw Data: {selected_raw_temp} K")
        st.plotly_chart(fig, width="stretch")


# ==========================================
# TAB 2: Individual Temperature Viewer
# ==========================================
with tab2:
    if not st.session_state.sorted_temps:
        st.info("Import data in Tab 1 to proceed.")
    else:
        st.header("Individual Scan Viewer")
        
        c_temp, c_plotval, c_cmap, c_log = st.columns([2, 1.5, 1.5, 1])
        t_idx = st.session_state.sorted_temps.index(st.session_state.get('tab2_temp', st.session_state.sorted_temps[0]))
        selected_temp = c_temp.select_slider("Select Temperature (K):", options=st.session_state.sorted_temps, value=st.session_state.sorted_temps[t_idx], key='tab2_temp')
        val_to_plot = c_plotval.selectbox("Plot Value", options=['Amplitude', 'Real', 'Imaginary', 'Phase'], key="tab2_vtoplot")
        
        cmap_opts =['magma', 'jet', 'viridis', 'cividis', 'inferno', 'blackbody', 'bwr', 'icefire']
        default_cmap = 'bwr' if val_to_plot in ['Real', 'Imaginary'] else 'icefire' if val_to_plot == 'Phase' else 'magma'
        cmap_2d = c_cmap.selectbox("Colormap", options=cmap_opts, index=cmap_opts.index(default_cmap), key="tab2_cmap")
        log_scale = c_log.checkbox("Log Scale", value=False, key="tab2_log")
        
        st.divider()
        scan = st.session_state.temp_scans[selected_temp]
        
        c_xfreq, c_yfreq, c_xtime, c_ytime = st.columns(4)
        max_p_freq = c_xfreq.number_input("Max Probe Freq (THz)", value=3.0, step=0.5, key="tab2_px")
        max_e_freq = c_yfreq.number_input("Max Excitation Freq (THz)", value=float(scan.freq_tau.max()), step=0.5, key="tab2_ex")
        min_tt_time = c_xtime.number_input("Min Probe Freq (ps)", value=float(st.session_state.config_loaded['win_min_time']), step=0.5, key="tab2_mint")
        max_tt_time = c_ytime.number_input("Max Probe time (ps)", value=float(st.session_state.config_loaded['win_max_time']), step=0.5, key="tab2_maxt")

        z_2d = get_2d_plot_data(scan, val_to_plot, log_scale, 0)
        max_2d = float(np.nanmax(np.abs(z_2d))) if not np.isnan(np.nanmax(z_2d)) else 1.0
        min_2d_def = -max_2d if (val_to_plot in['Real', 'Imaginary', 'Phase'] and not log_scale) else float(np.nanmin(z_2d))
        
        z_tf = get_tf_plot_data(scan, val_to_plot, log_scale)
        max_tf = float(np.nanmax(np.abs(z_tf))) if not np.isnan(np.nanmax(z_tf)) else 1.0
        min_tf_def = -max_tf if (val_to_plot in['Real', 'Imaginary', 'Phase'] and not log_scale) else float(np.nanmin(z_tf))

        z_tt = scan.windowed.drop(columns='time').values.T
        max_tt = float(np.nanmax(np.abs(z_tt))) if not np.isnan(np.nanmax(z_tt)) else 1.0

        # --- Plot 2D FFT ---
        c_p1, c_s1 = st.columns([8.5, 1.5])
        with c_s1:
            st.markdown("<br><br>", unsafe_allow_html=True)
            v2_max = st.number_input("vmax", value=max_2d, step=0.1, key="t2_v2max")
            v2_min = st.number_input("vmin", value=min_2d_def, step=0.1, key="t2_v2min")
        with c_p1:
            fig2d = go.Figure(data=go.Heatmap(z=z_2d, x=scan.freq_t, y=scan.freq_tau, colorscale=get_plotly_cmap(cmap_2d), zmin=v2_min, zmax=v2_max))
            
            # ADD DASHED WHITE LINE AT Y=0
            fig2d.add_hline(y=0, line_dash="dash", line_color="white", line_width=1.5)
            
            fig2d.update_layout(title=f"2D Frequency-Frequency Map ({selected_temp} K)", xaxis_title="Probe frequency ν_t (THz)", yaxis_title="Excitation frequency ν_τ (THz)", height=500, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
            fig2d.update_xaxes(range=[0, max_p_freq]); fig2d.update_yaxes(range=[-max_e_freq, max_e_freq])
            st.plotly_chart(fig2d, width="stretch")
            
        # --- Plot Time-Time Map and Time-Freq Map ---
        c_p2, c_p3, c_s2 = st.columns([4.5, 4.5, 1])
        with c_s2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            vt_max = st.number_input("vmax", value=max_tf, step=0.1, key="t3_vtmax")
            vt_min = st.number_input("vmin", value=min_tf_def, step=0.1, key="t3_vtmin")
            vtt_max = st.number_input("vmax (time-time plot)", value=max_tt, step=0.1, key="t2_vtmax")
        with c_p2:
            figtt = go.Figure(data=go.Heatmap(z=z_tt, x=scan.windowed['time'].values, y=scan.windowed.columns[1:].astype(float), colorscale=get_plotly_cmap('bwr'), zmin=-vtt_max, zmax=vtt_max))
            figtt.update_layout(title=f"Time-Time Map ({selected_temp} K)", xaxis_title="Probe time t (ps)", yaxis_title="Excitation delay τ (ps)", height=500, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
            figtt.update_xaxes(range=[min_tt_time, max_tt_time])
            st.plotly_chart(figtt, width="stretch")
        with c_p3:
            figtf = go.Figure(data=go.Heatmap(z=z_tf, x=scan.ft_1d_df['freq'].values, y=scan.ft_1d_df.columns[1:].astype(float), colorscale=get_plotly_cmap(cmap_2d), zmin=vt_min, zmax=vt_max))
            figtf.update_layout(title=f"Time-Frequency Map (Probe FFT) ({selected_temp} K)", xaxis_title="Probe frequency ν_t (THz)", yaxis_title="Excitation delay τ (ps)", height=500, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
            figtf.update_xaxes(range=[0, max_p_freq])
            st.plotly_chart(figtf, width="stretch")
            
        st.divider()
        
        # --- 1D Traces ---
        st.subheader("1D Time-Domain & Spectra")
        c_tau, c_fmax = st.columns([3, 1])
        taus = scan.A.columns[1:].astype(float)
        sel_tau = c_tau.select_slider("Excitation Delay τ (ps)", options=taus, value=taus[0], key='tab2_tau')
        fixed_fft_max = c_fmax.number_input("Max 1D FFT Amplitude", value=st.session_state.global_fft_max, step=0.1, help="Fixed max limit for all temperatures/delays")

        col_match = [c for c in scan.A.columns[1:] if abs(float(c) - sel_tau) < 1e-5][0]
        
        tr_tb, tr_fb = get_spectrum_traces(np.array([scan.B['time'], scan.B[col_match]]), np.array([scan.fft_B['freq'], scan.fft_B[col_match]]), 'blue', 'B')
        tr_ta, tr_fa = get_spectrum_traces(np.array([scan.A['time'], scan.A[col_match]]), np.array([scan.fft_A['freq'], scan.fft_A[col_match]]), 'red', 'A')
        
        fig_1d = make_subplots(rows=1, cols=2, subplot_titles=[f"Time Domain (τ = {sel_tau} ps, T = {selected_temp} K)", f"Spectrum (τ = {sel_tau} ps, T = {selected_temp} K)"])
        fig_1d.add_trace(tr_tb, row=1, col=1); fig_1d.add_trace(tr_fb, row=1, col=2)
        fig_1d.add_trace(tr_ta, row=1, col=1); fig_1d.add_trace(tr_fa, row=1, col=2)
        
        fig_1d.update_xaxes(title_text="Time (ps)", row=1, col=1)
        fig_1d.update_yaxes(title_text="Electric Field (a.u.)", row=1, col=1)
        fig_1d.update_xaxes(title_text="Frequency (THz)", range=[0, max_p_freq], row=1, col=2)
        
        if log_scale:
            fig_1d.update_yaxes(title_text="FFT Amplitude", type="log", row=1, col=2)
            fig_1d.update_yaxes(range=[np.log10(fixed_fft_max*1e-4), np.log10(fixed_fft_max)], row=1, col=2)
        else:
            fig_1d.update_yaxes(title_text="FFT Amplitude", type="linear", row=1, col=2)
            fig_1d.update_yaxes(range=[0, fixed_fft_max], row=1, col=2)
            
        fig_1d.update_layout(height=450, hovermode="x unified", template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_1d, width="stretch")

# ==========================================
# TAB 3: Temperature Overlays
# ==========================================
with tab3:
    if not st.session_state.sorted_temps:
        st.info("Import data in Tab 1 to proceed.")
    else:
        st.header("Temperature Dependency Overlays")
        
        st.subheader("Global Plotting Parameters")
        c_p1, c_p2 = st.columns(2)
        overlay_cmap = c_p1.selectbox("Overlay Colormap (Maps T_min to T_max)", 
                                      options=['magma', 'thermal', 'cividis', 'hot', 'solar', 'sunset', 'agsunset'], index=0)
        temp_colors = get_continuous_colors(st.session_state.sorted_temps, overlay_cmap)
        tab3_max_freq = c_p2.number_input("Maximum Frequency Axis (THz)", value=4.0, step=0.5, key="tab3_max_freq")
        
        st.divider()
        st.subheader("1D Spectra Overlay")
        c_o1, c_o2, c_o3 = st.columns(3)
        ov_ch = c_o1.selectbox("Channel to Plot", options=['A', 'B'], key="ov_ch")
        
        ref_scan = st.session_state.temp_scans[st.session_state.sorted_temps[0]]
        ref_taus = ref_scan.A.columns[1:].astype(float)
        ov_tau = c_o2.select_slider("Excitation Delay τ (ps)", options=ref_taus, value=ref_taus[0], key="ov_tau")
        ov_log = c_o3.checkbox("Log Scale Y-Axis", value=True, key="ov_log")
        
        fig_ov_1d = go.Figure()
        for i, T in enumerate(st.session_state.sorted_temps):
            sc = st.session_state.temp_scans[T]
            col_m =[c for c in sc.A.columns[1:] if abs(float(c) - ov_tau) < 1e-5][0]
            fft_df = sc.fft_A if ov_ch == 'A' else sc.fft_B
            fig_ov_1d.add_trace(go.Scatter(
                x=fft_df['freq'], y=fft_df[col_m], name=f"{T} K", 
                mode='lines',
                line=dict(color=temp_colors[i], width=2),
                # Custom tooltip placing the X value on a new line!
                hovertemplate=f"<b>{T} K</b> | Val: %{{y:.4f}}<br>Freq: %{{x:.3f}} THz<extra></extra>" 
            ))
            
        fig_ov_1d.update_xaxes(
            title_text="Frequency (THz)", range=[0, tab3_max_freq],
            showspikes=True, spikemode="toaxis", spikecolor="black", spikethickness=1
        )
        fig_ov_1d.update_yaxes(title_text="FFT Amplitude (a.u.)", type="log" if ov_log else "linear")
        fig_ov_1d.update_layout(height=450, hovermode="closest", template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_ov_1d, width="stretch")
        
        st.divider()
        st.subheader("2D Cross Sections Overlay & Interactive Trend")
        col_cs_ctrl, col_cs_plot = st.columns([1.5, 3])
        
        with col_cs_ctrl:
            cs_x_axis = st.selectbox("X-Axis",["Probe Frequency (ν_t)", "Excitation Frequency (ν_τ)"], key="ov_cs_x")
            cs_y_axis = st.selectbox("Y-Axis",["Amplitude", "Real", "Imaginary", "Phase"], key="ov_cs_y")
            
            st.markdown("**Define Slice Line:**")
            line_type = st.radio("Line Preset",["y=0", "y=x", "y=-x", "y=2x", "x=0 (Vertical)", "Custom"])
            
            m, b, is_vert, x_val = 0.0, 0.0, False, 0.0
            if line_type == "y=0": m, b = 0.0, 0.0
            elif line_type == "y=x": m, b = 1.0, 0.0
            elif line_type == "y=-x": m, b = -1.0, 0.0
            elif line_type == "y=2x": m, b = 2.0, 0.0
            elif line_type == "x=0 (Vertical)": is_vert, x_val = True, 0.0
            else:
                is_v = st.checkbox("Vertical (x=c)")
                if is_v:
                    is_vert = True
                    x_val = st.number_input("x-intercept (c)", value=0.0, step=0.1)
                else:
                    m = st.number_input("Slope (m)", value=1.0, step=0.1)
                    b = st.number_input("Intercept (b)", value=0.0, step=0.1)
                    
            line_def = {'m': m, 'b': b, 'is_vertical': is_vert, 'x_val': x_val}

            st.markdown("<br>**Extract Trend**", unsafe_allow_html=True)
            st.caption("Click directly on any point in the plot to extract its trend!")
            
            st.session_state.trend_freq_manual = st.number_input(
                "Target Frequency (THz)", 
                value=st.session_state.trend_freq_manual, 
                step=0.1, 
                key="trend_freq_manual_ui"
            )

        with col_cs_plot:
            fig_ov_cs = go.Figure()
            
            trend_x = []
            trend_y =[]
            colors_for_trend =[]
            
            for i, T in enumerate(st.session_state.sorted_temps):
                sc = st.session_state.temp_scans[T]
                x_arr, y_arr, x_lbl = get_cross_section(sc, line_def, cs_x_axis, cs_y_axis)
                
                if x_arr is not None:
                    fig_ov_cs.add_trace(go.Scatter(
                        x=x_arr, y=y_arr, name=f"{T} K", 
                        mode='lines+markers', # Required so clicks register perfectly
                        marker=dict(size=3, color=temp_colors[i], opacity=0.01), # Highly transparent markers for selection
                        line=dict(color=temp_colors[i], width=2),
                        hovertemplate=f"<b>{T} K</b> | Val: %{{y:.4f}}<br>Freq: %{{x:.3f}} THz<extra></extra>"
                    ))
                    
                    idx = np.argmin(np.abs(x_arr - st.session_state.trend_freq_manual))
                    trend_x.append(T)
                    trend_y.append(y_arr[idx])
                    colors_for_trend.append(temp_colors[i])
            
            if not fig_ov_cs.data:
                fig_ov_cs.update_layout(title="Selected line falls outside data boundaries or is invalid for selected axis.", template="plotly_white")
                st.plotly_chart(fig_ov_cs, width="stretch")
            else:
                if cs_x_axis == "Probe Frequency (ν_t)":
                    fig_ov_cs.update_xaxes(range=[0, tab3_max_freq])
                else:
                    fig_ov_cs.update_xaxes(range=[-tab3_max_freq, tab3_max_freq])

                fig_ov_cs.update_xaxes(showspikes=True, spikemode="toaxis", spikecolor="black", spikethickness=1)
                
                # hovermode='closest' correctly isolates single traces
                fig_ov_cs.update_layout(
                    xaxis_title=x_lbl, yaxis_title=f"{cs_y_axis} (a.u.)", 
                    height=450, template="plotly_white", 
                    hovermode="closest", clickmode="event+select", margin=dict(l=0, r=0, t=10, b=0)
                )
                
                # Render with clicking support enabled
                selection_event = st.plotly_chart(
                    fig_ov_cs, 
                    width="stretch", 
                    on_select="rerun", 
                    selection_mode=["points"]
                )
                
                # Check for clicks and instantly synchronize the number_input target
                if selection_event and selection_event.get("selection", {}).get("points"):
                    clicked_x = float(selection_event["selection"]["points"][0]["x"])
                    if abs(clicked_x - st.session_state.trend_freq_manual) > 1e-6:
                        # Direct state update from click forces the number_input above to perfectly sync!
                        st.session_state.trend_freq_manual = clicked_x
                        st.rerun()

            # --- TREND VS TEMPERATURE PLOT ---
            if trend_x:
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=trend_x, y=trend_y, mode='lines+markers', 
                    line=dict(color='gray', dash='dash'),
                    marker=dict(size=12, color=colors_for_trend, line=dict(width=1, color='black')),
                    name=f"Value at {st.session_state.trend_freq_manual:.2f} THz"
                ))
                fig_trend.update_layout(
                    title=f"Temperature Dependency (at {st.session_state.trend_freq_manual:.2f} THz)", 
                    xaxis_title="Temperature (K)", 
                    yaxis_title=f"{cs_y_axis} (a.u.)", 
                    height=350, template="plotly_white", margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_trend, width="stretch")