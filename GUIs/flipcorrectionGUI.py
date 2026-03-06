import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import sys
import shutil
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- DYNAMIC PATH RESOLUTION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- IMPORT USER PACKAGE ---
try:
    from mcgillthz.twodim_analysis import THzExp
except ImportError as e:
    st.error(f"Import Error: {e}\nEnsure your folder structure matches: parent/mcgillthz/ and parent/GUIs/")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Phase Flip Correction", layout="wide")
st.title("Batch Phase Flip Auto-Correction")
st.markdown("Load multiple datasets. The software will automatically attempt Type 2 (Analytical) correction, verifying it with a second pass, and falling back to Type 1 (Average) if the noise persists. Review the results dataset by dataset, then save all.")

# --- STATE MANAGEMENT ---
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'dataset_list' not in st.session_state:
    st.session_state.dataset_list =[]
if 'current_base_path' not in st.session_state:
    st.session_state.current_base_path = None

# --- HELPER FUNCTIONS ---
def select_multiple_files():
    """Opens a native OS file dialog allowing multiple file selection."""
    script = """
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.attributes('-topmost', True)
root.withdraw()
file_paths = filedialog.askopenfilenames(
    title="Select 2D Data Files (You can select multiple)",
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
    """Extracts unique base paths from a list of channel CSV files."""
    basenames = set()
    for p in file_paths:
        base = p.replace('_AB.csv', '').replace('_A.csv', '').replace('_B.csv', '').replace('_NL.csv', '')
        basenames.add(base)
    return list(basenames)

def auto_correct_dataset(base_path, threshold, scan_orig=None):
    """
    Helper function to run the correction pipeline on a single dataset.
    """
    if scan_orig is None:
        scan_orig = THzExp(base_path, shift_times=True) 
        
    flip_indices, flip_types = scan_orig.detect_flips(threshold_sigma=threshold)
    
    if len(flip_indices) == 0:
        return {
            'scan_original': scan_orig,
            'scan_corrected': scan_orig,
            'flips':[],
            'status': f'✅ No flips detected at {threshold}σ.',
            'is_saved': False
        }
    else:
        # Attempt Type 2
        scan_t2 = THzExp(base_path, shift_times=True)
        for i, t in zip(flip_indices, flip_types):
            scan_t2.correct_flip(i, t, how=2)
            
        rem_indices, _ = scan_t2.detect_flips(threshold_sigma=threshold)
        
        if len(rem_indices) == 0:
            return {
                'scan_original': scan_orig,
                'scan_corrected': scan_t2,
                'flips':[{'tau_idx': i, 'type': t, 'how': 2} for i, t in zip(flip_indices, flip_types)],
                'status': f'🛠️ Corrected using Type 2 (Analytical Math) at {threshold}σ.',
                'is_saved': False
            }
        else:
            # Fallback to Type 1
            scan_t1 = THzExp(base_path, shift_times=True)
            for i, t in zip(flip_indices, flip_types):
                scan_t1.correct_flip(i, t, how=1)
                
            return {
                'scan_original': scan_orig,
                'scan_corrected': scan_t1,
                'flips':[{'tau_idx': i, 'type': t, 'how': 1} for i, t in zip(flip_indices, flip_types)],
                'status': f'⚠️ Corrected using Type 1 (Average Adjacent) at {threshold}σ - Type 2 verification failed.',
                'is_saved': False
            }

def plotly_2x2_grid(scan, title):
    """Renders a 2x2 interactive grid using Plotly."""
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
            z=data, x=t, y=tau,
            colorscale='RdBu_r', zmin=-vmax_abs, zmax=vmax_abs,
            coloraxis="coloraxis" if i != 1 else "coloraxis2",
            hovertemplate="Time: %{x:.2f} ps<br>Delay: %{y:.2f} ps<br>Field: %{z:.4f}<extra></extra>"
        ), row=positions[i][0], col=positions[i][1])

    fig.update_layout(
        title_text=title, height=550, template="plotly_white",
        coloraxis=dict(colorscale='RdBu_r', colorbar_x=0.45, colorbar_title="A/B/AB"),
        coloraxis2=dict(colorscale='RdBu_r', colorbar_x=1.0, colorbar_title="NL")
    )
    fig.update_xaxes(title_text="THz time t (ps)", row=2)
    fig.update_yaxes(title_text="Excitation delay τ (ps)", col=1)
    return fig


# --- SIDEBAR CONTROLS & AUTO-PROCESSING ---
st.sidebar.header("1. Load & Auto-Correct")
threshold = st.sidebar.number_input("Detection Threshold (Sigma)", value=4.0, step=0.5)

if st.sidebar.button("📁 Browse & Process Flips", type="primary", use_container_width=True):
    chosen_paths = select_multiple_files()
    if chosen_paths:
        bases = extract_basenames(chosen_paths)
        st.session_state.datasets = {} 
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        for idx, base in enumerate(bases):
            base_name_short = os.path.basename(base)
            status_text.text(f"Processing: {base_name_short}...")
            
            try:
                st.session_state.datasets[base] = auto_correct_dataset(base, threshold)
            except Exception as e:
                st.sidebar.error(f"Failed to process {base_name_short}: {e}")
                
            progress_bar.progress((idx + 1) / len(bases))
            
        status_text.text("Processing Complete!")
        
        # Sort datasets so the ones WITH flips appear first in the list
        bases.sort(key=lambda b: len(st.session_state.datasets[b]['flips']) == 0)
        st.session_state.dataset_list = bases
        st.session_state.current_base_path = bases[0] if bases else None
        
        st.rerun()


st.sidebar.divider()
st.sidebar.header("2. Export All")
if st.sidebar.button("💾 Save All Files", use_container_width=True):
    if not st.session_state.datasets:
        st.sidebar.warning("No datasets loaded.")
    else:
        saved_count = 0
        with st.spinner("Backing up raw files and saving corrected A, B, and AB datasets..."):
            for base, data in st.session_state.datasets.items():
                if not data['is_saved']:
                    raw_dir = os.path.join(os.path.dirname(base), 'raw files')
                    os.makedirs(raw_dir, exist_ok=True)
                    
                    scan_c = data['scan_corrected']
                    channels = {'A': scan_c.A, 'B': scan_c.B, 'AB': scan_c.AB}
                    
                    try:
                        for ch_name, df in channels.items():
                            orig_file = f"{base}_{ch_name}.csv"
                            if os.path.exists(orig_file):
                                raw_file_path = os.path.join(raw_dir, os.path.basename(orig_file))
                                if not os.path.exists(raw_file_path):
                                    shutil.copy(orig_file, raw_file_path)
                                
                                with open(orig_file, 'r') as f:
                                    header = f.readline().strip()
                                
                                data_to_write = df.drop(columns='time').values.T
                                np.savetxt(orig_file, data_to_write, delimiter=',', header=header, comments='')
                                
                        st.session_state.datasets[base]['is_saved'] = True
                        saved_count += 1
                    except Exception as e:
                        st.sidebar.error(f"Error saving {os.path.basename(base)}: {e}")
                        
        if saved_count > 0:
            st.sidebar.success(f"Successfully backed up and overwritten A, B, and AB arrays for {saved_count} datasets!")
            st.balloons()
        else:
            st.sidebar.info("All loaded datasets are already saved.")


# --- MAIN WORKFLOW (REVIEW) ---
if not st.session_state.dataset_list:
    st.info("👈 Use the sidebar to load files. The script will auto-correct and allow you to review them here.")
else:
    st.subheader(f"Reviewing {len(st.session_state.dataset_list)} Datasets")
    
    # 1. Format function to dynamically render emoji/text WITHOUT changing the tracked value
    def get_display_name(p):
        if p not in st.session_state.datasets:
            return p
        data = st.session_state.datasets[p]
        b_name = os.path.basename(p)
        if len(data['flips']) > 0:
            return f"🔄 {b_name} ({len(data['flips'])} flips)"
        return f"✅ {b_name}"
    
    # Safe state fallback
    if st.session_state.current_base_path not in st.session_state.dataset_list:
        st.session_state.current_base_path = st.session_state.dataset_list[0] if st.session_state.dataset_list else None
        
    # 2. Bulletproof Navigation Callbacks
    def get_safe_index():
        try:
            return st.session_state.dataset_list.index(st.session_state.current_base_path)
        except ValueError:
            return 0

    def go_prev():
        idx = get_safe_index()
        st.session_state.current_base_path = st.session_state.dataset_list[(idx - 1) % len(st.session_state.dataset_list)]

    def go_next():
        idx = get_safe_index()
        st.session_state.current_base_path = st.session_state.dataset_list[(idx + 1) % len(st.session_state.dataset_list)]
    
    # 3. Render Navigation Bar
    st.markdown("**Dataset Navigation:**")
    col_prev, col_sel, col_next = st.columns([1, 4, 1])
    with col_prev:
        st.button("⬅️ Previous", on_click=go_prev, use_container_width=True)
    with col_sel:
        # Binding the selectbox directly to current_base_path handles state perfectly internally
        st.selectbox(
            "Select Dataset", 
            options=st.session_state.dataset_list, 
            format_func=get_display_name,
            key='current_base_path',
            label_visibility="collapsed"
        )
    with col_next:
        st.button("Next ➡️", on_click=go_next, use_container_width=True)
        
    # 4. Extract Current Dataset
    selected_base_path = st.session_state.current_base_path
    if selected_base_path and selected_base_path in st.session_state.datasets:
        data = st.session_state.datasets[selected_base_path]
        
        # Highlight Status Bar
        if "No flips detected" in data['status']:
            st.success(data['status'])
        elif "Type 2" in data['status']:
            st.info(data['status'])
        else:
            st.warning(data['status'])
            
        # Detail exact flip locations
        if data['flips']:
            taus = data['scan_original'].A.columns[1:].astype(float)
            flip_desc = ", ".join([f"τ={taus[f['tau_idx']]}ps ({f['type']})" for f in data['flips']])
            st.write(f"**Detected Phase Flips:** {flip_desc}")
        
        # --- INDIVIDUAL RE-EVALUATION CONTROLS ---
        with st.expander("⚙️ Re-evaluate Dataset with New Threshold", expanded=False):
            col1, col2 = st.columns([2, 1])
            new_thresh = col1.number_input("New Detection Threshold (Sigma)", value=4.0, step=0.5, key="new_thresh_input")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if col2.button("Re-detect & Correct", use_container_width=True):
                with st.spinner("Re-evaluating dataset..."):
                    try:
                        updated_data = auto_correct_dataset(selected_base_path, new_thresh, scan_orig=data['scan_original'])
                        st.session_state.datasets[selected_base_path] = updated_data
                        
                        # STRICT ENFORCEMENT: Force UI to stay locked on this path after rerun
                        st.session_state.current_base_path = selected_base_path
                    except Exception as e:
                        st.error(f"Error updating dataset: {e}")
                st.rerun()
        
        st.divider()
        
        # 5. Compare Grids Side-by-Side
        col_bef, col_aft = st.columns(2)
        
        with col_bef:
            st.markdown(f"**Original Data**")
            fig_bef = plotly_2x2_grid(data['scan_original'], "Raw Data")
            st.plotly_chart(fig_bef, use_container_width=True)
            
        with col_aft:
            st.markdown(f"**Final Data**")
            fig_aft = plotly_2x2_grid(data['scan_corrected'], "Corrected Data (to be saved)")
            st.plotly_chart(fig_aft, use_container_width=True)