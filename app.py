"""
NIR Spectra Analyzer - Main Web Application

A Streamlit-based web application for near-infrared spectral data analysis.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from file_readers import SpectralFileReader, SpectralData
from spectral_processing import SpectralProcessor


import pandas as pd
def main():
    st.set_page_config(
        page_title="NIR Spectra Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 NIR Spectra Analyzer")
    st.markdown("*A comprehensive tool for near-infrared spectral data analysis*")
    
    # Custom CSS to minimize sidebar width
    st.markdown("""
        <style>
        .css-1d391kg {
            width: 20rem;
        }
        .css-1lcbmhc {
            max-width: 20rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Sidebar for file upload and options
    with st.sidebar:
        st.header("📁 Data Input")
        
        uploaded_files = st.file_uploader(
            "Choose spectral data files",
            type=['asd', 'spc', 'csv', 'txt'],
            help="Supported formats: ASD, SPC, CSV, TXT",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"Files uploaded: {[f.name for f in uploaded_files]}")
            
            # Process multiple files
            spectral_datasets = []
            file_info = []
            
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Read the spectral data
                    spectral_data = SpectralFileReader.read_file(temp_path)
                    spectral_datasets.append({
                        'name': uploaded_file.name,
                        'data': spectral_data,
                        'filename': uploaded_file.name
                    })
                    
                    file_info.append({
                        'name': uploaded_file.name,
                        'format': spectral_data.metadata.get('format', 'Unknown'),
                        'points': len(spectral_data.wavelengths),
                        'wl_range': f"{spectral_data.wavelengths.min():.1f} - {spectral_data.wavelengths.max():.1f} nm"
                    })
                    
                    # Clean up temp file
                    Path(temp_path).unlink()
                    
                except Exception as e:
                    st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
                    if Path(temp_path).exists():
                        Path(temp_path).unlink()
            
            if spectral_datasets:
                st.session_state.spectral_datasets = spectral_datasets
                st.success(f"✅ {len(spectral_datasets)} file(s) loaded successfully!")
                
                # Display file info table
                if file_info:
                    st.subheader("📋 File Information")
                    info_df = pd.DataFrame(file_info)
                    st.dataframe(info_df, width='stretch')
        
        # Clear button for loaded data
        if 'spectral_datasets' in st.session_state:
            st.markdown("---")
            if st.button("🗑️ Clear All Spectra", help="Remove all loaded spectral data and start over"):
                # Clear all session state data
                keys_to_clear = ['spectral_datasets', 'processed_data']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("All spectra cleared!")
                st.rerun()
    
    # Main content area
    if 'spectral_datasets' in st.session_state:
        spectral_datasets = st.session_state.spectral_datasets
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Spectrum View", "🔧 Preprocessing", "📊 Analysis"])
        
        with tab1:
            st.header("Spectral Data Visualization")
            
            col1, col2 = st.columns([3, 1])
            
            # Calculate overall wavelength range from all datasets
            all_wl_min = min(dataset['data'].wavelengths.min() for dataset in spectral_datasets)
            all_wl_max = max(dataset['data'].wavelengths.max() for dataset in spectral_datasets)
            
            # Initialize session state for wavelength range if not exists
            if 'wl_min' not in st.session_state:
                st.session_state.wl_min = float(all_wl_min)
            if 'wl_max' not in st.session_state:
                st.session_state.wl_max = float(all_wl_max)
            
            # Controls above the plot
            st.subheader("Display Controls")
            
            # Row 1: File selection and color palette
            control_col1, control_col2 = st.columns(2)
            with control_col1:
                selected_files = st.multiselect(
                    "Select files to display",
                    options=[dataset['name'] for dataset in spectral_datasets],
                    default=[dataset['name'] for dataset in spectral_datasets],
                    help="Select which spectra to display on the plot"
                )
            with control_col2:
                color_palette = st.selectbox(
                    "Color Palette",
                    options=["Plotly", "Viridis", "Rainbow", "Custom"],
                    help="Choose color scheme for multiple spectra"
                )
            
            # Row 2: Wavelength controls
            wl_col1, wl_col2, wl_col3 = st.columns([1, 1, 2])
            with wl_col1:
                manual_wl_min = st.number_input(
                    "Min Wavelength (nm)",
                    min_value=float(all_wl_min),
                    max_value=float(all_wl_max),
                    value=st.session_state.wl_min,
                    step=1.0
                )
            with wl_col2:
                manual_wl_max = st.number_input(
                    "Max Wavelength (nm)",
                    min_value=float(all_wl_min),
                    max_value=float(all_wl_max),
                    value=st.session_state.wl_max,
                    step=1.0
                )
            with wl_col3:
                wl_range = st.slider(
                    "Wavelength Range (nm)",
                    min_value=float(all_wl_min),
                    max_value=float(all_wl_max),
                    value=(manual_wl_min, manual_wl_max),
                    step=1.0,
                    label_visibility="collapsed"
                )
            
            # Row 3: Quick range buttons
            preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
            with preset_col1:
                if st.button("UV (350-400)", key="uv_btn"):
                    st.session_state.wl_min = 350.0
                    st.session_state.wl_max = 400.0
                    st.rerun()
            with preset_col2:
                if st.button("Visible (380-780)", key="vis_btn"):
                    st.session_state.wl_min = 380.0
                    st.session_state.wl_max = 780.0
                    st.rerun()
            with preset_col3:
                if st.button("NIR (780-2500)", key="nir_btn"):
                    st.session_state.wl_min = 780.0
                    st.session_state.wl_max = 2500.0
                    st.rerun()
            with preset_col4:
                if st.button("Full Range", key="full_btn"):
                    st.session_state.wl_min = float(all_wl_min)
                    st.session_state.wl_max = float(all_wl_max)
                    st.rerun()
            
            # Update session state from manual inputs or slider (whichever changed)
            st.session_state.wl_min = manual_wl_min
            st.session_state.wl_max = manual_wl_max
            
            
            # Create interactive plot with multiple spectra
            fig = go.Figure()
            
            # Define color palettes
            if color_palette == "Plotly":
                colors = px.colors.qualitative.Plotly
            elif color_palette == "Viridis":
                colors = px.colors.sequential.Viridis
            elif color_palette == "Rainbow":
                colors = px.colors.qualitative.Set3
            else:  # Custom
                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            # Plot each selected spectrum
            for i, dataset in enumerate(spectral_datasets):
                if dataset['name'] in selected_files:
                    spectral_data = dataset['data']
                    
                    # Filter data based on wavelength range
                    mask = (spectral_data.wavelengths >= wl_range[0]) & (spectral_data.wavelengths <= wl_range[1])
                    filtered_wl = spectral_data.wavelengths[mask]
                    filtered_intensities = spectral_data.intensities[mask]
                    
                    fig.add_trace(go.Scatter(
                        x=filtered_wl,
                        y=filtered_intensities,
                        mode='lines',
                        name=dataset['name'],
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                      'Wavelength: %{x:.1f} nm<br>' +
                                      'Intensity: %{y:.4f}<br>' +
                                      '<extra></extra>'
                    ))
            
            fig.update_layout(
                title=f"Near-Infrared Spectra ({len(selected_files)} files)",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Intensity/Reflectance",
                height=500,
                showlegend=True,
                hovermode='closest',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01
                )
            )
                
            st.plotly_chart(fig, use_container_width=True)  # This one is fine for plotly
        
        with tab2:
            st.header("Spectral Preprocessing")
            st.markdown("*Apply preprocessing to multiple spectra for baseline correction, smoothing, derivatives, and normalization*")
            
            if len(spectral_datasets) == 0:
                st.warning("No spectral data loaded. Please upload files in the Spectrum View tab first.")
                return
            
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                st.subheader("📊 Sample Selection")
                
                # Multiple file selection
                selected_files_processing = st.multiselect(
                    "Select files to process",
                    options=[dataset['name'] for dataset in spectral_datasets],
                    default=[dataset['name'] for dataset in spectral_datasets][:3],  # Default first 3
                    help="Select multiple files for batch processing"
                )
                
                if not selected_files_processing:
                    st.warning("Please select at least one file to process.")
                    return
                
                # Get wavelength ranges for all selected files
                selected_datasets = [d for d in spectral_datasets if d['name'] in selected_files_processing]
                all_wl_min = min(ds['data'].wavelengths.min() for ds in selected_datasets)
                all_wl_max = max(ds['data'].wavelengths.max() for ds in selected_datasets)
                
                # --- WAVELENGTH RANGE SELECTION ---
                st.subheader("📏 Wavelength Range")
                
                # Initialize session state for preprocessing wavelength range if not exists
                if 'preproc_wl_min' not in st.session_state:
                    st.session_state.preproc_wl_min = float(all_wl_min)
                if 'preproc_wl_max' not in st.session_state:
                    st.session_state.preproc_wl_max = float(all_wl_max)
                
                # Manual wavelength inputs
                manual_col1, manual_col2 = st.columns(2)
                with manual_col1:
                    manual_wl_min = st.number_input(
                        "Min (nm)",
                        min_value=float(all_wl_min),
                        max_value=float(all_wl_max),
                        value=st.session_state.preproc_wl_min,
                        step=1.0,
                        key="preproc_manual_min"
                    )
                with manual_col2:
                    manual_wl_max = st.number_input(
                        "Max (nm)", 
                        min_value=float(all_wl_min),
                        max_value=float(all_wl_max),
                        value=st.session_state.preproc_wl_max,
                        step=1.0,
                        key="preproc_manual_max"
                    )
                
                # Quick preset buttons
                preset_col1, preset_col2 = st.columns(2)
                with preset_col1:
                    if st.button("NIR (780-2500)", key="preproc_nir_btn"):
                        st.session_state.preproc_wl_min = 780.0
                        st.session_state.preproc_wl_max = 2500.0
                        st.rerun()
                with preset_col2:
                    if st.button("Full Range", key="preproc_full_btn"):
                        st.session_state.preproc_wl_min = float(all_wl_min)
                        st.session_state.preproc_wl_max = float(all_wl_max)
                        st.rerun()
                
                # Update session state
                st.session_state.preproc_wl_min = manual_wl_min
                st.session_state.preproc_wl_max = manual_wl_max
                wavelength_range = (manual_wl_min, manual_wl_max)
                
                st.divider()
                
                # --- BASELINE CORRECTION ---
                st.subheader("📈 Baseline Correction")
                baseline_correction = st.selectbox(
                    "Method",
                    options=["None", "AsLS", "Polynomial", "Rolling Ball"],
                    help="AsLS = Asymmetric Least Squares (recommended for NIR)"
                )
                
                baseline_params = {}
                if baseline_correction == "AsLS":
                    col1, col2 = st.columns(2)
                    with col1:
                        lam = st.selectbox("Smoothness", [1e4, 1e5, 1e6, 1e7], index=2, help="Higher = smoother baseline")
                    with col2:
                        p = st.selectbox("Asymmetry", [0.001, 0.01, 0.1], index=0, help="Lower = follows peaks less")
                    baseline_params = {'lam': float(lam), 'p': float(p)}
                elif baseline_correction == "Polynomial":
                    degree = st.slider("Polynomial Degree", 1, 5, 2)
                    baseline_params = {'degree': degree}
                elif baseline_correction == "Rolling Ball":
                    window_size = st.slider("Window Size", 10, 200, 100, step=10)
                    baseline_params = {'window_size': window_size}
                
                st.divider()
                
                # --- SMOOTHING ---
                st.subheader("🌊 Smoothing")
                apply_smoothing = st.checkbox("Apply Smoothing")
                smooth_method = "savgol"
                smooth_window = 5
                smooth_polyorder = 2
                
                if apply_smoothing:
                    smooth_method = st.selectbox(
                        "Method",
                        ["savgol", "moving_average", "gaussian"],
                        help="Savitzky-Golay recommended for spectral data"
                    )
                    
                    if smooth_method == "savgol":
                        col1, col2 = st.columns(2)
                        with col1:
                            smooth_window = st.slider("Window Size", 3, 21, 5, step=2, help="Must be odd")
                        with col2:
                            smooth_polyorder = st.slider("Polynomial Order", 1, 5, 2, help="Usually 2 or 3")
                    else:
                        smooth_window = st.slider("Window Size", 3, 21, 5, step=2)
                
                st.divider()
                
                # --- DERIVATIVES ---
                st.subheader("📊 Derivatives")
                derivative_order = st.selectbox(
                    "Order",
                    options=[0, 1, 2],
                    index=0,
                    format_func=lambda x: f"{x} (None)" if x == 0 else f"{x} ({'First' if x == 1 else 'Second'} derivative)",
                    help="2nd derivative enhances peaks, reduces baseline effects"
                )
                
                st.divider()
                
                # --- NORMALIZATION ---
                st.subheader("⚖️ Normalization")
                normalization = st.selectbox(
                    "Method",
                    options=["None", "Min-Max", "Standard", "SNV"],
                    help="SNV = Standard Normal Variate (recommended for NIR)"
                )
                
                st.divider()
                
                # --- PROCESSING BUTTONS ---
                st.subheader("🔄 Processing")
                
                # Create processing summary
                processing_steps = []
                if wavelength_range[0] != all_wl_min or wavelength_range[1] != all_wl_max:
                    processing_steps.append(f"Wavelength: {wavelength_range[0]:.0f}-{wavelength_range[1]:.0f} nm")
                if baseline_correction != "None":
                    processing_steps.append(f"Baseline: {baseline_correction}")
                if apply_smoothing:
                    processing_steps.append(f"Smooth: {smooth_method} (w={smooth_window})")
                if derivative_order > 0:
                    processing_steps.append(f"Derivative: {derivative_order}")
                if normalization != "None":
                    processing_steps.append(f"Normalize: {normalization}")
                
                if processing_steps:
                    st.info("**Steps to apply:**\n" + "\n".join([f"• {step}" for step in processing_steps]))
                else:
                    st.warning("No processing steps selected.")
                
                # Processing buttons
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    process_clicked = st.button(
                        "🔄 Process Spectra",
                        disabled=not processing_steps,
                        help=f"Apply processing to {len(selected_files_processing)} files"
                    )
                with col_btn2:
                    if st.button("🗑️ Clear Results"):
                        if 'processed_batch_data' in st.session_state:
                            del st.session_state.processed_batch_data
                            st.success("Processed data cleared!")
                            st.rerun()
                
                # Apply processing if button clicked
                if process_clicked:
                    try:
                        # Prepare processing parameters
                        process_params = {
                            'wavelength_range': wavelength_range,
                            'baseline_correction': baseline_correction.lower() if baseline_correction != "None" else None,
                            'baseline_params': baseline_params if baseline_correction != "None" else None,
                            'smooth': apply_smoothing,
                            'smooth_window': smooth_window if apply_smoothing else None,
                            'smooth_method': smooth_method if apply_smoothing else 'savgol',
                            'smooth_polyorder': smooth_polyorder if apply_smoothing and smooth_method == 'savgol' else 2,
                            'derivative_order': derivative_order,
                            'normalize': normalization.lower() if normalization != "None" else None
                        }
                        
                        # Process all selected files
                        processed_results = []
                        progress_bar = st.progress(0)
                        
                        for i, dataset in enumerate(selected_datasets):
                            # Update progress
                            progress_bar.progress((i + 1) / len(selected_datasets))
                            
                            # Apply processing
                            processed_data = SpectralProcessor.process_spectrum(
                                dataset['data'], **process_params
                            )
                            
                            processed_results.append({
                                'name': dataset['name'],
                                'original': dataset['data'],
                                'processed': processed_data
                            })
                        
                        # Store results in session state
                        st.session_state.processed_batch_data = processed_results
                        st.session_state.processing_params = process_params
                        
                        progress_bar.empty()
                        st.success(f"✅ Successfully processed {len(processed_results)} spectra!")
                        
                    except Exception as e:
                        st.error(f"❌ Processing error: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
            
            # --- RIGHT COLUMN: VISUALIZATION ---
            with col_right:
                if 'processed_batch_data' in st.session_state:
                    processed_results = st.session_state.processed_batch_data
                    processing_params = st.session_state.get('processing_params', {})
                    
                    st.subheader(f"🔍 Results ({len(processed_results)} spectra)")
                    
                    # Visualization options
                    viz_col1, viz_col2 = st.columns(2)
                    with viz_col1:
                        show_original = st.checkbox("Show Original", value=True)
                    with viz_col2:
                        show_processed = st.checkbox("Show Processed", value=True)
                    
                    # Select which files to display
                    files_to_show = st.multiselect(
                        "Files to display",
                        options=[result['name'] for result in processed_results],
                        default=[result['name'] for result in processed_results][:5]  # Default first 5
                    )
                    
                    if files_to_show:
                        # Create comparison plot
                        if processing_params.get('derivative_order', 0) > 0:
                            # Use subplots for derivatives to handle scale differences
                            fig = make_subplots(
                                rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.08,
                                subplot_titles=(
                                    "Original Spectra" if show_original else None,
                                    f"Processed Spectra (Derivative {processing_params['derivative_order']})" if show_processed else None
                                )
                            )
                            
                            colors = px.colors.qualitative.Plotly
                            
                            for i, result in enumerate(processed_results):
                                if result['name'] in files_to_show:
                                    color = colors[i % len(colors)]
                                    
                                    if show_original:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=result['original'].wavelengths,
                                                y=result['original'].intensities,
                                                mode='lines',
                                                name=f"{result['name']} (Orig)",
                                                line=dict(color=color, width=2),
                                                legendgroup=result['name']
                                            ), row=1, col=1
                                        )
                                    
                                    if show_processed:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=result['processed'].wavelengths,
                                                y=result['processed'].intensities,
                                                mode='lines',
                                                name=f"{result['name']} (Proc)",
                                                line=dict(color=color, width=2, dash='dash'),
                                                legendgroup=result['name']
                                            ), row=2, col=1
                                        )
                            
                            fig.update_yaxes(title_text="Intensity", row=1, col=1)
                            fig.update_yaxes(title_text="Processed Intensity", row=2, col=1)
                            fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=1)
                            fig.update_layout(height=700, showlegend=True)
                            
                        else:
                            # Single plot for non-derivative processing
                            fig = go.Figure()
                            colors = px.colors.qualitative.Plotly
                            
                            for i, result in enumerate(processed_results):
                                if result['name'] in files_to_show:
                                    color = colors[i % len(colors)]
                                    
                                    if show_original:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=result['original'].wavelengths,
                                                y=result['original'].intensities,
                                                mode='lines',
                                                name=f"{result['name']} (Original)",
                                                line=dict(color=color, width=2),
                                                opacity=0.7
                                            )
                                        )
                                    
                                    if show_processed:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=result['processed'].wavelengths,
                                                y=result['processed'].intensities,
                                                mode='lines',
                                                name=f"{result['name']} (Processed)",
                                                line=dict(color=color, width=2, dash='dash')
                                            )
                                        )
                            
                            fig.update_layout(
                                title=f"Spectral Preprocessing Comparison",
                                xaxis_title="Wavelength (nm)",
                                yaxis_title="Intensity",
                                height=600,
                                showlegend=True,
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Processing summary
                        st.subheader("📝 Processing Summary")
                        summary_text = []
                        if processing_params.get('wavelength_range'):
                            wl_range = processing_params['wavelength_range']
                            summary_text.append(f"• **Wavelength Range**: {wl_range[0]:.0f} - {wl_range[1]:.0f} nm")
                        
                        if processing_params.get('baseline_correction'):
                            method = processing_params['baseline_correction'].upper()
                            params = processing_params.get('baseline_params', {})
                            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                            summary_text.append(f"• **Baseline Correction**: {method} ({param_str})")
                        
                        if processing_params.get('smooth'):
                            method = processing_params.get('smooth_method', 'savgol')
                            window = processing_params.get('smooth_window', 5)
                            if method == 'savgol':
                                poly = processing_params.get('smooth_polyorder', 2)
                                summary_text.append(f"• **Smoothing**: {method.upper()} (window={window}, poly={poly})")
                            else:
                                summary_text.append(f"• **Smoothing**: {method.upper()} (window={window})")
                        
                        if processing_params.get('derivative_order', 0) > 0:
                            order = processing_params['derivative_order']
                            summary_text.append(f"• **Derivative**: {order}{'st' if order == 1 else 'nd'} order")
                        
                        if processing_params.get('normalize'):
                            method = processing_params['normalize'].upper()
                            summary_text.append(f"• **Normalization**: {method}")
                        
                        st.markdown("\n".join(summary_text))
                        
                        # Export functionality
                        st.subheader("💾 Export")
                        if st.button("💾 Export Processed Data"):
                            try:
                                import io
                                
                                # Create a combined DataFrame with all processed spectra
                                export_data = {}
                                
                                # Add wavelengths (assuming all have same wavelength grid after processing)
                                first_result = processed_results[0]['processed']
                                export_data['Wavelength_nm'] = first_result.wavelengths
                                
                                # Add processed intensities for each file
                                for result in processed_results:
                                    if result['name'] in files_to_show:
                                        export_data[f"{result['name']}_processed"] = result['processed'].intensities
                                        export_data[f"{result['name']}_original"] = result['original'].intensities
                                
                                df = pd.DataFrame(export_data)
                                
                                # Create CSV download
                                csv_buffer = io.StringIO()
                                df.to_csv(csv_buffer, index=False)
                                csv_data = csv_buffer.getvalue()
                                
                                st.download_button(
                                    label="💾 Download as CSV",
                                    data=csv_data,
                                    file_name="processed_spectra.csv",
                                    mime="text/csv"
                                )
                                
                                st.success("✅ Export ready! Click the download button above.")
                                
                            except Exception as e:
                                st.error(f"Export error: {str(e)}")
                    
                    else:
                        st.info("Select files to display in the visualization.")
                
                else:
                    st.info("👈 Configure processing parameters and click 'Process Spectra' to see results here.")
            
        
        with tab3:
            st.header("Advanced Analysis")
            st.info("🚧 Advanced analysis features (PCA, PLS, Neural Networks) will be implemented in future versions.")
            
            st.subheader("Planned Features:")
            st.markdown("""
            - **Principal Component Analysis (PCA)** for dimensionality reduction
            - **Partial Least Squares (PLS)** regression for quantitative analysis
            - **Neural Network models** for prediction of plant properties (%N, etc.)
            - **Spectral library comparison**
            - **Batch processing** for multiple files
            - **Export processed data** to various formats
            """)
    
    else:
        st.info("👈 Please upload a spectral data file to begin analysis.")
        
        # Show supported formats
        st.subheader("📋 Supported File Formats")
        
        supported_formats = SpectralFileReader.supported_formats()
        format_info = {
            '.asd': 'ASD FieldSpec files (binary format)',
            '.spc': 'SPC files (Galactic Industries format)', 
            '.csv': 'Comma-separated values (wavelength, intensity)',
            '.txt': 'Text files (space or tab separated)'
        }
        
        for fmt in supported_formats:
            st.write(f"**{fmt.upper()}**: {format_info.get(fmt, 'Supported format')}")


if __name__ == "__main__":
    main()