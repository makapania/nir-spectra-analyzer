"""
NIR Spectra Analyzer - Main Web Application

A Streamlit-based web application for near-infrared spectral data analysis.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from file_readers import SpectralFileReader, SpectralData
from spectral_processing import SpectralProcessor


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
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Processing Options")
                
                # File selection for preprocessing
                selected_file_for_processing = st.selectbox(
                    "Select file to process",
                    options=[dataset['name'] for dataset in spectral_datasets],
                    help="Choose which spectrum to apply preprocessing to"
                )
                
                # Get the selected dataset
                selected_dataset = next(d for d in spectral_datasets if d['name'] == selected_file_for_processing)
                spectral_data = selected_dataset['data']
                
                # Derivative options
                derivative_order = st.selectbox(
                    "Derivative Order",
                    options=[0, 1, 2],
                    index=0,
                    help="0 = No derivative, 1 = First derivative, 2 = Second derivative"
                )
                
                # Smoothing options
                apply_smoothing = st.checkbox("Apply Smoothing")
                if apply_smoothing:
                    window_size = st.slider("Smoothing Window", 3, 21, 5, step=2)
                
                # Normalization options
                normalization = st.selectbox(
                    "Normalization",
                    options=["None", "Min-Max", "Standard", "SNV"],
                    help="SNV = Standard Normal Variate"
                )
                
                # Processing buttons
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("🔄 Apply Processing"):
                        try:
                            processed_data = SpectralProcessor.process_spectrum(
                                spectral_data,
                                derivative_order=derivative_order,
                                smooth=apply_smoothing,
                                smooth_window=window_size if apply_smoothing else None,
                                normalize=normalization.lower() if normalization != "None" else None
                            )
                            st.session_state.processed_data = processed_data
                            st.success("✅ Processing applied!")
                        except Exception as e:
                            st.error(f"❌ Processing error: {str(e)}")
                
                with col_btn2:
                    if st.button("🗑️ Clear Processing", help="Clear processed data"):
                        if 'processed_data' in st.session_state:
                            del st.session_state.processed_data
                            st.success("Processed data cleared!")
                            st.rerun()
            
            with col2:
                if 'processed_data' in st.session_state:
                    processed_data = st.session_state.processed_data
                    
                    # Plot comparison
                    fig = go.Figure()
                    
                    # Original spectrum
                    fig.add_trace(go.Scatter(
                        x=spectral_data.wavelengths,
                        y=spectral_data.intensities,
                        mode='lines',
                        name='Original',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Processed spectrum
                    fig.add_trace(go.Scatter(
                        x=processed_data.wavelengths,
                        y=processed_data.intensities,
                        mode='lines',
                        name='Processed',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Original vs Processed Spectrum",
                        xaxis_title="Wavelength (nm)",
                        yaxis_title="Intensity",
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
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