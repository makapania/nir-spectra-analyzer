# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Commands

### Development
```bash
# Run the Streamlit web application
streamlit run app.py

# Create and activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run tests with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_file_readers.py
```

### Code Quality
```bash
# Format code
black src/ tests/ app.py

# Check code quality
flake8 src/ tests/ app.py
```

## Architecture Overview

This is a Streamlit-based NIR (Near-Infrared) spectroscopy analysis application with a modular architecture:

### Core Components

**SpectralData Container**: Central data structure (`src/file_readers.py:SpectralData`) that wraps wavelengths, intensities, and metadata. All processing functions operate on this container.

**File Reading Pipeline**: Multi-format file reader (`src/file_readers.py:SpectralFileReader`) with graceful fallbacks:
- Primary: Uses specialized libraries (spectral, spc-spectra)
- Fallback: Custom parsing for edge cases
- Formats: ASD (binary), SPC (Galactic Industries), CSV, TXT

**Processing Pipeline**: Sequential preprocessing (`src/spectral_processing.py:SpectralProcessor`) following spectroscopy best practices:
1. Wavelength range selection
2. Smoothing (Savitzky-Golay, moving average, Gaussian)
3. Derivatives (1st/2nd order using Savitzky-Golay)
4. Normalization (SNV, Min-Max, Standard)

**Web Interface**: Three-tab Streamlit interface (`app.py`):
- Spectrum View: Interactive Plotly visualization with wavelength selection
- Preprocessing: Real-time parameter adjustment with before/after comparison
- Analysis: Placeholder for future ML features (PCA, PLS, Neural Networks)

### Key Architectural Patterns

**Graceful Degradation**: Optional dependencies (spectral, spc-spectra) with informative warnings when unavailable.

**Immutable Processing**: All processing functions return new SpectralData objects rather than modifying existing ones.

**Metadata Preservation**: Processing steps are tracked in metadata for reproducibility.

**Error Handling**: Comprehensive exception handling with user-friendly error messages in the web interface.

### Data Flow
1. File upload → SpectralFileReader → SpectralData container
2. User selects processing parameters → SpectralProcessor → New processed SpectralData
3. Visualization updates automatically via Streamlit reactivity
4. Session state maintains both original and processed data

### File Structure
```
src/
├── file_readers.py      # File format handling and SpectralData container
└── spectral_processing.py  # Preprocessing algorithms and analysis
app.py                   # Streamlit web interface
tests/                   # Unit tests focusing on core data handling
```

## Development Notes

**Testing Strategy**: Core data processing logic is unit tested. UI components rely on manual testing due to Streamlit's architecture.

**Dependency Management**: Optional dependencies for proprietary formats (ASD, SPC) allow the application to run with basic functionality even if specialized libraries are unavailable.

**Future Extensions**: Architecture is designed for ML integration - the SpectralData container can easily be extended with additional metadata for model training/prediction.

**Performance Considerations**: Large spectral files are loaded entirely into memory. For very large datasets, consider streaming or chunked processing approaches.