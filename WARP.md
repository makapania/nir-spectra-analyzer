# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Commands

### Development
```bash
# Run the Streamlit web application
streamlit run app.py

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

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

# Run a single test
pytest tests/test_file_readers.py::TestSpectralData::test_spectral_data_creation
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
1. File upload → temporary file → `SpectralFileReader.read_file()` → SpectralData → session state
2. User selects processing parameters → `SpectralProcessor.process_spectrum()` → New processed SpectralData
3. Visualization updates automatically via Streamlit reactivity
4. Session state maintains both original and processed data (`spectral_datasets`, `processed_batch_data`)

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

**Dependency Management**: Optional dependencies for proprietary formats (ASD, SPC) allow the application to run with basic functionality even if specialized libraries are unavailable. Wrap optional imports in try/except with warnings.

**Future Extensions**: Architecture is designed for ML integration - the SpectralData container can easily be extended with additional metadata for model training/prediction.

**Performance Considerations**: Large spectral files are loaded entirely into memory. For very large datasets, consider streaming or chunked processing approaches.

### Important Implementation Details

**ASD File Parsing** (`src/file_readers.py:_read_asd`):
- Binary structure based on SpecDAL's proven byte positions
- Header at bytes 0-483, spectrum data starts at byte 484
- Key positions: version (0:3), wavelength start (191), step (195), channels (204), data format (199)
- Reference spectrum handling for as6/as7/as8 versions
- Calculates percent reflectance when reference available: `target / reference`

**SPC File Handling** (`src/file_readers.py:_read_spc`):
- Prefers spectrochempy with automatic unit conversion (wavenumber → wavelength)
- Manual parser uses fixed 350-2500 nm range (found empirically)
- Applies inversion: `1 - (x/max)` for raw signal data
- Data starts at byte 512 after 512-byte header

**Derivative Calculation** (`src/spectral_processing.py:calculate_derivative`):
- Adaptive window sizing: ~1.5% of data length, clipped to [7, 51], must be odd
- Uses median wavelength spacing for non-uniform grids
- Fallback to `np.gradient` with proper spacing for edge cases

**Baseline Correction** (`src/spectral_processing.py:baseline_correction`):
- AsLS (Asymmetric Least Squares) uses sparse matrix operations - reference: Eilers & Boelens 2005
- Default parameters: `lam=1e6` (smoothness), `p=0.001` (asymmetry)
- Applied before other processing steps to remove instrumental drift

**Session State Management**:
- `spectral_datasets`: List of uploaded file data with original SpectralData
- `processed_batch_data`: Results with both `original` and `processed` SpectralData
- `processing_params`: Dictionary of applied preprocessing parameters
- Wavelength range: `wl_min`, `wl_max`, `preproc_wl_min`, `preproc_wl_max`

### Common Gotchas
- Streamlit reruns entire script on interaction → use session state for persistence
- Savitzky-Golay requires odd window size ≥ polynomial order + 2
- ASD/SPC files may have invalid/missing data → always validate array lengths
- Wavelength arrays may be descending → sort when necessary
- Division by zero in reflectance calculations → use `np.where()` or masking
