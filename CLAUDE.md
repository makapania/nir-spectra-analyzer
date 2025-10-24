# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
```bash
# Run the Streamlit web application
streamlit run app.py

# Create and activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Install test dependencies (if not already in requirements.txt)
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

This is a Streamlit-based NIR (Near-Infrared) spectroscopy analysis application with a modular, functional architecture designed for interactive spectral data processing and visualization.

### Core Components

**SpectralData Container** (`src/file_readers.py:44-58`): Immutable data structure wrapping:
- `wavelengths`: NumPy array of wavelength values (typically in nm)
- `intensities`: NumPy array of spectral intensities/reflectance values
- `metadata`: Dictionary for format info, processing history, and instrument parameters
- All processing operations return new SpectralData instances rather than modifying in place

**File Reading Pipeline** (`src/file_readers.py:61-424`): Multi-format reader with graceful fallbacks:
- **ASD files** (`_read_asd`): Binary format parsing with reference spectrum handling
  - Primary: `spectral` library
  - Fallback: Manual binary parsing following SpecDAL approach
  - Handles versions: ASD, asd, as6, as7, as8
  - Computes reflectance from target/reference when available
- **SPC files** (`_read_spc`): Galactic Industries format
  - Primary: `spectrochempy` library with unit conversion
  - Fallback: Manual header parsing with fixed wavelength range (350-2500 nm)
  - Applies linear inversion for raw signal data
- **CSV/TXT files** (`_read_csv`): Auto-detects separators (comma, tab, space)
- All readers raise informative `ValueError` on failure

**Processing Pipeline** (`src/spectral_processing.py:15-348`): Sequential preprocessing via `SpectralProcessor.process_spectrum`:
1. **Wavelength range selection** (`select_wavelength_range`): Focus on specific regions
2. **Baseline correction** (`baseline_correction`):
   - AsLS (Asymmetric Least Squares) - recommended for NIR
   - Polynomial detrending
   - Rolling ball (morphological)
3. **Smoothing** (`smooth_spectrum`):
   - Savitzky-Golay (preserves features, recommended)
   - Moving average
   - Gaussian filter
4. **Derivatives** (`calculate_derivative`):
   - 1st/2nd order via Savitzky-Golay filter
   - Adaptive window sizing based on data length
   - Accounts for non-uniform wavelength spacing
5. **Normalization** (`normalize_spectrum`):
   - SNV (Standard Normal Variate) - recommended for NIR
   - Min-Max scaling
   - Standard scaling
   - MSC (Multiplicative Scatter Correction)

**Web Interface** (`app.py`): Three-tab Streamlit application:
- **Spectrum View** (tab1): Multi-file visualization with Plotly
  - File selection and color palette controls
  - Interactive wavelength range controls (manual input + slider + preset buttons)
  - Session state manages `wl_min`, `wl_max` for persistence
- **Preprocessing** (tab2): Batch processing interface
  - Left column: Parameter controls and processing summary
  - Right column: Before/after visualization with subplot handling for derivatives
  - Session state stores `processed_batch_data` and `processing_params`
  - Export functionality for CSV download
- **Analysis** (tab3): Placeholder for future ML features (PCA, PLS, Neural Networks)

### Key Architectural Patterns

**Graceful Degradation**: Optional dependencies with warnings when unavailable:
- `spectral` library for ASD files → falls back to manual parsing
- `spectrochempy`/`galvani` for SPC files → falls back to manual parsing
- Application runs with basic CSV/TXT support even without specialized libraries

**Immutable Processing**: All `SpectralProcessor` methods return new data structures. Original data preserved in session state for comparison and reverting.

**Metadata Tracking**: Processing history stored in `metadata` dict:
- Processing flags: `processed`, `smoothed`, `baseline_corrected`, `normalized`
- Parameters: `derivative_order`, `smooth_window`, `smooth_method`, `wavelength_range`
- Enables reproducibility and troubleshooting

**Session State Management**: Streamlit session state stores:
- `spectral_datasets`: List of uploaded file data with original SpectralData
- `processed_batch_data`: Results with both `original` and `processed` SpectralData
- `processing_params`: Dictionary of applied preprocessing parameters
- Wavelength range state: `wl_min`, `wl_max`, `preproc_wl_min`, `preproc_wl_max`

**Error Handling**: Multi-layer approach:
- Library-level: Try specialized libraries first, fall back to manual parsing
- Processing-level: Catch exceptions in Savitzky-Golay, fall back to simpler methods
- UI-level: Display user-friendly error messages with traceback in Streamlit

### Data Flow

1. **Upload**: File → temporary file → `SpectralFileReader.read_file()` → `SpectralData` → session state
2. **Batch Processing**: User selects files + parameters → `SpectralProcessor.process_spectrum()` for each → results in session state
3. **Visualization**: Session state data → Plotly figures → Streamlit display
4. **Export**: Processed data → pandas DataFrame → CSV download button

### Important Implementation Details

**ASD File Parsing** (`src/file_readers.py:91-214`):
- Binary structure based on SpecDAL's proven byte positions
- Header at bytes 0-483, spectrum data starts at byte 484
- Key positions: version (0:3), wavelength start (191), step (195), channels (204), data format (199)
- Reference spectrum handling for as6/as7/as8 versions
- Calculates percent reflectance when reference available

**SPC File Handling** (`src/file_readers.py:217-381`):
- Prefers spectrochempy with automatic unit conversion (wavenumber → wavelength)
- Manual parser uses fixed 350-2500 nm range (found empirically)
- Applies inversion: `1 - (x/max)` for raw signal data
- Data starts at byte 512 after 512-byte header

**Derivative Calculation** (`src/spectral_processing.py:18-72`):
- Adaptive window sizing: ~1.5% of data length, clipped to [7, 51], must be odd
- Uses median wavelength spacing for non-uniform grids
- Fallback to `np.gradient` with proper spacing for edge cases

**Baseline Correction** (`src/spectral_processing.py:156-230`):
- AsLS uses sparse matrix operations for efficiency (reference: Eilers & Boelens 2005)
- Default parameters: `lam=1e6` (smoothness), `p=0.001` (asymmetry)
- Applied before other processing steps to remove instrumental drift

### File Structure
```
src/
├── __init__.py
├── file_readers.py          # SpectralData container + multi-format readers
└── spectral_processing.py   # SpectralProcessor + SpectralAnalysis classes
app.py                       # Streamlit web interface (main entry point)
tests/
└── test_file_readers.py     # Unit tests for core data handling
requirements.txt             # All dependencies (including optional ones)
```

## Development Best Practices

**Adding New File Formats**:
1. Add static method `_read_format()` to `SpectralFileReader`
2. Update `read_file()` to route to new method
3. Update `supported_formats()` to include new extension
4. Return `SpectralData` with appropriate metadata
5. Add error handling with informative messages

**Adding New Processing Methods**:
1. Add static method to `SpectralProcessor` class
2. Method signature: `(wavelengths, intensities, **params) -> (wavelengths, intensities)`
3. Add to `process_spectrum()` pipeline if appropriate
4. Update metadata tracking
5. Consider UI integration in `app.py` tab2

**Testing Strategy**:
- Unit tests focus on core data processing logic (file reading, SpectralData container)
- Processing methods tested implicitly through integration
- UI components require manual testing due to Streamlit architecture
- Use temporary files for file I/O tests

**Dependency Management**:
- Required: numpy, pandas, scipy, scikit-learn, matplotlib, plotly, streamlit
- Optional: spectral, spc-spectra, spectrochempy, galvani (for proprietary formats)
- Wrap optional imports in try/except with warnings
- Always provide fallback implementations

**Common Gotchas**:
- Streamlit reruns entire script on interaction → use session state for persistence
- Savitzky-Golay requires odd window size ≥ polynomial order + 2
- ASD/SPC files may have invalid/missing data → always validate array lengths
- Wavelength arrays may be descending → sort when necessary
- Division by zero in reflectance calculations → use `np.where()` or masking

## Future Development Roadmap

**Phase 1**: Core analysis features (peak detection, area under curve - partially implemented in `SpectralAnalysis` class)

**Phase 2**: Machine learning integration (PCA, PLS regression - scikit-learn already in requirements)

**Phase 3**: Neural network models for plant property prediction (will require TensorFlow/PyTorch)

**Phase 4**: Production features (authentication, API, Docker, cloud deployment)

Architecture is designed to accommodate these features through the extensible SpectralData container and modular processing pipeline.
