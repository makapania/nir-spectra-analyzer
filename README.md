# NIR Spectra Analyzer

A comprehensive web-based application for near-infrared (NIR) spectral data analysis, preprocessing, and machine learning applications.

## 🎯 Overview

The NIR Spectra Analyzer is designed to handle the complete workflow of NIR spectroscopy data analysis, from reading proprietary file formats to advanced preprocessing and analysis. Built with Streamlit for easy deployment and interactive use.

### Key Features

- **Multi-format File Support**: Read ASD, SPC, CSV, and TXT files
- **Interactive Visualization**: Plotly-based interactive spectral plots with zoom, pan, and selection
- **Advanced Preprocessing**: Derivatives, smoothing, normalization (SNV, Min-Max, Standard)
- **Wavelength Selection**: Focus on specific spectral regions
- **Browser-based Interface**: No complex installation required for end users

### Planned Features (Roadmap)

- **Principal Component Analysis (PCA)** for dimensionality reduction
- **Partial Least Squares (PLS)** regression for quantitative analysis  
- **Neural Network models** for prediction of plant properties (%N, moisture, etc.)
- **Spectral library comparison** and identification
- **Batch processing** for multiple files
- **Export functionality** for processed data
- **Database integration** for spectral data management

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/nir-spectra-analyzer.git
   cd nir-spectra-analyzer
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📊 Usage

### Basic Workflow

1. **Upload Data**: Use the sidebar to upload your spectral data file (ASD, SPC, CSV, or TXT)
2. **Visualize**: View your spectrum with interactive controls for wavelength range selection
3. **Preprocess**: Apply various preprocessing techniques:
   - **Derivatives**: Calculate 1st or 2nd derivatives using Savitzky-Golay filters
   - **Smoothing**: Reduce noise with Savitzky-Golay, moving average, or Gaussian filters
   - **Normalization**: Apply SNV, Min-Max, or Standard normalization
4. **Analyze**: Compare original vs processed spectra side-by-side

### File Format Support

#### ASD Files (.asd)
- **Format**: ASD FieldSpec binary format
- **Library**: Uses `spectral` library for reading
- **Notes**: Proprietary format from ASD spectrometers; most comprehensive spectral information

#### SPC Files (.spc)
- **Format**: Galactic Industries SPC format
- **Library**: Uses `spc-spectra` library
- **Notes**: Common format for various spectrometer manufacturers

#### CSV/TXT Files (.csv, .txt)
- **Format**: Two-column format (wavelength, intensity)
- **Separators**: Comma, tab, or space-separated
- **Notes**: Most flexible format for data exchange

### Example Data Processing

```python
from src.file_readers import read_spectrum
from src.spectral_processing import SpectralProcessor

# Read spectral data
spectrum = read_spectrum('sample.asd')

# Apply preprocessing
processed = SpectralProcessor.process_spectrum(
    spectrum,
    derivative_order=1,      # First derivative
    smooth=True,            # Apply smoothing
    smooth_window=11,       # Window size
    normalize='snv'         # SNV normalization
)

# Access processed data
wavelengths = processed.wavelengths
intensities = processed.intensities
```

## 🧪 Development

### Project Structure

```
nir-spectra-analyzer/
├── src/                    # Source code
│   ├── __init__.py
│   ├── file_readers.py     # File format readers
│   └── spectral_processing.py  # Preprocessing functions
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Running Tests

```bash
# Install development dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run tests with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Check code quality
flake8 src/ tests/
```

## 🔬 Technical Details

### ASD File Reading

ASD files are proprietary binary formats that can be challenging to read. The application uses multiple approaches:

1. **Primary Method**: `spectral` library - Most reliable for standard ASD files
2. **Fallback Methods**: Custom binary parsing for edge cases
3. **Error Handling**: Graceful degradation with informative error messages

### Preprocessing Pipeline

The preprocessing pipeline follows spectroscopy best practices:

1. **Wavelength Range Selection**: Applied first to focus on regions of interest
2. **Smoothing**: Noise reduction while preserving spectral features
3. **Derivatives**: Enhanced feature detection and baseline removal
4. **Normalization**: Standardization for comparison and analysis

### Web Deployment

Built with Streamlit for several advantages:
- **Easy Deployment**: Single command to start the application
- **Interactive Components**: Built-in widgets for user interaction
- **Real-time Updates**: Immediate feedback as parameters change
- **Browser Compatibility**: Works on any modern web browser

## 🚀 Future Development

### Phase 1: Core Analysis Features
- [ ] Spectral peak detection and annotation
- [ ] Area under curve calculations
- [ ] Spectral similarity metrics
- [ ] Export processed data (CSV, Excel)

### Phase 2: Machine Learning Integration
- [ ] Principal Component Analysis (PCA)
- [ ] Partial Least Squares (PLS) regression
- [ ] Cross-validation tools
- [ ] Model performance metrics

### Phase 3: Advanced Features
- [ ] Neural network models for spectral analysis
- [ ] Plant property prediction (%N, moisture, etc.)
- [ ] Spectral database integration
- [ ] Multi-file batch processing

### Phase 4: Production Features
- [ ] User authentication and data management
- [ ] API endpoints for programmatic access
- [ ] Docker containerization
- [ ] Cloud deployment options

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation for new features
- Ensure backward compatibility

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Spectral Analysis Community**: For open-source tools and algorithms
- **ASD and SPC Libraries**: Essential for proprietary format support
- **Streamlit Team**: For the excellent web application framework
- **Scientific Python Ecosystem**: NumPy, SciPy, Pandas, and Scikit-learn

## 📞 Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/yourusername/nir-spectra-analyzer/issues)
- **Documentation**: Comprehensive guides in the `docs/` directory
- **Examples**: Sample data and notebooks in `examples/`

---

**Built with ❤️ for the spectroscopy community**