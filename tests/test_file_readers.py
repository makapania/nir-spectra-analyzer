"""
Unit tests for the file readers module.
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from file_readers import SpectralData, SpectralFileReader, read_spectrum


class TestSpectralData:
    """Test the SpectralData class."""
    
    def test_spectral_data_creation(self):
        """Test creating a SpectralData object."""
        wavelengths = np.array([400, 500, 600, 700, 800])
        intensities = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        metadata = {'format': 'test'}
        
        data = SpectralData(wavelengths, intensities, metadata)
        
        assert np.array_equal(data.wavelengths, wavelengths)
        assert np.array_equal(data.intensities, intensities)
        assert data.metadata['format'] == 'test'
    
    def test_to_dataframe(self):
        """Test converting to DataFrame."""
        wavelengths = np.array([400, 500, 600])
        intensities = np.array([0.1, 0.2, 0.3])
        
        data = SpectralData(wavelengths, intensities)
        df = data.to_dataframe()
        
        assert len(df) == 3
        assert 'wavelength' in df.columns
        assert 'intensity' in df.columns
        assert df['wavelength'].iloc[0] == 400
        assert df['intensity'].iloc[0] == 0.1


class TestSpectralFileReader:
    """Test the SpectralFileReader class."""
    
    def test_supported_formats(self):
        """Test that supported formats are returned."""
        formats = SpectralFileReader.supported_formats()
        assert '.csv' in formats
        assert '.txt' in formats
        # Note: .asd and .spc might not be available if libraries aren't installed
    
    def test_csv_reading(self):
        """Test reading CSV files."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("wavelength,intensity\n")
            f.write("400,0.1\n")
            f.write("500,0.2\n")
            f.write("600,0.3\n")
            temp_path = f.name
        
        try:
            data = SpectralFileReader.read_file(temp_path)
            
            assert len(data.wavelengths) == 3
            assert len(data.intensities) == 3
            assert data.wavelengths[0] == 400
            assert data.intensities[0] == 0.1
            assert data.metadata['format'] == 'CSV'
            
        finally:
            os.unlink(temp_path)
    
    def test_unsupported_format(self):
        """Test error handling for unsupported formats."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                SpectralFileReader.read_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_convenience_function(self):
        """Test the convenience read_spectrum function."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("400,0.1\n")
            f.write("500,0.2\n")
            temp_path = f.name
        
        try:
            data = read_spectrum(temp_path)
            assert isinstance(data, SpectralData)
            assert len(data.wavelengths) == 2
            
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])