"""
Spectral preprocessing and analysis functions for NIR spectroscopy.

This module provides various preprocessing techniques commonly used in
near-infrared spectroscopy including derivatives, smoothing, and normalization.
"""

import numpy as np
from scipy import signal, ndimage
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional, Union
from file_readers import SpectralData


class SpectralProcessor:
    """Class containing static methods for spectral preprocessing."""
    
    @staticmethod
    def calculate_derivative(wavelengths: np.ndarray, intensities: np.ndarray, 
                           order: int = 1) -> tuple:
        """
        Calculate spectral derivatives using Savitzky-Golay filter.
        
        Args:
            wavelengths: Array of wavelength values
            intensities: Array of intensity values
            order: Derivative order (1 or 2)
            
        Returns:
            Tuple of (wavelengths, derivative_intensities)
        """
        if order == 0:
            return wavelengths, intensities
        
        # Use Savitzky-Golay filter for smooth derivatives
        window_length = min(11, len(intensities) // 4)
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(5, window_length)
        
        try:
            derivative = signal.savgol_filter(
                intensities, 
                window_length=window_length, 
                polyorder=3, 
                deriv=order
            )
            return wavelengths, derivative
        except Exception:
            # Fallback to simple numerical derivative
            if order == 1:
                derivative = np.gradient(intensities, wavelengths)
            elif order == 2:
                first_deriv = np.gradient(intensities, wavelengths)
                derivative = np.gradient(first_deriv, wavelengths)
            else:
                raise ValueError(f"Unsupported derivative order: {order}")
            
            return wavelengths, derivative
    
    @staticmethod
    def smooth_spectrum(wavelengths: np.ndarray, intensities: np.ndarray, 
                       window_size: int = 5, method: str = 'savgol') -> tuple:
        """
        Smooth spectral data.
        
        Args:
            wavelengths: Array of wavelength values
            intensities: Array of intensity values
            window_size: Size of smoothing window (must be odd)
            method: Smoothing method ('savgol', 'moving_average', 'gaussian')
            
        Returns:
            Tuple of (wavelengths, smoothed_intensities)
        """
        if window_size % 2 == 0:
            window_size += 1
        
        if method == 'savgol':
            try:
                smoothed = signal.savgol_filter(intensities, window_size, polyorder=2)
            except Exception:
                # Fallback to moving average
                smoothed = SpectralProcessor._moving_average(intensities, window_size)
        elif method == 'moving_average':
            smoothed = SpectralProcessor._moving_average(intensities, window_size)
        elif method == 'gaussian':
            sigma = window_size / 6  # Convert window size to sigma
            smoothed = ndimage.gaussian_filter1d(intensities, sigma)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        
        return wavelengths, smoothed
    
    @staticmethod
    def _moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
        """Simple moving average smoothing."""
        pad_width = window_size // 2
        padded_data = np.pad(data, pad_width, mode='edge')
        smoothed = np.convolve(padded_data, np.ones(window_size)/window_size, mode='valid')
        return smoothed
    
    @staticmethod
    def normalize_spectrum(wavelengths: np.ndarray, intensities: np.ndarray, 
                          method: str = 'minmax') -> tuple:
        """
        Normalize spectral data.
        
        Args:
            wavelengths: Array of wavelength values
            intensities: Array of intensity values
            method: Normalization method ('minmax', 'standard', 'snv', 'msc')
            
        Returns:
            Tuple of (wavelengths, normalized_intensities)
        """
        if method == 'minmax':
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(intensities.reshape(-1, 1)).flatten()
        elif method == 'standard':
            scaler = StandardScaler()
            normalized = scaler.fit_transform(intensities.reshape(-1, 1)).flatten()
        elif method == 'snv':  # Standard Normal Variate
            normalized = SpectralProcessor._snv_normalize(intensities)
        elif method == 'msc':  # Multiplicative Scatter Correction
            # For single spectrum, MSC is similar to mean centering
            normalized = intensities - np.mean(intensities)
            normalized = normalized / np.std(normalized)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return wavelengths, normalized
    
    @staticmethod
    def _snv_normalize(intensities: np.ndarray) -> np.ndarray:
        """Standard Normal Variate normalization."""
        mean_intensity = np.mean(intensities)
        std_intensity = np.std(intensities)
        if std_intensity == 0:
            return intensities - mean_intensity
        return (intensities - mean_intensity) / std_intensity
    
    @staticmethod
    def select_wavelength_range(wavelengths: np.ndarray, intensities: np.ndarray,
                               wl_min: float, wl_max: float) -> tuple:
        """
        Select a specific wavelength range.
        
        Args:
            wavelengths: Array of wavelength values
            intensities: Array of intensity values
            wl_min: Minimum wavelength
            wl_max: Maximum wavelength
            
        Returns:
            Tuple of (selected_wavelengths, selected_intensities)
        """
        mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        return wavelengths[mask], intensities[mask]
    
    @staticmethod
    def process_spectrum(spectral_data: SpectralData, 
                        derivative_order: int = 0,
                        smooth: bool = False,
                        smooth_window: Optional[int] = None,
                        smooth_method: str = 'savgol',
                        normalize: Optional[str] = None,
                        wavelength_range: Optional[tuple] = None) -> SpectralData:
        """
        Apply multiple preprocessing steps to spectral data.
        
        Args:
            spectral_data: SpectralData object
            derivative_order: Order of derivative (0, 1, or 2)
            smooth: Whether to apply smoothing
            smooth_window: Size of smoothing window
            smooth_method: Smoothing method
            normalize: Normalization method
            wavelength_range: Tuple of (min_wl, max_wl) for range selection
            
        Returns:
            Processed SpectralData object
        """
        wavelengths = spectral_data.wavelengths.copy()
        intensities = spectral_data.intensities.copy()
        
        # Apply wavelength range selection first
        if wavelength_range is not None:
            wavelengths, intensities = SpectralProcessor.select_wavelength_range(
                wavelengths, intensities, wavelength_range[0], wavelength_range[1]
            )
        
        # Apply smoothing
        if smooth and smooth_window is not None:
            wavelengths, intensities = SpectralProcessor.smooth_spectrum(
                wavelengths, intensities, smooth_window, smooth_method
            )
        
        # Apply derivative
        if derivative_order > 0:
            wavelengths, intensities = SpectralProcessor.calculate_derivative(
                wavelengths, intensities, derivative_order
            )
        
        # Apply normalization
        if normalize is not None:
            wavelengths, intensities = SpectralProcessor.normalize_spectrum(
                wavelengths, intensities, normalize
            )
        
        # Create new metadata
        new_metadata = spectral_data.metadata.copy()
        new_metadata.update({
            'processed': True,
            'derivative_order': derivative_order,
            'smoothed': smooth,
            'smooth_window': smooth_window if smooth else None,
            'smooth_method': smooth_method if smooth else None,
            'normalized': normalize,
            'wavelength_range': wavelength_range
        })
        
        return SpectralData(wavelengths, intensities, new_metadata)


class SpectralAnalysis:
    """Class for advanced spectral analysis methods."""
    
    @staticmethod
    def find_peaks(spectral_data: SpectralData, height: Optional[float] = None,
                  distance: Optional[int] = None) -> dict:
        """
        Find peaks in spectral data.
        
        Args:
            spectral_data: SpectralData object
            height: Minimum peak height
            distance: Minimum distance between peaks
            
        Returns:
            Dictionary with peak information
        """
        peaks, properties = signal.find_peaks(
            spectral_data.intensities,
            height=height,
            distance=distance
        )
        
        return {
            'peak_indices': peaks,
            'peak_wavelengths': spectral_data.wavelengths[peaks],
            'peak_intensities': spectral_data.intensities[peaks],
            'properties': properties
        }
    
    @staticmethod
    def calculate_area_under_curve(spectral_data: SpectralData, 
                                  wl_range: Optional[tuple] = None) -> float:
        """
        Calculate area under the spectral curve.
        
        Args:
            spectral_data: SpectralData object
            wl_range: Optional wavelength range (min, max)
            
        Returns:
            Area under curve
        """
        if wl_range is not None:
            wavelengths, intensities = SpectralProcessor.select_wavelength_range(
                spectral_data.wavelengths, 
                spectral_data.intensities,
                wl_range[0], wl_range[1]
            )
        else:
            wavelengths = spectral_data.wavelengths
            intensities = spectral_data.intensities
        
        return np.trapz(intensities, wavelengths)