"""
Spectral preprocessing and analysis functions for NIR spectroscopy.

This module provides various preprocessing techniques commonly used in
near-infrared spectroscopy including derivatives, smoothing, and normalization.
"""

import numpy as np
from scipy import signal, ndimage, sparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional, Union, List
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
        
        # Use Savitzky-Golay filter for smooth derivatives.
        # Ensure odd window, >= polyorder+2, and < len(data)
        n = len(intensities)
        if n < 7:
            return wavelengths, np.zeros_like(intensities)
        # Heuristic window ~ 1.5% of length, clipped to [7, 51]
        target = max(7, min(51, int(max(7, round(0.015 * n)))))
        if target % 2 == 0:
            target += 1
        window_length = min(target, n - (1 - (n % 2)))  # keep odd and < n
        polyorder = min(3, window_length - 2)

        # Account for non-uniform wavelength spacing
        diffs = np.diff(wavelengths.astype(float))
        delta = float(np.median(diffs)) if diffs.size else 1.0
        if not np.isfinite(delta) or delta <= 0:
            delta = 1.0
        
        try:
            derivative = signal.savgol_filter(
                intensities.astype(float),
                window_length=window_length,
                polyorder=polyorder,
                deriv=order,
                delta=delta,
                mode='interp'
            )
            return wavelengths, derivative
        except Exception:
            # Fallback to robust numerical derivative with unequal spacing
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
    def baseline_correction(wavelengths: np.ndarray, intensities: np.ndarray,
                           method: str = 'als', **kwargs) -> tuple:
        """
        Apply baseline correction to spectral data.
        
        Args:
            wavelengths: Array of wavelength values
            intensities: Array of intensity values
            method: Baseline correction method ('als', 'polynomial', 'rolling_ball')
            **kwargs: Method-specific parameters
            
        Returns:
            Tuple of (wavelengths, baseline_corrected_intensities)
        """
        if method == 'als':
            # Asymmetric Least Squares baseline correction
            lam = kwargs.get('lam', 1e6)  # smoothness parameter
            p = kwargs.get('p', 0.001)    # asymmetry parameter
            corrected = SpectralProcessor._als_baseline(intensities, lam, p)
        elif method == 'polynomial':
            # Polynomial detrending
            degree = kwargs.get('degree', 2)
            corrected = SpectralProcessor._polynomial_baseline(wavelengths, intensities, degree)
        elif method == 'rolling_ball':
            # Rolling ball baseline correction
            window_size = kwargs.get('window_size', 100)
            corrected = SpectralProcessor._rolling_ball_baseline(intensities, window_size)
        else:
            raise ValueError(f"Unknown baseline correction method: {method}")
        
        return wavelengths, corrected
    
    @staticmethod
    def _als_baseline(y: np.ndarray, lam: float = 1e6, p: float = 0.001, niter: int = 10) -> np.ndarray:
        """
        Asymmetric Least Squares baseline correction.
        
        Reference: Eilers, P. H. C. and Boelens, H. F. M. (2005).
        """
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        
        for i in range(niter):
            W.setdiag(w)  # Do not use np.diag here, it is much slower
            Z = W + lam * D.dot(D.transpose())
            z = sparse.linalg.spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        
        return y - z
    
    @staticmethod
    def _polynomial_baseline(wavelengths: np.ndarray, intensities: np.ndarray, degree: int = 2) -> np.ndarray:
        """Polynomial baseline correction."""
        coeffs = np.polyfit(wavelengths, intensities, degree)
        baseline = np.polyval(coeffs, wavelengths)
        return intensities - baseline
    
    @staticmethod
    def _rolling_ball_baseline(intensities: np.ndarray, window_size: int = 100) -> np.ndarray:
        """Rolling ball baseline correction using morphological operations."""
        from scipy.ndimage import minimum_filter, maximum_filter
        
        # Ensure odd window size
        if window_size % 2 == 0:
            window_size += 1
        
        # Rolling ball is approximated by morphological opening
        # (erosion followed by dilation)
        baseline = minimum_filter(intensities, size=window_size)
        baseline = maximum_filter(baseline, size=window_size)
        
        return intensities - baseline
    
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
                        smooth_polyorder: int = 2,
                        baseline_correction: Optional[str] = None,
                        baseline_params: Optional[dict] = None,
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
            smooth_polyorder: Polynomial order for Savitzky-Golay
            baseline_correction: Baseline correction method ('als', 'polynomial', 'rolling_ball')
            baseline_params: Parameters for baseline correction
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
        
        # Apply baseline correction first (before other processing)
        if baseline_correction is not None:
            baseline_params = baseline_params or {}
            wavelengths, intensities = SpectralProcessor.baseline_correction(
                wavelengths, intensities, baseline_correction, **baseline_params
            )
        
        # Apply smoothing
        if smooth and smooth_window is not None:
            if smooth_method == 'savgol':
                try:
                    intensities = signal.savgol_filter(
                        intensities, smooth_window, smooth_polyorder
                    )
                except Exception:
                    # Fallback to moving average
                    wavelengths, intensities = SpectralProcessor.smooth_spectrum(
                        wavelengths, intensities, smooth_window, 'moving_average'
                    )
            else:
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
            'smooth_polyorder': smooth_polyorder if smooth and smooth_method == 'savgol' else None,
            'baseline_corrected': baseline_correction,
            'baseline_params': baseline_params if baseline_correction else None,
            'normalized': normalize,
            'wavelength_range': wavelength_range
        })
        
        return SpectralData(wavelengths, intensities, new_metadata)
    
    @staticmethod
    def process_batch(datasets: List[SpectralData], **kwargs) -> List[SpectralData]:
        """Apply processing to a list of SpectralData objects."""
        results = []
        for ds in datasets:
            results.append(SpectralProcessor.process_spectrum(ds, **kwargs))
        return results


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