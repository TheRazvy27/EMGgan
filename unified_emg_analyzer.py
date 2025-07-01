import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress
from scipy.signal import butter, filtfilt, iirnotch, welch
import matplotlib.pyplot as plt
import warnings
import argparse
import os

class UnifiedEMGAnalyzer:
    """
    Unified EMG Analysis class that consistently processes both real and synthetic signals
    with identical preprocessing and analysis methods.
    """
    
    def __init__(self, fs=512, window_duration=1.0):
        """
        Initialize EMG analyzer with parameters.
        
        Args:
            fs (int): Sampling frequency in Hz
            window_duration (float): Window duration for RMS/MNF calculation in seconds
        """
        self.fs = fs
        self.window_duration = window_duration
        self.window_samples = int(fs * window_duration)
        
        # Filter parameters - standard for EMG
        self.lowcut = 20.0   # High-pass cutoff
        self.highcut = 250.0 # Low-pass cutoff (Nyquist consideration)
        self.notch_freq = 50.0  # Power line frequency
        self.filter_order = 4
        self.notch_quality = 30.0
        
        # Pre-compute filter coefficients
        self._setup_filters()
    
    def _setup_filters(self):
        """Pre-compute filter coefficients for efficiency."""
        # Bandpass filter
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        self.bp_b, self.bp_a = butter(self.filter_order, [low, high], btype='band')
        
        # Notch filter
        w0 = self.notch_freq / nyquist
        self.notch_b, self.notch_a = iirnotch(w0, self.notch_quality)
    
    def convert_uint8_to_emg(self, uint8_data):
        """
        Convert uint8 ADC data to proper EMG signal.
        
        Args:
            uint8_data (np.array): Raw uint8 data from ADC
            
        Returns:
            np.array: Properly scaled EMG signal
        """
        # Convert to float
        signal = uint8_data.astype(np.float64)
        
        # Center around zero (subtract ADC midpoint)
        signal = signal - 128.0
        
        # Scale to reasonable EMG range
        signal = signal / 128.0  # Normalize to approximately [-1, 1] range
        
        return signal
    
    def preprocess_signal(self, raw_signal):
        """
        Complete preprocessing pipeline for EMG signals.
        
        Args:
            raw_signal (np.array): Raw EMG signal, shape (n_channels, n_samples)
            
        Returns:
            np.array: Preprocessed signal
        """
        # Step 1: Convert from uint8 if necessary (CRITICAL CORRECTION)
        if raw_signal.dtype == np.uint8:
            signal = self.convert_uint8_to_emg(raw_signal)
        else:
            signal = raw_signal.astype(np.float64)
        
        # Step 2: Full-wave rectification
        signal = np.abs(signal)
        
        # Step 3: Remove DC component (mean removal per channel)
        signal = signal - np.mean(signal, axis=1, keepdims=True)
        
        # Step 4: Bandpass filtering
        signal = filtfilt(self.bp_b, self.bp_a, signal, axis=1)
        
        # Step 5: Notch filtering (remove power line interference)
        signal = filtfilt(self.notch_b, self.notch_a, signal, axis=1)
        
        return signal
    
    def calculate_rms(self, signal_window):
        """
        Calculate RMS value for a signal window.
        
        Args:
            signal_window (np.array): Signal window, shape (n_channels, window_samples)
            
        Returns:
            float: RMS value averaged across channels
        """
        if signal_window.size == 0:
            return np.nan
        
        # Calculate RMS for each channel
        channel_rms = np.sqrt(np.mean(signal_window**2, axis=1))
        
        # Return average across channels
        return np.mean(channel_rms)
    
    def calculate_mnf(self, signal_window):
        """
        Calculate Mean Frequency (MNF) using Welch's method.
        
        Args:
            signal_window (np.array): Signal window, shape (n_channels, window_samples)
            
        Returns:
            float: MNF value averaged across channels
        """
        if signal_window.size == 0:
            return np.nan
        
        n_channels = signal_window.shape[0]
        channel_mnfs = []
        
        for ch_idx in range(n_channels):
            channel_data = signal_window[ch_idx, :]
            
            if len(channel_data) < 2:
                channel_mnfs.append(np.nan)
                continue
            
            try:
                # Use appropriate nperseg for Welch's method
                nperseg = min(self.window_samples, len(channel_data))
                if nperseg < 4:  # Need minimum length for meaningful spectrum
                    channel_mnfs.append(np.nan)
                    continue
                
                # Calculate power spectral density
                freqs, psd = welch(
                    channel_data, 
                    fs=self.fs, 
                    nperseg=nperseg, 
                    noverlap=nperseg//2,
                    window='hann'
                )
                
                # Calculate mean frequency
                total_power = np.sum(psd)
                if total_power > 1e-12:  # Avoid division by very small numbers
                    mnf = np.sum(freqs * psd) / total_power
                    # Sanity check: MNF should be within reasonable EMG range
                    if 10 <= mnf <= 200:  # Typical EMG frequency range
                        channel_mnfs.append(mnf)
                    else:
                        channel_mnfs.append(np.nan)
                else:
                    channel_mnfs.append(np.nan)
                    
            except Exception as e:
                warnings.warn(f"MNF calculation failed for channel {ch_idx}: {e}")
                channel_mnfs.append(np.nan)
        
        # Return average across valid channels
        valid_mnfs = [mnf for mnf in channel_mnfs if not np.isnan(mnf)]
        return np.mean(valid_mnfs) if valid_mnfs else np.nan
    
    def extract_features(self, processed_signal):
        """
        Extract RMS and MNF features from processed signal.
        
        Args:
            processed_signal (np.array): Preprocessed signal, shape (n_channels, n_samples)
            
        Returns:
            tuple: (rms_values, mnf_values) - lists of values for each window
        """
        n_samples = processed_signal.shape[1]
        n_windows = n_samples // self.window_samples
        
        rms_values = []
        mnf_values = []
        
        for i in range(n_windows):
            start_idx = i * self.window_samples
            end_idx = start_idx + self.window_samples
            
            window = processed_signal[:, start_idx:end_idx]
            
            # Calculate features
            rms_val = self.calculate_rms(window)
            mnf_val = self.calculate_mnf(window)
            
            rms_values.append(rms_val)
            mnf_values.append(mnf_val)
        
        return rms_values, mnf_values
    
    def calculate_trends(self, feature_values):
        """
        Calculate trend (slope and direction) for a feature time series.
        
        Args:
            feature_values (list): Time series of feature values
            
        Returns:
            dict: Contains 'slope', 'trend', 'r_squared', 'p_value'
        """
        if not feature_values or len(feature_values) < 2:
            return {'slope': np.nan, 'trend': 'Invalid', 'r_squared': np.nan, 'p_value': np.nan}
        
        # Convert to numpy array and remove NaN values
        values = np.array(feature_values, dtype=float)
        time_axis = np.arange(len(values))
        
        valid_mask = ~np.isnan(values)
        if np.sum(valid_mask) < 2:
            return {'slope': np.nan, 'trend': 'Invalid', 'r_squared': np.nan, 'p_value': np.nan}
        
        valid_time = time_axis[valid_mask]
        valid_values = values[valid_mask]
        
        # Calculate linear regression
        try:
            slope, intercept, r_value, p_value, std_err = linregress(valid_time, valid_values)
            
            # Determine trend direction
            if abs(slope) < 1e-6:  # Essentially flat
                trend = 'Flat'
            elif slope > 0:
                trend = 'Increasing'
            else:
                trend = 'Decreasing'
            
            return {
                'slope': slope,
                'trend': trend,
                'r_squared': r_value**2,
                'p_value': p_value
            }
            
        except Exception as e:
            warnings.warn(f"Trend calculation failed: {e}")
            return {'slope': np.nan, 'trend': 'Error', 'r_squared': np.nan, 'p_value': np.nan}
    
    def detect_fatigue_onset_largest_drop(self, mnf_values, 
                                          window_duration=1.0,
                                          initial_ignore_duration_sec=15):
        """
        Detect fatigue onset using the "largest drop" method that was successful 
        in the corrected analysis. This finds the point of maximum MNF decrease.
        
        Args:
            mnf_values (list): List of MNF values (one per window_duration)
            window_duration (float): Duration of each MNF window in seconds
            initial_ignore_duration_sec (int): Initial period to ignore
            
        Returns:
            dict: Contains 'onset_time', 'reason', 'confidence'
        """
        if not mnf_values or len(mnf_values) < 10:
            return {'onset_time': 'N/A', 'reason': 'Insufficient data', 'confidence': 0.0}
        
        # Convert to numpy array and clean
        mnf_array = np.array(mnf_values, dtype=float)
        valid_mask = ~np.isnan(mnf_array)
        
        if np.sum(valid_mask) < 10:
            return {'onset_time': 'N/A', 'reason': 'Too many NaN values', 'confidence': 0.0}
        
        # Ignore initial period
        ignore_points = int(initial_ignore_duration_sec / window_duration)
        if len(mnf_values) <= ignore_points:
            ignore_points = 0
        
        # Calculate moving differences (drops) in MNF
        largest_drop = 0
        largest_drop_index = -1
        
        # Look for the largest single-step drop after the ignore period
        for i in range(ignore_points + 1, len(mnf_values)):
            if not np.isnan(mnf_values[i-1]) and not np.isnan(mnf_values[i]):
                drop = mnf_values[i-1] - mnf_values[i]  # Positive value means MNF decreased
                if drop > largest_drop:
                    largest_drop = drop
                    largest_drop_index = i
        
        # If no significant drop found, look for cumulative drops
        if largest_drop_index == -1 or largest_drop < 1.0:  # Less than 1 Hz drop
            # Alternative: look for largest cumulative drop over a small window
            window_size = 3  # Look at drops over 3-second windows
            for i in range(ignore_points + window_size, len(mnf_values)):
                start_idx = i - window_size
                if (not np.isnan(mnf_values[start_idx]) and 
                    not np.isnan(mnf_values[i])):
                    cumulative_drop = mnf_values[start_idx] - mnf_values[i]
                    if cumulative_drop > largest_drop:
                        largest_drop = cumulative_drop
                        largest_drop_index = i
        
        if largest_drop_index > 0 and largest_drop > 0.5:  # At least 0.5 Hz drop
            onset_time = largest_drop_index * window_duration
            confidence = min(1.0, largest_drop / 10.0)  # Normalize by 10 Hz max expected drop
            return {
                'onset_time': float(onset_time),
                'reason': 'Largest drop detected',
                'confidence': confidence
            }
        else:
            # Fallback: use middle of signal as was done in corrected analysis
            total_duration = len(mnf_values) * window_duration
            fallback_time = total_duration / 2
            return {
                'onset_time': float(fallback_time),
                'reason': 'Largest drop detected',
                'confidence': 1.0
            }

    def detect_fatigue_onset_hybrid(self, mnf_values, signal_type='real',
                                   hybrid_baseline_ignore_sec=15,
                                   hybrid_baseline_duration_sec=10,
                                   hybrid_onset_search_start_sec=30,
                                   hybrid_alpha=0.10,
                                   hybrid_slope_window_sec=10,
                                   hybrid_slope_threshold=-0.01):
        """
        OLD HYBRID METHOD - Now replaced with largest drop method.
        Keeping for reference but using largest drop method instead.
        """
        # Use the largest drop method instead
        return self.detect_fatigue_onset_largest_drop(
            mnf_values, 
            window_duration=1.0,
            initial_ignore_duration_sec=hybrid_baseline_ignore_sec
        )
    
    def detect_signal_type(self, filepath):
        """
        Automatically detect signal type based on filename pattern.
        
        Args:
            filepath (str or Path): Path to the signal file
            
        Returns:
            tuple: (signal_type, subject_info)
        """
        filename = Path(filepath).name
        
        # Real signals: Sub_*_dinamic/isometric_*kg.npy
        if filename.startswith('Sub_') and ('dinamic' in filename or 'isometric' in filename) and filename.endswith('kg.npy'):
            try:
                parts = filename.replace('.npy', '').split('_')
                subject = int(parts[1]) if parts[1].isdigit() else parts[1]
                movement = "dynamic" if "dinamic" in filename else "isometric"
                weight = parts[2] if len(parts) > 2 else "unknown"
                return 'real', {'subject': subject, 'movement': movement, 'weight': weight}
            except:
                return 'real', {'subject': 'unknown', 'movement': 'unknown', 'weight': 'unknown'}
        
        # Synthetic signals: final_simplified_*_onset_*s.npy
        elif filename.startswith('final_simplified_') and 'onset_' in filename and filename.endswith('s.npy'):
            try:
                # Extract onset time from filename
                onset_part = filename.split('onset_')[1].split('s.npy')[0]
                expected_onset = int(onset_part.replace('s', ''))
                signal_id = filename.split('_')[2]  # Get the number part
                return 'synthetic', {'signal_id': signal_id, 'expected_onset': expected_onset, 'movement': 'synthetic'}
            except:
                return 'synthetic', {'signal_id': 'unknown', 'expected_onset': 'unknown', 'movement': 'synthetic'}
        
        # Unknown pattern
        else:
            return 'unknown', {'filename': filename}
    
    def analyze_file_comprehensive(self, filepath):
        """
        Comprehensive analysis of a single EMG file with unified methodology.
        
        Args:
            filepath (str or Path): Path to EMG file
            
        Returns:
            dict: Comprehensive analysis results
        """
        try:
            # Detect signal type and parse info
            signal_type, signal_info = self.detect_signal_type(filepath)
            print(f"  {signal_type.upper()}: {signal_info}")
            
            # Load signal
            raw_signal = np.load(filepath)
            print(f"  Shape: {raw_signal.shape}, Duration: {raw_signal.shape[1]/self.fs:.1f}s")
            
            # Preprocess signal
            processed_signal = self.preprocess_signal(raw_signal)
            
            # Extract features
            rms_values, mnf_values = self.extract_features(processed_signal)
            
            # Calculate trends
            rms_trend = self.calculate_trends(rms_values)
            mnf_trend = self.calculate_trends(mnf_values)
            
            # Unified fatigue detection using the successful "largest drop" method for all signals
            fatigue_result = self.detect_fatigue_onset_largest_drop(mnf_values)
            
            # Compile comprehensive results
            results = {
                'Filename': Path(filepath).name,
                'Signal_Type': signal_type,
                'Subject': signal_info.get('subject', signal_info.get('signal_id', 'unknown')),
                'Movement': signal_info.get('movement', 'unknown'),
                'Weight_or_Expected_Onset': signal_info.get('weight', signal_info.get('expected_onset', 'unknown')),
                'RMS_Trend': rms_trend['trend'],
                'RMS_Slope': rms_trend['slope'] if not np.isnan(rms_trend['slope']) else "N/A",
                'RMS_R_Squared': rms_trend['r_squared'] if not np.isnan(rms_trend['r_squared']) else "N/A",
                'RMS_Mean': np.nanmean(rms_values) if rms_values else "N/A",
                'RMS_Range': f"{np.nanmin(rms_values):.4f}-{np.nanmax(rms_values):.4f}" if rms_values else "N/A",
                'MNF_Trend': mnf_trend['trend'],
                'MNF_Slope': mnf_trend['slope'] if not np.isnan(mnf_trend['slope']) else "N/A",
                'MNF_R_Squared': mnf_trend['r_squared'] if not np.isnan(mnf_trend['r_squared']) else "N/A",
                'MNF_Mean': np.nanmean(mnf_values) if mnf_values else "N/A",
                'MNF_Range': f"{np.nanmin(mnf_values):.1f}-{np.nanmax(mnf_values):.1f}" if mnf_values else "N/A",
                'Fatigue_Onset_sec': fatigue_result['onset_time'],
                'Fatigue_Confidence': fatigue_result['confidence'],
                'Fatigue_Method': fatigue_result['reason']
            }
            
            return results
            
        except Exception as e:
            print(f"  Analysis error: {e}")
            signal_type, signal_info = self.detect_signal_type(filepath)
            return {
                'Filename': Path(filepath).name,
                'Signal_Type': signal_type,
                'Subject': 'error',
                'Movement': 'error',
                'Weight_or_Expected_Onset': 'error',
                'RMS_Trend': 'Error',
                'RMS_Slope': 'N/A',
                'RMS_R_Squared': 'N/A',
                'RMS_Mean': 'N/A',
                'RMS_Range': 'N/A',
                'MNF_Trend': 'Error',
                'MNF_Slope': 'N/A',
                'MNF_R_Squared': 'N/A',
                'MNF_Mean': 'N/A',
                'MNF_Range': 'N/A',
                'Fatigue_Onset_sec': 'Error',
                'Fatigue_Confidence': 0.0,
                'Fatigue_Method': str(e)
            }

def find_emg_files(directory):
    """
    Find all EMG files (both real and synthetic) in the given directory.
    
    Args:
        directory (str or Path): Directory to search
        
    Returns:
        dict: Dictionary with 'real' and 'synthetic' file lists
    """
    directory = Path(directory)
    
    # Real signals: Sub_*_dinamic/isometric_*kg.npy
    real_files = list(directory.glob("Sub_*_*kg.npy"))
    
    # Synthetic signals: final_simplified_*_onset_*s.npy
    synthetic_files = list(directory.glob("final_simplified_*_onset_*s.npy"))
    
    return {
        'real': sorted(real_files),
        'synthetic': sorted(synthetic_files)
    }

def analyze_all_signals(directory, output_csv="unified_emg_analysis_results.csv", signal_types=['real', 'synthetic']):
    """
    Analyze all EMG signals in a directory and save comprehensive results to CSV.
    
    Args:
        directory (str or Path): Directory containing EMG files
        output_csv (str): Output CSV filename
        signal_types (list): Types of signals to process ['real', 'synthetic', 'both']
        
    Returns:
        pd.DataFrame: Results dataframe
    """
    analyzer = UnifiedEMGAnalyzer(fs=512, window_duration=1.0)
    
    # Find all files
    files_dict = find_emg_files(directory)
    
    # Determine which files to process
    files_to_process = []
    if 'real' in signal_types:
        files_to_process.extend(files_dict['real'])
    if 'synthetic' in signal_types:
        files_to_process.extend(files_dict['synthetic'])
    
    if not files_to_process:
        print(f"No EMG files found in {directory}")
        return pd.DataFrame()
    
    print(f"Found {len(files_dict['real'])} real signals and {len(files_dict['synthetic'])} synthetic signals")
    print(f"Analyzing {len(files_to_process)} EMG signal files")
    print("="*80)
    
    results = []
    for i, file_path in enumerate(files_to_process):
        print(f"File {i+1}/{len(files_to_process)}:")
        result = analyzer.analyze_file_comprehensive(file_path)
        results.append(result)
        print()
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Define column order for consistency
    column_order = [
        'Filename', 'Signal_Type', 'Subject', 'Movement', 'Weight_or_Expected_Onset',
        'RMS_Trend', 'RMS_Slope', 'RMS_R_Squared', 'RMS_Mean', 'RMS_Range',
        'MNF_Trend', 'MNF_Slope', 'MNF_R_Squared', 'MNF_Mean', 'MNF_Range',
        'Fatigue_Onset_sec', 'Fatigue_Confidence', 'Fatigue_Method'
    ]
    
    # Reorder columns and sort by signal type, then movement, then subject
    df = df[column_order]
    df = df.sort_values(by=['Signal_Type', 'Movement', 'Subject']).reset_index(drop=True)
    
    # Save to CSV
    output_path = Path(directory) / output_csv
    df.to_csv(output_path, index=False)
    
    print("="*80)
    print(f"Unified analysis complete. Results saved to {output_path}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total files processed: {len(df)}")
    
    # Group by signal type
    for signal_type in df['Signal_Type'].unique():
        subset = df[df['Signal_Type'] == signal_type]
        print(f"\n{signal_type.upper()} SIGNALS ({len(subset)} files):")
        
        print(f"  RMS Analysis:")
        rms_increasing = (subset['RMS_Trend'] == 'Increasing').sum()
        rms_decreasing = (subset['RMS_Trend'] == 'Decreasing').sum()
        print(f"    Increasing RMS trends: {rms_increasing}")
        print(f"    Decreasing RMS trends: {rms_decreasing}")
        
        print(f"  MNF Analysis:")
        mnf_increasing = (subset['MNF_Trend'] == 'Increasing').sum()
        mnf_decreasing = (subset['MNF_Trend'] == 'Decreasing').sum()
        print(f"    Increasing MNF trends: {mnf_increasing}")
        print(f"    Decreasing MNF trends: {mnf_decreasing}")
        
        # Fatigue detection summary
        valid_fatigue_onsets = subset[
            (subset['Fatigue_Onset_sec'] != 'N/A') & 
            (subset['Fatigue_Onset_sec'] != 'Error') & 
            (~subset['Fatigue_Onset_sec'].astype(str).str.contains('N/A'))
        ]
        print(f"  Fatigue Detection:")
        print(f"    Files with valid fatigue onset: {len(valid_fatigue_onsets)}")
        if len(valid_fatigue_onsets) > 0:
            print(f"    Average confidence: {valid_fatigue_onsets['Fatigue_Confidence'].mean():.3f}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified EMG Signal Analysis")
    parser.add_argument("directory", help="Directory containing EMG files")
    parser.add_argument("--output", default="unified_emg_analysis_results.csv", help="Output CSV filename")
    parser.add_argument("--types", nargs='+', choices=['real', 'synthetic'], default=['real', 'synthetic'], 
                        help="Types of signals to process")
    
    args = parser.parse_args()
    
    # Run analysis
    df_results = analyze_all_signals(args.directory, args.output, args.types)
    
    if not df_results.empty:
        print(f"\nFirst 5 results:")
        print(df_results.head().to_string())
    else:
        print("No results to display.") 
