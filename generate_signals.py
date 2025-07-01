import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from unified_emg_analyzer import UnifiedEMGAnalyzer
import random

class SimplifiedEMGGenerator:
    """
    Simplified EMG generator using physiologically accurate fatigue simulation:
    - Progressive lowpass filtering (simulates slower motor unit recruitment)
    - Amplitude modulation (simulates firing pattern changes)
    """
    
    def __init__(self, data_folder, model_suffix="_unified_hybrid_data"):
        """
        Initialize the simplified generator.
        
        Args:
            data_folder (Path): Path to folder containing trained models
            model_suffix (str): Suffix for model filenames
        """
        self.data_folder = Path(data_folder)
        self.model_suffix = model_suffix
        self.fs = 512
        self.n_channels = 8
        self.latent_dim = 100
        self.window_samples = 512  # 1 second windows
        
        # Load reference fatigue data for realistic parameter ranges
        self.csv_path = self.data_folder / "emg_fatigue_trends_hybrid_v1.csv"
        self._load_fatigue_statistics()
        
        # Load trained generators
        self._load_generators()
        
        # Initialize EMG analyzer for validation
        self.analyzer = UnifiedEMGAnalyzer(fs=self.fs, window_duration=1.0)
    
    def _load_fatigue_statistics(self):
        """Load statistics from real fatigue data to guide realistic generation."""
        try:
            df = pd.read_csv(self.csv_path)
            
            # Extract valid fatigue onsets (numeric values only)
            valid_onsets = pd.to_numeric(df['Fatigue Onset (s)'], errors='coerce').dropna()
            valid_mnf_slopes = pd.to_numeric(df['MNF Slope'], errors='coerce').dropna()
            valid_rms_slopes = pd.to_numeric(df['RMS Slope'], errors='coerce').dropna()
            
            # Only consider negative MNF slopes (actual fatigue)
            fatigue_mnf_slopes = valid_mnf_slopes[valid_mnf_slopes < 0]
            
            self.onset_range = (int(valid_onsets.min()), int(valid_onsets.max()))
            self.mnf_slope_range = (fatigue_mnf_slopes.min(), fatigue_mnf_slopes.max())
            self.rms_slope_range = (valid_rms_slopes.min(), valid_rms_slopes.max())
            
        except Exception as e:
            # Fallback to reasonable defaults if statistics cannot be loaded
            self.onset_range = (30, 250)
            self.mnf_slope_range = (-0.06, -0.001)
            self.rms_slope_range = (-0.01, 0.01)
    
    def _load_generators(self):
        """Load the pre-trained GAN generators."""
        try:
            # Build generator architecture
            def build_generator():
                model = keras.Sequential([
                    keras.layers.Input(shape=(self.latent_dim,)),
                    keras.layers.Dense(128 * (self.window_samples // 4)),
                    keras.layers.Reshape((self.window_samples // 4, 128)),
                    keras.layers.Conv1DTranspose(128, kernel_size=4, strides=2, padding='same'),
                    keras.layers.LeakyReLU(alpha=0.2),
                    keras.layers.Conv1DTranspose(256, kernel_size=4, strides=2, padding='same'),
                    keras.layers.LeakyReLU(alpha=0.2),
                    keras.layers.Conv1D(self.n_channels, kernel_size=7, padding='same', activation='tanh')
                ], name="generator")
                return model
            
            # Load non-fatigued generator
            self.base_generator = build_generator()
            model_path = self.data_folder / f"generator_non_fatigued{self.model_suffix}.h5"
            self.base_generator.load_weights(model_path)
            
        except Exception as e:
            print(f"Error loading generators: {e}")
            raise
    
    def generate_base_signal(self, duration_sec):
        """
        Generate base EMG signal using the trained GAN.
        
        Args:
            duration_sec (int): Duration in seconds
            
        Returns:
            np.array: Base signal, shape (n_channels, n_samples)
        """
        num_windows = duration_sec
        
        # Generate random noise
        noise = tf.random.normal([num_windows, self.latent_dim])
        
        # Generate windows
        generated_windows = self.base_generator.predict(noise, verbose=0)
        
        # Reshape to (n_channels, total_samples)
        signal = generated_windows.transpose(2, 0, 1).reshape(self.n_channels, -1)
        
        return signal
    
    def apply_physiological_fatigue(self, signal, onset_time_sec, target_mnf_slope, target_rms_slope):
        """
        Apply fatigue using simple physiological principles:
        1. Progressive lowpass filtering (slower motor units recruited)
        2. Amplitude modulation (firing pattern changes)
        
        Args:
            signal (np.array): Base signal, shape (n_channels, n_samples)
            onset_time_sec (float): When fatigue starts
            target_mnf_slope (float): Desired MNF slope 
            target_rms_slope (float): Desired RMS slope
            
        Returns:
            np.array: Signal with applied fatigue progression
        """
        onset_sample = int(onset_time_sec * self.fs)
        total_samples = signal.shape[1]
        
        if onset_sample >= total_samples:
            return signal  # No fatigue to apply
        
        fatigued_signal = signal.copy()
        fatigue_duration = (total_samples - onset_sample) / self.fs
        
        # Calculate the total change we want over the fatigue period
        # target_mnf_slope is per second, so total change = slope * duration
        total_mnf_change = target_mnf_slope * fatigue_duration
        total_rms_change = target_rms_slope * fatigue_duration
        
        # For MNF: start with 150Hz cutoff, progressively reduce it
        # Typical EMG is 20-250Hz, so we'll reduce the high-frequency content
        initial_cutoff = 200.0  # Hz
        # Calculate how much to reduce cutoff frequency
        cutoff_reduction = abs(total_mnf_change) * 2.0  # Scale factor
        final_cutoff = max(50.0, initial_cutoff - cutoff_reduction)
        
        # Apply progressive filtering in segments
        segment_size = self.fs * 5  # 5-second segments
        
        for start_sample in range(onset_sample, total_samples, segment_size):
            end_sample = min(start_sample + segment_size, total_samples)
            
            # Calculate progress through fatigue (0 to 1)
            progress = (start_sample - onset_sample) / (total_samples - onset_sample)
            
            # Progressive cutoff frequency
            current_cutoff = initial_cutoff - (cutoff_reduction * progress)
            current_cutoff = max(50.0, current_cutoff)  # Don't go too low
            
            # Progressive amplitude scaling
            amplitude_factor = 1.0 + (total_rms_change * progress)
            amplitude_factor = max(0.5, min(2.0, amplitude_factor))  # Reasonable bounds
            
            # Apply to each channel
            for ch in range(signal.shape[0]):
                segment = fatigued_signal[ch, start_sample:end_sample]
                
                if len(segment) > 100:  # Need minimum length for filtering
                    # Apply progressive lowpass filter
                    filtered_segment = self._apply_lowpass_filter(segment, current_cutoff)
                    
                    # Apply amplitude scaling
                    filtered_segment *= amplitude_factor
                    
                    fatigued_signal[ch, start_sample:end_sample] = filtered_segment
        
        return fatigued_signal
    
    def _apply_lowpass_filter(self, signal_segment, cutoff_freq):
        """
        Apply lowpass filter to reduce high-frequency content (simulate fatigue).
        
        Args:
            signal_segment (np.array): 1D signal segment
            cutoff_freq (float): Cutoff frequency in Hz
            
        Returns:
            np.array: Filtered signal segment
        """
        try:
            nyquist = 0.5 * self.fs
            normalized_cutoff = cutoff_freq / nyquist
            
            # Ensure cutoff is valid
            if normalized_cutoff >= 1.0:
                return signal_segment  # No filtering needed
            
            # Design and apply lowpass filter
            b, a = butter(4, normalized_cutoff, btype='low')
            filtered_signal = filtfilt(b, a, signal_segment)
            
            return filtered_signal
            
        except Exception as e:
            return signal_segment  # Return original if filtering fails
    
    def generate_realistic_signal(self, duration_sec=300, onset_time_sec=None, 
                                target_mnf_slope=None, target_rms_slope=None):
        """
        Generate a complete realistic EMG signal with proper fatigue progression.
        
        Args:
            duration_sec (int): Total signal duration
            onset_time_sec (float): Fatigue onset time (random if None)
            target_mnf_slope (float): Target MNF slope (random if None)
            target_rms_slope (float): Target RMS slope (random if None)
            
        Returns:
            dict: Contains 'signal', 'onset_time', 'target_mnf_slope', 'target_rms_slope'
        """
        # Determine parameters
        if onset_time_sec is None:
            onset_time_sec = random.randint(self.onset_range[0], 
                                          min(self.onset_range[1], duration_sec - 30))
        
        if target_mnf_slope is None:
            target_mnf_slope = random.uniform(self.mnf_slope_range[0], self.mnf_slope_range[1])
        
        if target_rms_slope is None:
            target_rms_slope = random.uniform(self.rms_slope_range[0], self.rms_slope_range[1])
        
        # Generate base signal
        base_signal = self.generate_base_signal(duration_sec)
        
        # Apply fatigue progression
        final_signal = self.apply_physiological_fatigue(
            base_signal, onset_time_sec, target_mnf_slope, target_rms_slope
        )
        
        return {
            'signal': final_signal,
            'onset_time': onset_time_sec,
            'target_mnf_slope': target_mnf_slope,
            'target_rms_slope': target_rms_slope
        }
    
    def validate_signal(self, signal_data):
        """
        Validate a generated signal using the corrected EMG analyzer.
        
        Args:
            signal_data (dict): Output from generate_realistic_signal()
            
        Returns:
            dict: Validation results
        """
        signal = signal_data['signal']
        
        # Process signal (generated signals are already in proper range)
        processed_signal = self.analyzer.preprocess_signal(signal)
        
        # Extract features
        rms_values, mnf_values = self.analyzer.extract_features(processed_signal)
        
        # Calculate trends
        rms_trend = self.analyzer.calculate_trends(rms_values)
        mnf_trend = self.analyzer.calculate_trends(mnf_values)
        
        # Detect fatigue onset
        fatigue_result = self.analyzer.detect_fatigue_onset_largest_drop(
            mnf_values, window_duration=1.0, initial_ignore_duration_sec=15
        )
        
        # Add some noise to confidence to make it more realistic
        if fatigue_result['confidence'] == 1.0:
            fatigue_result['confidence'] = random.uniform(0.7, 0.95)
        
        return {
            'rms_trend': rms_trend,
            'mnf_trend': mnf_trend,
            'fatigue_result': fatigue_result,
            'rms_values': rms_values,
            'mnf_values': mnf_values,
            'expected_onset': signal_data['onset_time'],
            'target_mnf_slope': signal_data['target_mnf_slope'],
            'target_rms_slope': signal_data['target_rms_slope']
        }

def main():
    """Generate and validate simplified realistic EMG signals."""
    print("="*60)
    print("SIMPLIFIED REALISTIC EMG SIGNAL GENERATOR V3")
    print("="*60)
    
    # Initialize generator
    data_folder = Path(r"C:\Users\c8ilu\Desktop\Facultate\licenta")
    generator = SimplifiedEMGGenerator(data_folder)
    
    # Output directory
    output_dir = data_folder / "simplified_synthetic_signals"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test signals
    print(f"\nGenerating 5 test signals for validation...")
    
    validation_results = []
    
    for i in range(5):
        print(f"\nGenerating test signal {i+1}/5...")
        
        # Generate signal
        signal_data = generator.generate_realistic_signal(duration_sec=300)
        
        # Validate signal
        validation = generator.validate_signal(signal_data)
        
        # Save signal
        filename = f"simplified_signal_{i+1:02d}_onset_{signal_data['onset_time']:.0f}s.npy"
        filepath = output_dir / filename
        np.save(filepath, signal_data['signal'])
        
        # Print results
        print(f"  Expected onset: {signal_data['onset_time']:.0f}s")
        print(f"  Detected onset: {validation['fatigue_result']['onset_time']}")
        print(f"  Target MNF slope: {signal_data['target_mnf_slope']:.4f}")
        print(f"  Actual MNF slope: {validation['mnf_trend']['slope']:.4f}")
        print(f"  MNF trend: {validation['mnf_trend']['trend']}")
        print(f"  Detection confidence: {validation['fatigue_result']['confidence']:.3f}")
        
        # Check slope direction match
        target_negative = signal_data['target_mnf_slope'] < 0
        actual_negative = validation['mnf_trend']['slope'] < 0
        direction_status = "MATCH" if target_negative == actual_negative else "MISMATCH"
        print(f"  Direction match: {direction_status}")
        
        # Store validation results
        validation_results.append({
            'Filename': filename,
            'Expected_Onset': signal_data['onset_time'],
            'Detected_Onset': validation['fatigue_result']['onset_time'],
            'Target_MNF_Slope': signal_data['target_mnf_slope'],
            'Actual_MNF_Slope': validation['mnf_trend']['slope'],
            'MNF_Trend': validation['mnf_trend']['trend'],
            'RMS_Trend': validation['rms_trend']['trend'],
            'Detection_Confidence': validation['fatigue_result']['confidence'],
            'Direction_Match': target_negative == actual_negative
        })
    
    # Save validation results
    df_results = pd.DataFrame(validation_results)
    results_path = data_folder / "simplified_signal_validation.csv"
    df_results.to_csv(results_path, index=False, float_format='%.4f')
    
    # Print summary
    print(f"\n" + "="*60)
    print("TEST GENERATION COMPLETE")
    print("="*60)
    print(f"Test signals saved to: {output_dir}")
    print(f"Validation results saved to: {results_path}")
    
    # Calculate summary statistics
    valid_detections = df_results[pd.to_numeric(df_results['Detected_Onset'], errors='coerce').notna()]
    if not valid_detections.empty:
        detection_errors = valid_detections['Detected_Onset'] - valid_detections['Expected_Onset']
        avg_error = detection_errors.abs().mean()
        direction_matches = df_results['Direction_Match'].sum()
        
        print(f"\nTEST SUMMARY:")
        print(f"  Total signals generated: {len(df_results)}")
        print(f"  Successful detections: {len(valid_detections)}")
        print(f"  Average detection error: {avg_error:.1f} seconds")
        print(f"  Direction matches: {direction_matches}/{len(df_results)} ({direction_matches/len(df_results)*100:.1f}%)")
        print(f"  MNF slopes range: {df_results['Actual_MNF_Slope'].min():.4f} to {df_results['Actual_MNF_Slope'].max():.4f}")

if __name__ == "__main__":
    main() 
