import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import time

# Configuration
DATA_FOLDER = Path(r"C:\\Users\\c8ilu\\Desktop\\Facultate\\licenta")
CSV_FILENAME = "emg_fatigue_trends_hybrid_v1.csv" 
CSV_PATH = DATA_FOLDER / CSV_FILENAME
NPY_PATTERN_DYN = "Sub_*_dinamic_*kg.npy"
NPY_PATTERN_ISO = "Sub_*_isometric_*kg.npy"
SIGNAL_FS = 512  # Sampling frequency
WINDOW_SECONDS = 1 # Duration of windows for GAN training
WINDOW_SAMPLES = int(SIGNAL_FS * WINDOW_SECONDS) # Samples per window
N_CHANNELS = 8 # Expected number of channels in EMG data

# Filtering parameters
BANDPASS_LOW = 20
BANDPASS_HIGH = 250
NOTCH_FREQ = 50
Q_FACTOR = 30

# GAN parameters
LATENT_DIM = 100 
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
ADAM_BETA_1 = 0.5

# Configuration for final generated signal length
OUTPUT_SIGNAL_SECONDS = 5

# Output paths for the unified models
DATA_SUFFIX = "_unified_hybrid_data" 
SAMPLE_SAVE_DIR = DATA_FOLDER / f"gan_samples{DATA_SUFFIX}"
LONG_SAMPLE_SAVE_DIR = SAMPLE_SAVE_DIR / "long_format_samples"
SAMPLE_SAVE_INTERVAL = 10

# Helper Functions

def apply_filters(signal, fs=SIGNAL_FS):
    """Applies bandpass and notch filtering to EMG signal."""
    signal = signal.astype(np.float64)
    b, a = butter(4, [BANDPASS_LOW / (fs / 2), BANDPASS_HIGH / (fs / 2)], btype='band')
    filtered = filtfilt(b, a, signal, axis=1)
    notch_b, notch_a = iirnotch(NOTCH_FREQ / (fs / 2), Q=Q_FACTOR)
    filtered = filtfilt(notch_b, notch_a, filtered, axis=1)
    return filtered

def preprocess_signal(raw_signal):
    """Applies rectification, mean removal, and filtering to EMG signal."""
    if raw_signal.shape[0] != N_CHANNELS:
        print(f"Signal has {raw_signal.shape[0]} channels, expected {N_CHANNELS}. Adapting signal dimensions.")
    
    signal_to_process = raw_signal[:N_CHANNELS, :] if raw_signal.shape[0] >= N_CHANNELS else raw_signal

    rectified_signal = np.abs(signal_to_process)
    mean_removed_signal = rectified_signal - np.mean(rectified_signal, axis=1, keepdims=True)
    filtered_signal = apply_filters(mean_removed_signal)
    return filtered_signal

def load_and_segment_data(csv_path, data_folder):
    """Loads CSV data, processes NPY files, and segments signals for GAN training."""
    print(f"Loading EMG data from {csv_path.name}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}")
        return {}

    data_windows = {
        "non_fatigued": [], 
        "fatigued": []
    }
    
    processed_files = 0
    skipped_files_not_found = 0

    # Determine subject column name
    subject_col = [col for col in df.columns if 'subject' in col.lower()][0]
    
    for _, row in df.iterrows():
        subject_val = row[subject_col]
        movement_type_csv = row['Movement'].lower()
        onset_time_s_val = row['Fatigue Onset (s)']

        # Skip rows where onset is marked as non-decreasing MNF
        if isinstance(onset_time_s_val, str) and 'non-decreasing' in onset_time_s_val.lower():
            continue

        # Convert valid onsets to numeric
        onset_time_sec = pd.to_numeric(onset_time_s_val, errors='coerce')

        if pd.isna(onset_time_sec):
            continue

        # File path resolution
        movement_filename_part = "dinamic" if movement_type_csv == "dynamic" else "isometric"
        glob_pattern = f"Sub_{subject_val}_{movement_filename_part}_*kg.npy"
        potential_files = list(data_folder.glob(glob_pattern))
        
        if not potential_files:
            skipped_files_not_found += 1
            continue
        file_path_to_load = potential_files[0]

        try:
            raw_signal = np.load(file_path_to_load)
            if raw_signal.shape[0] < N_CHANNELS:
                padding = np.zeros((N_CHANNELS - raw_signal.shape[0], raw_signal.shape[1]))
                raw_signal = np.vstack((raw_signal, padding))
            elif raw_signal.shape[0] > N_CHANNELS:
                raw_signal = raw_signal[:N_CHANNELS, :]

            processed_signal = preprocess_signal(raw_signal)
            signal_duration_samples = processed_signal.shape[1]
            onset_sample = int(onset_time_sec * SIGNAL_FS)

            # Signal splitting and segmentation
            signal_non_fatigued_part = None
            signal_fatigued_part = None

            if onset_sample <= 0:
                signal_fatigued_part = processed_signal
            elif onset_sample >= signal_duration_samples:
                signal_non_fatigued_part = processed_signal
            else:
                signal_non_fatigued_part = processed_signal[:, :onset_sample]
                signal_fatigued_part = processed_signal[:, onset_sample:]
            
            # Process non-fatigued windows
            if signal_non_fatigued_part is not None and signal_non_fatigued_part.shape[1] >= WINDOW_SAMPLES:
                num_nf_win = signal_non_fatigued_part.shape[1] // WINDOW_SAMPLES
                truncated_nf = signal_non_fatigued_part[:, :num_nf_win * WINDOW_SAMPLES]
                windows_nf_multi_channel = truncated_nf.reshape(N_CHANNELS, num_nf_win, WINDOW_SAMPLES).transpose(1, 0, 2)
                data_windows["non_fatigued"].extend(windows_nf_multi_channel)

            # Process fatigued windows
            if signal_fatigued_part is not None and signal_fatigued_part.shape[1] >= WINDOW_SAMPLES:
                num_f_win = signal_fatigued_part.shape[1] // WINDOW_SAMPLES
                truncated_f = signal_fatigued_part[:, :num_f_win * WINDOW_SAMPLES]
                windows_f_multi_channel = truncated_f.reshape(N_CHANNELS, num_f_win, WINDOW_SAMPLES).transpose(1, 0, 2)
                data_windows["fatigued"].extend(windows_f_multi_channel)
            
            processed_files += 1
        except Exception as e:
            print(f"Error processing {file_path_to_load.name}: {e}")

    print(f"Data processing complete: {processed_files} files processed, {skipped_files_not_found} files not found")

    prepared_data = {}
    for key, windows_list in data_windows.items():
        if not windows_list:
            print(f"No windows found for {key} data. GAN training for this category will be skipped.")
            prepared_data[key] = np.array([])
            continue
        
        data_array = np.array(windows_list).astype('float32')
        data_reshaped_for_scaling = data_array.reshape(-1, WINDOW_SAMPLES)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scaled_flat = scaler.fit_transform(data_reshaped_for_scaling.T).T
        data_scaled = data_scaled_flat.reshape(data_array.shape[0], N_CHANNELS, WINDOW_SAMPLES)
        
        # Reshape for Conv1D: (batch, steps, channels)
        prepared_data[key] = data_scaled.transpose(0, 2, 1)
        print(f"Prepared {prepared_data[key].shape[0]} windows for {key} data training")

    return prepared_data

# GAN Model Definitions

def build_generator(latent_dim, output_seq_len, n_channels_out):
    """Build generator model for GAN."""
    model = keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(128 * (output_seq_len // 4)),
        layers.Reshape((output_seq_len // 4, 128)),
        layers.Conv1DTranspose(128, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1DTranspose(256, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(n_channels_out, kernel_size=7, padding='same', activation='tanh')
    ], name="generator")
    
    return model

def build_discriminator(input_seq_len, n_channels_in):
    """Build discriminator model for GAN."""
    model = keras.Sequential([
        layers.Input(shape=(input_seq_len, n_channels_in)),
        layers.Conv1D(64, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(128, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ], name="discriminator")
    
    return model

def save_generated_samples(epoch, generator, latent_dim, save_dir, prefix_base, examples=5, fig_size=(10, 5)):
    """Save generated samples for visualization."""
    noise = tf.random.normal([examples, latent_dim])
    generated_samples = generator.predict(noise, verbose=0)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(examples, 1, figsize=fig_size)
    if examples == 1:
        axes = [axes]
    
    for i in range(examples):
        sample = generated_samples[i]
        avg_signal = np.mean(sample, axis=1)
        axes[i].plot(avg_signal)
        axes[i].set_title(f"Sample {i+1}")
    
    plt.tight_layout()
    fig_path = save_dir / f"{prefix_base}_epoch_{epoch}.png"
    plt.savefig(fig_path)
    plt.close()

def train_gan(generator, discriminator, dataset, latent_dim, epochs, batch_size, gan_type_prefix):
    """Train GAN with generator and discriminator."""
    gan = keras.Sequential([generator, discriminator])
    discriminator.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1), 
                         loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False
    gan.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1), 
                loss='binary_crossentropy')
    
    print(f"Training {gan_type_prefix} GAN for {epochs} epochs")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train discriminator
        half_batch = batch_size // 2
        idx = np.random.randint(0, dataset.shape[0], half_batch)
        real_samples = dataset[idx]
        real_labels = np.ones((half_batch, 1))
        
        noise = tf.random.normal([half_batch, latent_dim])
        fake_samples = generator.predict(noise, verbose=0)
        fake_labels = np.zeros((half_batch, 1))
        
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
        avg_disc_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
        
        # Train generator
        discriminator.trainable = False
        noise = tf.random.normal([batch_size, latent_dim])
        real_labels = np.ones((batch_size, 1))
        avg_gen_loss = gan.train_on_batch(noise, real_labels)
        
        epoch_duration = time.time() - epoch_start
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_duration:.2f}s | Gen Loss: {avg_gen_loss:.4f} | Disc Loss: {avg_disc_loss:.4f}")

        if (epoch + 1) % SAMPLE_SAVE_INTERVAL == 0:
            save_generated_samples(epoch + 1, generator, latent_dim, SAMPLE_SAVE_DIR, 
                                 f"{gan_type_prefix}_sample", examples=3)

    return generator, discriminator

def main():
    """Main execution function for GAN training."""
    print("EMG Signal GAN Training System")
    print("="*50)
    
    # Load and prepare data
    data = load_and_segment_data(CSV_PATH, DATA_FOLDER)
    
    trained_generators = {}
    
    for state in ["non_fatigued", "fatigued"]:
        if state not in data or data[state].size == 0:
            print(f"Skipping {state} GAN training - no data available")
            continue
            
        print(f"Training {state} GAN")
        
        # Build models
        generator = build_generator(LATENT_DIM, WINDOW_SAMPLES, N_CHANNELS)
        discriminator = build_discriminator(WINDOW_SAMPLES, N_CHANNELS)
        
        # Train GAN
        trained_gen, _ = train_gan(generator, discriminator, data[state], 
                                 LATENT_DIM, EPOCHS, BATCH_SIZE, state)
        
        # Save trained generator
        model_filename = f"generator_{state}{DATA_SUFFIX}.h5"
        model_path = DATA_FOLDER / model_filename
        trained_gen.save_weights(model_path)
        print(f"Saved {state} generator to {model_filename}")
        
        trained_generators[state] = trained_gen

    print("GAN training completed successfully")

if __name__ == "__main__":
    main()
