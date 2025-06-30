import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.signal import welch, butter, filtfilt, iirnotch, spectrogram
from scipy.stats import linregress

def apply_filters(signal, fs=512):
    b, a = butter(4, [20 / (fs / 2), 250 / (fs / 2)], btype='band')
    filtered = filtfilt(b, a, signal, axis=1)
    notch_b, notch_a = iirnotch(50 / (fs / 2), Q=30)
    filtered = filtfilt(notch_b, notch_a, filtered, axis=1)
    return filtered

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
    if file_path:
        try:
            global data
            raw_data = np.load(file_path).astype(np.float64)

            # Check for minimum signal length required by the filter
            # The filter needs len(signal) > 3 * filter_order.
            # Our bandpass is 4th order (creating 8th order filter), so len > 3*9=27
            if raw_data.shape[1] <= 27:
                messagebox.showerror("Error", f"Signal is too short to be processed.\n\nFile: {file_path}\nSignal length is {raw_data.shape[1]} samples, but must be > 27.")
                return

            if len(raw_data.shape) not in [2, 3]:
                messagebox.showerror("Error", "Invalid file dimensions.")
                return
            raw_data = np.abs(raw_data)  # Full-wave rectification
            raw_data -= np.mean(raw_data, axis=1, keepdims=True)
            data = apply_filters(raw_data)
            messagebox.showinfo("Success", "File loaded and filtered successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

def display_signals():
    try:
        if data is None:
            messagebox.showerror("Error", "No data loaded.")
            return
        fig, axs = plt.subplots(data.shape[0], 1, figsize=(10, 6))
        fig.suptitle("Filtered EMG Signals (All Channels)", fontsize=16)
        for i in range(data.shape[0]):
            axs[i].plot(data[i], label=f'Channel {i+1}')
            axs[i].set_ylabel(f'Channel {i+1}')
            axs[i].grid(True)
            axs[i].legend()
        axs[-1].set_xlabel("Time (Samples)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def compute_max_amplitude():
    try:
        max_amplitude = np.max(np.abs(data), axis=1)
        max_amp_text.delete("1.0", tk.END)
        max_amp_text.insert(tk.END, "\n".join([f"Channel {i+1}: {amp:.2f}" for i, amp in enumerate(max_amplitude)]))
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def compute_rms():
    try:
        sampling_rate = 512
        segment_length = sampling_rate
        rms_values = []
        slopes = []
        for i in range(data.shape[0]):
            rms_per_segment = [np.sqrt(np.mean(data[i, j:j+segment_length]**2))
                               for j in range(0, data.shape[1], segment_length)]
            rms_values.append(rms_per_segment)
            x = np.arange(len(rms_per_segment))
            slope, _, _, _, _ = linregress(x, rms_per_segment)
            slopes.append(slope)

        rms_slope_text.delete("1.0", tk.END)
        rms_slope_text.insert(tk.END, "\n".join([f"Channel {i+1}: {slope:.6f}" for i, slope in enumerate(slopes)]))

        fig, axs = plt.subplots(len(rms_values), 1, figsize=(10, 6))
        fig.suptitle("RMS Values with Linear Regression", fontsize=16)
        for i in range(len(rms_values)):
            x = np.arange(len(rms_values[i]))
            axs[i].plot(x, rms_values[i], label=f'Channel {i+1}')
            axs[i].plot(x, slopes[i] * x + np.mean(rms_values[i]), linestyle='dashed', color='red', label='Linear Fit')
            axs[i].set_ylabel(f'Channel {i+1}')
            axs[i].grid(True)
            axs[i].legend()
        axs[-1].set_xlabel("Segment")
        plt.tight_layout()
        plt.show()
        return rms_values
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def compute_mnf():
    try:
        sampling_rate = 512
        segment_length = sampling_rate
        mnf_values = []
        slopes = []
        frequency_log = ""
        mnf_display_text = ""

        for i in range(data.shape[0]):
            mnf_per_segment = []
            for j in range(0, data.shape[1], segment_length):
                segment = data[i, j:j+segment_length]
                freqs, power = welch(segment, fs=sampling_rate, nperseg=segment_length)
                mnf = np.sum(freqs * power) / np.sum(power)
                mnf_per_segment.append(mnf)

                if mnf < 6 or mnf > 500:
                    frequency_log += f"Channel {i+1}, Segment {j//segment_length+1}: MNF = {mnf:.2f} Hz (out of range)\n"

            mnf_values.append(mnf_per_segment)

            x = np.arange(len(mnf_per_segment))
            slope, _, _, _, _ = linregress(x, mnf_per_segment)
            slopes.append(slope)

            mnf_display_text += f"Channel {i+1} MNF values:\n"
            mnf_display_text += ", ".join([f"{val:.2f}" for val in mnf_per_segment]) + "\n\n"

        mnf_slope_text.delete("1.0", tk.END)
        mnf_slope_text.insert(tk.END, "\n".join([f"Channel {i+1}: {slope:.6f}" for i, slope in enumerate(slopes)]))

        mnf_window = tk.Toplevel()
        mnf_window.title("MNF Values")
        mnf_textbox = tk.Text(mnf_window, width=80, height=25)
        mnf_textbox.pack()
        mnf_textbox.insert(tk.END, mnf_display_text)

        if frequency_log:
            messagebox.showwarning("Frequency Warning", f"Some MNF values are out of 6-500 Hz range:\n\n{frequency_log}")

        fig, axs = plt.subplots(len(mnf_values), 1, figsize=(10, 6))
        fig.suptitle("MNF Values with Linear Regression", fontsize=16)
        for i in range(len(mnf_values)):
            x = np.arange(len(mnf_values[i]))
            axs[i].plot(x, mnf_values[i], label=f'Channel {i+1}')
            axs[i].plot(x, slopes[i] * x + np.mean(mnf_values[i]), linestyle='dashed', color='red', label='Linear Fit')
            axs[i].set_ylabel(f'Channel {i+1}')
            axs[i].grid(True)
            axs[i].legend()
        axs[-1].set_xlabel("Segment")
        plt.tight_layout()
        plt.show()

        return mnf_values
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


# GUI setup
root = tk.Tk()
root.title("EMG")

data = None

open_button = tk.Button(root, text="Load .npy File", command=open_file)
open_button.pack(pady=10)

max_amp_button = tk.Button(root, text="Max Amplitudes", command=compute_max_amplitude)
max_amp_button.pack(pady=5)
max_amp_text = tk.Text(root, height=8, width=30)
max_amp_text.pack(pady=5)

rms_button = tk.Button(root, text="RMS and Slope", command=compute_rms)
rms_button.pack(pady=5)
rms_slope_text = tk.Text(root, height=8, width=30)
rms_slope_text.pack(pady=5)

mnf_button = tk.Button(root, text="MNF and Slope", command=compute_mnf)
mnf_button.pack(pady=5)
mnf_slope_text = tk.Text(root, height=8, width=30)
mnf_slope_text.pack(pady=5)

display_button = tk.Button(root, text="Display Signals", command=display_signals)
display_button.pack(pady=10)

root.mainloop()
