# -*- coding: utf-8 -*-
"""
ICT Charge Logger with accurate pulse width and ADC counts display
"""
import os
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
import epics
from matplotlib.widgets import SpanSelector
import csv
import datetime
import matplotlib.dates as mdates
from PIL import Image, ImageTk
import math
from tkinter import messagebox
import socket
import threading
from tkinter import filedialog

# ---------------- EPICS CONFIG ----------------
os.environ['EPICS_CA_ADDR_LIST'] = '192.168.0.112'
os.environ['EPICS_CA_AUTO_ADDR_LIST'] = 'NO'


REFRESH_INTERVAL = 1  # seconds

ADC_FULL_SCALE = 1.0
SAMPLE_RATE = 500e6
BUFFER_LEN = 32768
ICT_SENSITIVITY = 1.25
DEFAULT_SMOOTH_WINDOW = 15
USE_MIMIC_DEFAULT = False
charge_averaging = 1

# Tuning parameters
OFFSET_STATIC_THRESHOLD = 5        # minimal counts to consider correction
OFFSET_MIN_CHANGE = 2              # minimum PV-change (counts) to perform caput
BASELINE_WINDOW_FRAC = 0.10        # first 10% samples used to estimate baseline
MAD_FACTOR = 1.4826                # convert MAD -> sigma
MAD_K = 5.0                        # multiplier for sigma to form threshold
N_STABILITY_READS = 3              # repeated reads to require consensus
STABILITY_TOLERANCE = 2            # counts tolerance among repeated desired offsets
OFFSET_DEBOUNCE_SEC = 5.0          # don't flip offset more than once every N seconds
# ---------------- CONFIG ----------------
WAVEFORM_SAVE_LIMIT =1     # 1 for one file, 2 for two, etc.
WAVEFORM_PRECISION = ".3e"    # Scientific notation for filename (e.g., 5.123e-10)
WAVEFORM_SCALE_FACTOR = 1.0   # Set to 1000 to save in mV instead of V

# -------- Pulse timing gate --------
PULSE_SEARCH_START_NS = 200
PULSE_SEARCH_END_NS   = 6000.0
EXPECTED_PULSE_NS     = 350.0   # set close to your real pulse location
MAX_PEAK_SHIFT_SAMPLES = 5
OUTSIDE_SPIKE_RATIO = 3.0
BASELINE_END_NS     = 180
MIN_SNR = 6.0
MIN_VALID_FRAC = 0.3
# ---------------- HELPER FUNCTIONS ----------------
def read_adc():
    """Read ADC waveform from EPICS PV safely."""
    try:
        data = epics.caget(ADC_PV, timeout=2.0)
        if data is None:
            print(f"⚠️ Failed to read from PV {ADC_PV}")
            return None
        return np.array(data, dtype=np.float64)
    except Exception as e:
        print(f"⚠️ Error reading ADC PV: {e}")
        return None
    
def mimic_data(n=BUFFER_LEN):
    """Generate simulated ADC-like data."""
    t = np.linspace(0, 500e-9, n)
    pulse = np.exp(-((t - 250e-9)**2)/(2*(50e-9/2.355)**2))
    pulse += 0.02 * np.random.randn(n)
    return pulse

def adc_to_voltage(adc_counts):
    """Convert ADC counts to voltage."""
    # return (adc_counts / (2**(EFF_ADC_BITS - 1))) * (ADC_FULL_SCALE / 2)
    return (adc_counts* ADC_FULL_SCALE/ (2**14))*V_corr
def smooth_signal(signal, window):
    """Apply moving average smoothing."""
    if window < 2:
        return signal
    w = int(window)
    return np.convolve(signal, np.ones(w) / w, mode='same')


# updated FWHM bound detection 31/03/2026, above function is also working

def find_pulse_bounds_fwhm(signal, time_axis, smooth=True):
    signal = np.asarray(signal)
    time_axis = np.asarray(time_axis)

    if len(signal) != len(time_axis) or len(signal) < 3:
        return None, None, None

    work = signal.copy()

    # Optional smoothing
    if smooth:
        window = 5
        if len(work) > window:
            work = np.convolve(work, np.ones(window) / window, mode='same')

    # Estimate baseline from first 10% of waveform
    n_baseline = max(5, len(work) // 10)
    baseline = np.mean(work[:n_baseline])

    # Find peak
    # peak_idx = np.argmax(work)
    voltage_smooth = smooth_signal(signal, 15)
    peak_idx = np.argmax(voltage_smooth)
    peak_val = work[peak_idx]
    peak_height = peak_val - baseline

    if peak_height <= 0:
        return None, None, None

    # Half-maximum relative to baseline
    # half_max = baseline + 0.5 * peak_height
    half_max= 0.5*peak_val

    t_start = None
    t_end = None

    # Find left crossing
    for i in range(peak_idx - 1, -1, -1):
        if work[i] < half_max <= work[i + 1]:
            denom = work[i + 1] - work[i]
            if denom == 0:
                t_start = time_axis[i]
            else:
                frac = (half_max - work[i]) / denom
                t_start = time_axis[i] + frac * (time_axis[i + 1] - time_axis[i])
            break

    # Find right crossing
    for i in range(peak_idx, len(work) - 1):
        if work[i] >= half_max > work[i + 1]:
            denom = work[i + 1] - work[i]
            if denom == 0:
                t_end = time_axis[i]
            else:
                frac = (half_max - work[i]) / denom
                t_end = time_axis[i] + frac * (time_axis[i + 1] - time_axis[i])
            break

    if t_start is None or t_end is None:
        return None, None, None

    fwhm = t_end - t_start
    if fwhm <= 0 or np.isnan(fwhm):
        return None, None, None

    return t_start, t_end, fwhm

def find_pulse_bounds_shortpulse(signal, time_axis, baseline=0.0, smooth=True):
    """
    Pulse detection optimized for short pulses (<100 ns).

    Start  -> FWHM rising edge (50%)
    End    -> zero crossing OR 90% fall (10% of peak), whichever comes first

    Returns:
        t_start, t_end, width
    """

    # Optional smoothing
    if smooth:
        window = 5
        if len(signal) > window:
            signal = np.convolve(signal, np.ones(window)/window, mode='same')

    peak_idx = np.argmax(signal)
    peak_val = signal[peak_idx]

    if peak_val <= 0:
        return None, None, None

    half_max = 0.2 * peak_val
    fall_90 = 0.2 * peak_val  # 90% fall = 10% of peak

    # --- Start time: FWHM rising edge ---
    for i in range(peak_idx - 1, 0, -1):
        if signal[i] < half_max <= signal[i + 1]:
            frac = (half_max - signal[i]) / (signal[i + 1] - signal[i])
            t_start = time_axis[i] + frac * (time_axis[i + 1] - time_axis[i])
            break
    else:
        t_start = time_axis[0]

    # --- New fixed end time ---
 
    # if pulse_width_ns <0:
    #     FIXED_WIDTH = 70e-9  # 60 ns
        
    #     t_end = t_start + FIXED_WIDTH
    #     width = t_end - t_start
        
    # else:
        
    # # --- End time: search forward ---
    t_end_90 = None
    t_end_zero = None
    
    for i in range(peak_idx, len(signal) - 1):

        # 90% fall detection
        if t_end_90 is None and signal[i] >= fall_90 > signal[i + 1]:
            frac = (fall_90 - signal[i]) / (signal[i + 1] - signal[i])
            t_end_90 = time_axis[i] + frac * (time_axis[i + 1] - time_axis[i])

        # Zero crossing detection
        if t_end_zero is None and signal[i] >= baseline > signal[i + 1]:
            frac = (baseline - signal[i]) / (signal[i + 1] - signal[i])
            t_end_zero = time_axis[i] + frac * (time_axis[i + 1] - time_axis[i])

        if t_end_90 is not None and t_end_zero is not None:
            break

    # Choose earliest valid end
    candidates = [t for t in [t_end_zero, t_end_90] if t is not None]
    if not candidates:
        return None, None, None

    t_end = min(candidates)    
    
    
    
    if t_end > time_axis[-1]:
        return None, None, None
    
    width = t_end - t_start

    if width <= 0:
        return None, None, None

    return t_start, t_end, width

# Compute charge without compensating baseline integral

# def compute_auto_charge(time_s, voltage, sensitivity):
#     #t_start, t_end = find_pulse_bounds(voltage, time_s)
#     t_start, t_end, fwhm = find_pulse_bounds_fwhm(voltage, time_s)
#     if t_start is None or t_end is None or t_end <= t_start:
#         return None, None, None, None, None

#     # Indices corresponding to t_start and t_end
#     start_idx = np.searchsorted(time_s, t_start)
#     end_idx = np.searchsorted(time_s, t_end)

#     integral = np.trapz(voltage[start_idx:end_idx], time_s[start_idx:end_idx])
#     charge = integral / sensitivity
#     # peak_adc = np.max(v_raw)
#     pulse_samples = end_idx - start_idx
#     return charge, t_start, t_end, pulse_samples

# Compute charge with compensating baseline integral

def compute_auto_charge(time_s, voltage, sensitivity):
    
    # Choose pulse polarity
    peak_idx = np.argmax(voltage)
    peak_val = voltage[peak_idx]
     
    t_start, t_end, fwhm = find_pulse_bounds_fwhm(voltage, time_s)
    
    if t_start is None or t_end is None or t_end <= t_start:
        return None, None, None, None
    
    start_idx = np.searchsorted(time_s, t_start, side="left")
    end_idx = np.searchsorted(time_s, t_end, side="right")
    
    if end_idx <= start_idx:
        return None, None, None, None
    
    # --------- Pulse integration ---------
    pulse_time = time_s[start_idx:end_idx]
    pulse_volt = voltage[start_idx:end_idx]
    Vs_pulse = np.trapz(pulse_volt, pulse_time)
    
    if len(pulse_time) < 2:
        return None, None, None, None
    
    # Duration of pulse window
    duration = t_end - t_start
    # print(f"pulse width {duration}")
    # --------- Baseline window (same duration) ---------
    baseline_end   = t_start - 1e-9
    baseline_start = baseline_end - duration
    
    # If baseline window goes out of array, fallback to post-pulse window
    if baseline_start < time_s[0]:
        baseline_start = t_end + 1e-9
        baseline_end   = baseline_start + duration
    
    base_start_idx = np.searchsorted(time_s, baseline_start, side="left")
    base_end_idx   = np.searchsorted(time_s, baseline_end, side="right")
    
    # Check baseline region availability
    if base_end_idx > base_start_idx and base_end_idx <= len(time_s):
        baseline_time = time_s[base_start_idx:base_end_idx]
        baseline_volt = voltage[base_start_idx:base_end_idx]
        Vs_offset = np.trapz(baseline_volt, baseline_time)
    else:
        # If baseline cannot be computed, default to zero correction
        Vs_offset = 0.0
    
    # --------- Corrected integral ---------
    # Take intergration of offset and intergration of pulse separately
    # Vs_corrected = Vs_pulse - Vs_offset
    
    # First remove the offset from pulse and then take integral
    Vs_corrected = np.trapz(pulse_volt - np.mean(baseline_volt), pulse_time)
    
    # # --------- Charge ---------
    
    charge = Vs_corrected / sensitivity
    # print(duration)
    if 0 < duration <= 300e-9:
        
        #  Linear fitting
        # correction_factor= -1608236.475*duration + 1.386
    
        # new data collection 18/11/2025
        
        # polynomial of second order  corection_factor = 4E+12*duration**2 - 2E+06*duration + 1.3109
        # correction_factor = 7E+12*duration**2 - 3E+06*duration + 1.3788
        
        # 27/03/2026
        # Correction model log-normal fitting
        # correction_factor = 0.2694 * np.exp(-(np.log(duration) - np.log(4.55e-08))**2 / (2 * 0.4346**2)) + 1.0350
        
        # 03/042026
        correction_factor = 0.3571 * np.exp(-(np.log(duration) - np.log(5.71e-08))**2 / (2 * 0.4871**2)) + 1.0331
        # print(correction_factor)
        # print(correction_factor)
        charge= charge*correction_factor
        
        # pulse_samples = end_idx - start_idx
    
    return charge, t_start, t_end
    

def shift_signal_zero_pad(x, shift):
    y = np.zeros_like(x)

    if shift > 0:
        y[shift:] = x[:-shift]
    elif shift < 0:
        y[:shift] = x[-shift:]
    else:
        y[:] = x

    return y


def find_peak_idx_windowed(shot, expected_idx, search_half_width=20, positive_pulse=True):
    n = len(shot)

    start = max(0, expected_idx - search_half_width)
    end = min(n, expected_idx + search_half_width + 1)

    region = shot[start:end]

    if len(region) == 0:
        return expected_idx

    if positive_pulse:
        local_idx = np.argmax(region)
    else:
        local_idx = np.argmin(region)

    peak_idx = start + local_idx
    return peak_idx

def get_time_window_indices(time_axis, t_start_ns, t_end_ns):
    t0 = t_start_ns * 1e-9
    t1 = t_end_ns * 1e-9
    i0 = np.searchsorted(time_axis, t0, side="left")
    i1 = np.searchsorted(time_axis, t1, side="right")
    i0 = max(0, min(i0, len(time_axis)))
    i1 = max(0, min(i1, len(time_axis)))
    return i0, i1

def find_peak_idx_in_window(signal, time_axis, t_start_ns, t_end_ns, positive_pulse=True):
    i0, i1 = get_time_window_indices(time_axis, t_start_ns, t_end_ns)
    if i1 <= i0:
        return None

    region = signal[i0:i1]
    if len(region) == 0:
        return None

    local_idx = np.argmax(region) if positive_pulse else np.argmin(region)
    return i0 + local_idx

def gate_signal_to_window(signal, time_axis, t_start_ns, t_end_ns, fill_value=0.0):
    gated = np.full_like(signal, fill_value)
    i0, i1 = get_time_window_indices(time_axis, t_start_ns, t_end_ns)
    if i1 > i0:
        gated[i0:i1] = signal[i0:i1]
    return gated

def max_abs_outside_window(signal, time_axis, t_start_ns, t_end_ns):
    i0, i1 = get_time_window_indices(time_axis, t_start_ns, t_end_ns)

    outside_parts = []
    if i0 > 0:
        outside_parts.append(signal[:i0])
    if i1 < len(signal):
        outside_parts.append(signal[i1:])

    if not outside_parts:
        return 0.0

    outside = np.concatenate(outside_parts)
    if len(outside) == 0:
        return 0.0

    return float(np.max(np.abs(outside)))

def get_baseline_region_from_time(signal, time_axis, end_ns=180.0):
    end_idx = np.searchsorted(time_axis, end_ns * 1e-9, side="left")
    end_idx = max(end_idx, 20)  # ensure enough samples
    end_idx = min(end_idx, len(signal))
    region = signal[:end_idx]
    return region

# ---------------- GUI ----------------
class ICTChargeLoggerGUI:
    def __init__(self, master):
        self.master = master
        master.title("ICT Live Charge Logger")
        master.geometry("1500x900")


        self.background_recorded = False
        self.background_data = None
        self.background_time = None
        
        self.background_active = False
        self.background_data = None
        self.background_recorded = False


        # --- Config ---
        self.use_mimic = USE_MIMIC_DEFAULT
        self.sample_rate = SAMPLE_RATE
        self.buffer_len = BUFFER_LEN
        self.time_axis = np.arange(self.buffer_len) / self.sample_rate
        self.charges = []
        self.times = []
        
        # self.start_time = time.time()
        self.start_time_datetime = time.time()
        
        self.last_applied_offset = None
        self.last_offset_time = 0.0

        # --- File logging defaults ---
        self.csv_file = None
        self.csv_writer = None
        
        
        self.wf_saved_count = 0  # Track saved waveforms
        self.session_dir = "."   # save in the script's folder
        self.wf_saved_count = 0
        # Inside __init__(self, master):
        self.trig_pv_obj = epics.PV("libera:trigger:count")
        self.last_saved_trig = -1  # Initialize with a value that won't match the first count
        
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # --- Fonts ---
        self.font_btn = tkFont.Font(family="Arial", size=14, weight="bold")
        self.font_label = tkFont.Font(family="Arial", size=14, weight="bold")
        
        self.command_port = 5006  # 5005 is for Vacuum, 5006 is for Commands
        self.remote_active = False # For single shot measurements, trigger from single generator
        
        # Start the background listener thread
        self.socket_thread = threading.Thread(target=self.udp_command_listener, daemon=True)
        self.socket_thread.start()
              
        
        # --- Main Control Frame ---
        # --- Main Control Frame ---
        ctl_frame = tk.Frame(master, bg="#2E86C1")
        ctl_frame.pack(side=tk.TOP, fill=tk.X, pady=5)  # ctl_frame itself can be packed
        
        # Configure ctl_frame grid to handle two columns
        ctl_frame.columnconfigure(0, weight=1)
        ctl_frame.columnconfigure(1, weight=0)
        
        # --- Row 1, Column 0: Entry Fields ---
        row1_inputs = tk.Frame(ctl_frame, bg="#2E86C1")
        row1_inputs.grid(row=0, column=0, sticky="w", padx=10, pady=2)  # MUST use grid here
        
        tk.Label(row1_inputs, text="Source:", font=self.font_label, bg="#2E86C1", fg="white").pack(side=tk.LEFT)
        self.src_var = tk.StringVar(value="Mimic" if self.use_mimic else "Real")
        self.src_combo = ttk.Combobox(row1_inputs, textvariable=self.src_var,
                                      values=["Mimic", "Real"], width=6, font=self.font_label)
        self.src_combo.pack(side=tk.LEFT, padx=4)
        self.src_combo.bind("<<ComboboxSelected>>", self.on_source_change)
        
        tk.Label(row1_inputs, text="Channel:", font=self.font_label, bg="#2E86C1", fg="white").pack(side=tk.LEFT, padx=(15,0))
        self.ch_var = tk.StringVar(value="Ch1")
        self.ch_combo = ttk.Combobox(row1_inputs, textvariable=self.ch_var,
                                     values=["Ch1", "Ch2", "Ch3", "Ch4"], width=6, font=self.font_label)
        self.ch_combo.pack(side=tk.LEFT, padx=4)
        self.ch_combo.bind("<<ComboboxSelected>>", self.on_channel_change)
        
        tk.Label(row1_inputs, text="Arm Number:", font=self.font_label, bg="#2E86C1", fg="white").pack(side=tk.LEFT, padx=4)
        self.arm_number_var = tk.StringVar(value="0")
        self.entry_arm = tk.Entry(row1_inputs, textvariable=self.arm_number_var, width=6, justify='center', font=self.font_label)
        self.entry_arm.pack(side=tk.LEFT, padx=4)
        
        tk.Label(row1_inputs, text="Attenuation [0–31]:", font=self.font_label, bg="#2E86C1", fg="white").pack(side=tk.LEFT, padx=(15,0))
        self.att_var = tk.StringVar(value="31")
        self.att_entry = ttk.Entry(row1_inputs, textvariable=self.att_var, width=5, font=self.font_label)
        self.att_entry.pack(side=tk.LEFT, padx=4)
        
        tk.Label(row1_inputs, text="Avg Shots:", font=self.font_label, bg="#2E86C1", fg="white").pack(side=tk.LEFT, padx=(15,4))
        self.avg_shots_var = tk.StringVar(value="30")
        self.entry_avg_shots = tk.Entry(row1_inputs, textvariable=self.avg_shots_var, width=5, justify='center', font=self.font_label)
        self.entry_avg_shots.pack(side=tk.LEFT, padx=4)
        self.entry_avg_shots.bind("<FocusOut>", self.on_avg_shots_change)
        self.entry_avg_shots.bind("<Return>", self.on_avg_shots_change)
        
        # --- Row 2, Column 0: Buttons ---
        row2_buttons = tk.Frame(ctl_frame, bg="#2E86C1")
        row2_buttons.grid(row=1, column=0, sticky="w", padx=10, pady=5)
        
        self.btn_low_charge = tk.Button(row2_buttons, text="Low Charge: OFF", font=self.font_label,
                                        bg="#BFC9CA", command=self.toggle_low_charge_mode, width=15)
        self.btn_low_charge.pack(side=tk.LEFT, padx=5)
        
        self.btn_remote = tk.Button(row2_buttons, text="Single shot: OFF", font=self.font_label,
                                    bg="#BFC9CA", fg="black", command=self.toggle_remote_listener, width=15, relief="raised")
        self.btn_remote.pack(side=tk.LEFT, padx=5)
        
        self.btn_record_bg = tk.Button(row2_buttons, text="Record Background", font=self.font_label,
                                       bg="#BFC9CA", fg="black", width=18, relief="raised", bd=2, command=self.record_background)
        self.btn_record_bg.pack(side=tk.LEFT, padx=5)
        
        self.btn_start = tk.Button(row2_buttons, text="Start", font=self.font_btn,
                                   command=self.start_logging, bg="#BFC9CA", width=15)
        self.btn_start.pack(side=tk.LEFT, padx=10)
        
        self.btn_stop = tk.Button(row2_buttons, text="Stop", font=self.font_btn,
                                  command=self.stop_logging, bg="#BFC9CA", state=tk.DISABLED,width=15)
        self.btn_stop.pack(side=tk.LEFT, padx=4)
        
        # --- Column 1: Logo (Spanning Row 0 and Row 1) ---
        try:
            logo_image = Image.open(r"G:\My Drive\Libera ADC\logo.png")
            logo_image = logo_image.resize((300, 90), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_image)
        
            logo_label = tk.Label(ctl_frame, image=self.logo_photo, bg="#2E86C1")
            logo_label.grid(row=0, column=1, rowspan=2, padx=30, sticky="e")  # MUST use grid, not pack
        except Exception as e:
            print(f"Logo load failed: {e}")
            


        # --- Live Charge Label ---
        self.label_charge = tk.Label(master, text="Charge: -- C | Width: -- ns | Peak ADC: --", 
                                     font=("Arial", 16, "bold"), bg="#AED6F1", anchor="w")
        self.label_charge.pack(fill=tk.X, pady=4, padx=6)

        # --- Plot ---
        # --- Dual plots: Charge history (left) and waveform (right) ---
        self.fig, (self.ax_charge, self.ax_waveform) = plt.subplots(
            1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [5, 1]}
        )
        self.fig.subplots_adjust(wspace=0.35)
        
                # --- Optional: Background subtraction indicator text on waveform plot ---
        self.bg_text = self.ax_waveform.text(
            0.02, 0.9, '', transform=self.ax_waveform.transAxes,
            fontsize=12, color='green', verticalalignment='top'
        )

        
        # --- Connect mouse events for cursor dragging ---
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)



        # Left: Integrated charge vs time
        self.ax_charge.set_title("Integrated Charge vs Time", fontsize=22, fontweight='regular', pad=20)
        # self.ax_charge.set_xlabel("Time (s)")
        self.ax_charge.set_xlabel("Time [HH:MM:SS]", fontsize=22, fontweight='regular', labelpad=16)
        self.ax_charge.set_ylabel("Charge [C]", fontsize=22, fontweight='regular', labelpad=16)
        self.line_charge, = self.ax_charge.plot([], [], 'o-', color='blue')
        
        self.line_avg_charge, = self.ax_charge.plot([], [], 'o-', color='blue', lw=2)
        self.avg_charges = [] # List to store calculated averages for the plot
        
                # --- Add interactive vertical cursor to charge plot ---
        self.cursor_line = self.ax_charge.axvline(
            x=datetime.datetime.now(), color='green', linestyle='--', lw=2.0, alpha=0.8
        )
        self.cursor_text = self.ax_charge.text(
            0.02, 0.95, '', transform=self.ax_charge.transAxes,
            fontsize=14, color='red', verticalalignment='top'
        )
        self.dragging_cursor = False
        self.cursor_x = mdates.date2num(datetime.datetime.now())  # store cursor position persistently


        # Change tick label font size and family for both axes
        self.ax_charge.tick_params(axis='both', labelsize=20)  # font size for tick labels
        for label in (self.ax_charge.get_xticklabels() + self.ax_charge.get_yticklabels()):
            label.set_fontname('Arial')       # or 'DejaVu Sans', 'Calibri', etc.
            label.set_fontweight('regular')   # 'bold' if you prefer
        
        # >>> Add this block to increase the scientific notation offset font size <<<
        self.ax_charge.yaxis.get_offset_text().set_fontsize(22)     # increase offset text font size
        self.ax_charge.yaxis.get_offset_text().set_fontweight('regular')
        self.ax_charge.yaxis.get_offset_text().set_fontname('Arial')        

        self.ax_charge.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        self.ax_charge.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.fig.autofmt_xdate()
        
        # Right: Live waveform plot
        self.ax_waveform.set_title("Live Pulse", fontsize=22, fontweight='regular', pad=20)
        self.ax_waveform.set_xlabel("Time [ns]", fontsize=22, fontweight='regular', labelpad=12)
        self.ax_waveform.set_ylabel("Voltage [V]", fontsize=22, fontweight='regular', labelpad=12)
        self.line_waveform, = self.ax_waveform.plot([], [], '-', color='green', lw=1.2)
        self.pulse_span = None  # shaded FWHM area
        
        
        # Adjust tick label font (scale numbers)
        self.ax_waveform.tick_params(axis='both', labelsize=20)  # change tick label size
        
        for label in (self.ax_waveform.get_xticklabels() + self.ax_waveform.get_yticklabels()):
            label.set_fontname('Arial')       # or 'DejaVu Sans', 'Calibri', etc.
            label.set_fontweight('regular')   # or 'bold' if preferred

        
        # Embed figure in GUI
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


        self.streaming = False
        self.update_interval_ms = 5
        
        # TEMP low-charge mode flag (default False)
        self.low_charge_mode = False
        self.low_charge_avg_shots = int(self.avg_shots_var.get()) # number of shots to average
        self.low_charge_buffer = []    # buffer for multi-shot averaging
        
        # --- Logo (optional) ---
        # try:
        #     logo_image = Image.open(r"G:\My Drive\Libera ADC\logo.png")
        #     logo_image = logo_image.resize((150, 50), Image.LANCZOS)
        #     self.logo_photo = ImageTk.PhotoImage(logo_image)
        #     logo_label = tk.Label(ctl_frame, image=self.logo_photo, bg="#2E86C1")
        #     logo_label.pack(side=tk.RIGHT, padx=10)
        # except Exception as e:
        #     print("Logo load failed:", e)

    def on_source_change(self, event=None):
        self.use_mimic = (self.src_var.get() == "Mimic")
        
        
    def on_channel_change(self, event=None):
        """Update ADC PVs based on selected channel."""
        selected = self.ch_var.get()
        print(f"ℹ️ Channel changed to {selected}")
    
        # Update the PVs to the chosen channel dynamically
        
        global channel_number
        
        channel_number = int(selected[-1])  # extracts 1–4
        # ADC_PV = f"libera:signals:adc.Ch{channel_number}"
        
        # # --- ADC Offset PVs ---
        # ADC_OFFSET_SP_PV = f"libera:dsp:adc_offset:ch{channel_number}_sp" # write PV
        # ADC_OFFSET_MON_PV = f"libera:dsp:adc_offset:ch{channel_number}_mon" # read PV

        # print(f"→ ADC PV set to {ADC_PV}")
    
    def on_attenuation_change(self, event=None):
        """Automatically apply attenuation and update EFF_ADC_BITS when value changes."""
        try:
            att = float(self.att_var.get())
            if not (0 <= att <= 31):
                tk.messagebox.showwarning("Invalid Input", "Attenuation must be between 0 and 31.")
                return
        except ValueError:
            tk.messagebox.showwarning("Invalid Input", "Please enter a numeric attenuation value.")
            return
    
        # Determine channel number from combobox
        ch = int(self.ch_var.get()[-1])
    
        # --- Apply attenuation to Libera ---
        try:
            att_pv = f"libera:att:ch{ch}_sp"
            print(f"✉️ Setting attenuation {att:.1f} dB on {att_pv}")
            epics.caput(att_pv, att, wait=True, timeout=1.0)
        except Exception as e:
            print(f"⚠️ Failed to set attenuation: {e}")
            return
    
        # --- Reset offset of that channel ---
        try:
            offset_pv = f"libera:dsp:adc_offset:ch{ch}_sp"
            print(f"↪ Resetting {offset_pv} to 0")
            epics.caput(offset_pv, 0, wait=True, timeout=1.0)
        except Exception as e:
            print(f"⚠️ Failed to reset offset: {e}")
    
        # --- Update effective ADC bits ---
        global V_corr
        try:
            # Replace with your real formula here ↓
            # Example placeholder: EFF_ADC_BITS = 11.5 + 0.25 * (att - 10)
            
            # EFF_ADC_BITS = -0.166*att + 18.119
            V_corr = 0.0557 * math.exp(0.1151 * att)
          
            print(f"✅ Updated voltage correction factor = {V_corr:.3f}")
        except Exception as e:
            print(f"⚠️ Could not compute EFF_ADC_BITS: {e}")
    
        # # Optional info message
        # tk.messagebox.showinfo("Attenuation Applied",
        #                        f"Channel {ch} attenuation set to {att:.1f} dB\n"
        #                        f"Offset reset to 0\n"
        #                        f"Effective bits = {EFF_ADC_BITS:.3f}")




    def start_logging(self,remote=False):
       
        try:
            att_val = float(self.att_var.get())
            if not (0 <= att_val <= 31):
                messagebox.showerror("Invalid Attenuation", 
                                     f"Attenuation {att_val} is out of bounds (0–31).\n"
                                     "Please fix this before starting.")
                return # Exit the function; does not set self.streaming to True
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a numeric value for attenuation.")
            return
        
        self.wf_saved_count = 0  # Reset counter for the new run
        self.streaming = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        global selected_dir
        if remote:
            # AUTO-ENABLE logging for remote shots
            answer = True
            print("🚀 Remote Trigger: Logging auto-enabled.")
        
            # Default to script directory
            selected_dir = os.path.dirname(os.path.abspath(__file__))
        
        else:
            # Ask user if logging should be enabled
            answer = messagebox.askyesno("Enable Logging", "Do you want to enable data logging?")
        
            if answer:
                # Ask user to select directory only for non-remote logging
                selected_dir = filedialog.askdirectory(title="Select Log Directory")
        
                if not selected_dir:
                    print("No directory selected. Logging canceled.")
                    return
        
        if answer:
            os.makedirs(selected_dir, exist_ok=True)
        
            # Create filename inside selected directory
            filename = os.path.join(
                selected_dir,
                datetime.datetime.now().strftime("charge_log_%Y%m%d_%H%M%S.csv")
            )
        
            self.csv_file = open(filename, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
        
            self.csv_writer.writerow([
                "Time",
                "Charge(C)",
                "ADC_Max_Counts",
                "PulseWidth(s)",
                "AverageCharge(C)",
                "SNR"
            ])

            print(f"ℹ️ Logging started: {filename}")
            # Open CSV file
            # filename = datetime.datetime.now().strftime("charge_log_%Y%m%d_%H%M%S.csv")
            self.csv_file = open(filename, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["Time", "Charge(C)", "ADC_Max_Counts","PulseWidth(s)", "AverageCharge(C)", "SNR"])
            print(f"ℹ️ Logging started: {filename}")
        else:
            self.csv_file = None
            self.csv_writer = None
            print("ℹ️ Logging disabled by user. Acquisition will continue without saving.")
        
        # --- Apply Arm Number PV before acquisition ---
        try:
            arm_number_str = self.arm_number_var.get().strip()
            arm_number = int(arm_number_str)
            epics.caput("libera:dsp:arm_number_sp", arm_number, wait=True, timeout=1.0)
            print(f"✅ Arm number set to {arm_number}")
        except Exception as e:
            print(f"⚠️ Failed to set arm number: {e}")
     
        self.on_channel_change()
        
        global ADC_PV, ADC_OFFSET_SP_PV, ADC_OFFSET_MON_PV
        
        ADC_PV = f"libera:signals:adc.Ch{channel_number}"
        
        # --- ADC Offset PVs ---
        ADC_OFFSET_SP_PV = f"libera:dsp:adc_offset:ch{channel_number}_sp" # write PV
        ADC_OFFSET_MON_PV = f"libera:dsp:adc_offset:ch{channel_number}_mon" # read PV

        print(f"→ ADC PV set to {ADC_PV}")
        
        
        # --- Calculate effective number of bits---
        self.on_attenuation_change()
       
        # --- Apply ADC offset once before streaming ---
        self.apply_adc_offset_smart()

        # --- Then begin live acquisition ---
        self._schedule_update()

    def stop_logging(self):
        self.streaming = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        # Close CSV file
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

    def _schedule_update(self):
        if self.streaming:
            self.master.after(self.update_interval_ms, self._update_step)

    def on_avg_shots_change(self, event=None):
        """Update the number of shots to average from the GUI input."""
        try:
            val = int(self.avg_shots_var.get())
            if val < 1:
                raise ValueError
            self.low_charge_avg_shots = val
            # Optional: Clear buffer when the averaging window changes to avoid mixing
            self.low_charge_buffer.clear()
            print(f"ℹ️ Averaging shots updated to: {val}")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Average shots must be a positive integer.")
            self.avg_shots_var.set(str(self.low_charge_avg_shots))
            
    

    def _update_step(self):
        """Main periodic acquisition and plotting loop."""
        if not self.streaming:
            return
        
        # 1. Fetch current hardware trigger count
        hw_trig = self.trig_pv_obj.get()
        hw_trig_val = int(hw_trig) if hw_trig is not None else 0

        # 2. Check if this is a NEW trigger
        # If the hardware hasn't incremented, skip the save/log logic
        is_new_event = (hw_trig_val != self.last_saved_trig)
        
        
        # --- Acquire waveform ---
        if self.use_mimic:
            v_raw = mimic_data(self.buffer_len)
        else:
            data = read_adc()
            v_raw = data if data is not None else mimic_data(self.buffer_len)
            # --- Dynamically adjust time axis based on actual data length ---
        n_samples = len(v_raw)
        self.time_axis = np.arange(n_samples) / self.sample_rate
    
        # --- Convert and smooth ---
        v_volts = adc_to_voltage(v_raw)
        v_smoothed = smooth_signal(v_volts, DEFAULT_SMOOTH_WINDOW)
        v_work = v_smoothed.copy()
        
        
        # --- Apply background subtraction only if enabled ---
        if self.background_active and self.background_data is not None:
            if len(self.background_data) == len(v_work):
                offset = np.mean(v_work) - np.mean(self.background_data)
                v_work = v_work - (self.background_data + offset)
            else:
                print("⚠️ Background length mismatch — skipping subtraction")
        
        
        if self.low_charge_mode:

            self.low_charge_buffer.append(v_work.copy())
        
            if len(self.low_charge_buffer) < self.low_charge_avg_shots:
                self._schedule_update()
                return
        
            valid_shots = []
            zeroed_shots = []
            rejected_count = 0
        
            for shot in self.low_charge_buffer:
        
                shot = np.asarray(shot, dtype=float).ravel()
        
                # --- Smooth only for detection ---
                shot_detect = smooth_signal(shot, 15)
        
                # --- Baseline/noise from region BEFORE 200 ns ---
                baseline_region = get_baseline_region_from_time(
                    shot_detect,
                    self.time_axis,
                    end_ns=BASELINE_END_NS
                )
        
                if len(baseline_region) < 10:
                    rejected_count += 1
                    continue
        
                baseline = float(np.mean(baseline_region))
                noise = float(np.std(baseline_region))
        
                # Always subtract baseline so raw averaging is unbiased
                shot_zero = shot - baseline
                zeroed_shots.append(shot_zero)
        
                if noise <= 0:
                    rejected_count += 1
                    continue
        
                # --- Find peak only in allowed pulse window ---
                peak_idx = find_peak_idx_in_window(
                    shot_detect,
                    self.time_axis,
                    PULSE_SEARCH_START_NS,
                    PULSE_SEARCH_END_NS,
                    positive_pulse=True
                )
        
                if peak_idx is None:
                    rejected_count += 1
                    continue
        
                peak_height = float(shot_detect[peak_idx] - baseline)
                snr = peak_height / noise
                
                # --- Keep only real-signal-like shots ---
                if snr >= MIN_SNR:
                    shot_gated = gate_signal_to_window(
                        shot_zero,
                        self.time_axis,
                        PULSE_SEARCH_START_NS,
                        PULSE_SEARCH_END_NS,
                        fill_value=0.0
                    )
                    valid_shots.append(shot_gated)
                else:
                    rejected_count += 1
        
            # --- Decide if real signal exists ---
            MIN_VALID = max(2, int(MIN_VALID_FRAC * self.low_charge_avg_shots))
        
            if len(valid_shots) >= MIN_VALID:
                v_corrected = np.mean(np.stack(valid_shots, axis=0), axis=0)
                print(f"Low-charge: SIGNAL detected -> used {len(valid_shots)}, rejected {rejected_count}")
            else:
                # No signal: average all baseline-subtracted shots, no gating, no bias
                v_corrected = np.mean(np.stack(zeroed_shots, axis=0), axis=0)
                print(f"Low-charge: NO signal -> baseline-subtracted raw averaging ({len(zeroed_shots)} shots)")
        
            self.low_charge_buffer.clear()
        
        else:
            v_corrected = v_work
            
        # Store latest waveform for possible background recording
        self.current_waveform_y = v_smoothed
        self.current_waveform_x = self.time_axis
        
        # ---------- SIGNAL SNR (waveform-based) ----------

        # Smooth for stable peak detection (optional but recommended)
        v_detect = v_corrected
        
        # --- Baseline region (BEFORE pulse) ---
        baseline_region = v_detect[:np.searchsorted(self.time_axis, 180e-9)]
        baseline = np.mean(baseline_region)
        noise = np.std(baseline_region)
        
        # --- Signal peak (AFTER 200 ns) ---
        peak_idx = find_peak_idx_in_window(
            v_detect,
            self.time_axis,
            PULSE_SEARCH_START_NS,
            PULSE_SEARCH_END_NS,
            positive_pulse=True
        )
        
        if peak_idx is not None and noise > 0:
            peak_val = v_detect[peak_idx]
            signal_amp = peak_val - baseline
            snr_signal = signal_amp / noise
        else:
            snr_signal = 0.0
        
        peak_adc=np.max(v_raw)
        # --- Compute charge and pulse parameters ---
        v_for_charge = gate_signal_to_window(
            v_corrected,
            self.time_axis,
            PULSE_SEARCH_START_NS,
            PULSE_SEARCH_END_NS,
            fill_value=0.0
        )
        
        charge, t_start, t_end = compute_auto_charge(
            self.time_axis, v_for_charge, ICT_SENSITIVITY
        )
        
        
        
        
        # --- Update charge display and log ---
        if charge is not None:
            current_time = datetime.datetime.now()
            self.times.append(current_time)
            self.charges.append(charge)
    
            if t_start is not None and t_end is not None:
                pulse_width_ns = (t_end - t_start) * 1e9
            else:
                pulse_width_ns = 0
            avg_charge=np.mean(self.charges[-charge_averaging:])
            self.avg_charges.append(avg_charge)
            self.label_charge.config(
                text=f"Charge: {avg_charge:.3e} C   "
                     f"ADC counts: {peak_adc:.0f}   "
                     f"Width: {pulse_width_ns:.2f} ns"
            )
    

    #  un comment if want fixed squeezing time window
        # # --- Update left plot (Charge vs Time) ---
        # self.line_charge.set_data(self.times, self.charges)
        # self.ax_charge.relim()
        # self.ax_charge.autoscale_view()
        
       
        # --- Update left plot (Charge vs Time) ---
        # self.line_charge.set_data(self.times, self.charges)
        # To plot average charge
        self.line_avg_charge.set_data(self.times, self.avg_charges)
        
        # --- Keep a rolling 10-minute window on x-axis ---
        if len(self.times) > 0:
            current_time = self.times[-1]
            start_time = self.times[0]
            #  set minutes value to adujust window
            window = datetime.timedelta(minutes=1) 
        
            # Define the visible window
            if (current_time - start_time) < window:
                x_min, x_max = start_time, start_time + window
            else:
                x_min, x_max = current_time - window, current_time
        
            self.ax_charge.set_xlim(x_min, x_max)
        
            # --- Fix timezone mismatch: make sure all datetimes are naive ---
            cursor_time = mdates.num2date(self.cursor_x).replace(tzinfo=None)
            x_min = x_min.replace(tzinfo=None)
            x_max = x_max.replace(tzinfo=None)
        
            # --- Keep cursor visible (move it to edge if out of bounds) ---
            if not (x_min <= cursor_time <= x_max):
                # Move cursor to right edge (current time)
                self.cursor_x = mdates.date2num(current_time)
                self.cursor_line.set_xdata([self.cursor_x, self.cursor_x])
        
            # Recalculate Y limits normally
            self.ax_charge.relim()
            self.ax_charge.autoscale_view(scaley=True)
            
            # --- Only update cursor text if user is NOT dragging it ---
            if not getattr(self, "dragging_cursor", False):
                cursor_datetime = mdates.num2date(self.cursor_x).replace(tzinfo=None)
            
                # Find nearest charge point to cursor (optional: improve UX)
                if hasattr(self, "times") and hasattr(self, "charges") and len(self.times) > 0:
                    # Find the closest time index
                    time_diffs = [abs((t - cursor_datetime).total_seconds()) for t in self.times]
                    idx = time_diffs.index(min(time_diffs))
                    nearest_charge = self.charges[idx]
                    self.cursor_text.set_text(f"{cursor_datetime.strftime('%H:%M:%S')}\n{nearest_charge:.3e} C")
                else:
                    # fallback if no data yet
                    self.cursor_text.set_text(cursor_datetime.strftime("%H:%M:%S"))
                        
            
        # self.line_waveform.set_data(self.time_axis * 1e9, v_corrected)
        self.line_waveform.set_data(self.time_axis * 1e9, v_for_charge)

    
        # Remove old span if it exists
        if self.pulse_span:
            self.pulse_span.remove()
            self.pulse_span = None
    
        # Highlight detected pulse region
        if t_start is not None and t_end is not None:
            self.pulse_span = self.ax_waveform.axvspan(
                t_start * 1e9, t_end * 1e9,
                color='orange', alpha=0.3, label='Detected Pulse'
            )
    
            # Optional: zoom waveform around pulse (±50 ns window)
            window_ns = 100
            self.ax_waveform.set_xlim(
                (t_start - window_ns * 1e-9) * 1e9,
                (t_end + window_ns * 1e-9) * 1e9
            )
    
        else:
            # If no pulse detected, show full waveform range
            self.ax_waveform.set_xlim(self.time_axis[0] * 1e9,
                                      self.time_axis[-1] * 1e9)
    
        self.ax_waveform.relim()
        self.ax_waveform.autoscale_view(scaley=True)
    
        # --- Redraw both plots ---
        self.canvas.draw_idle()
    
        # Pulse width in seconds
        # Safe pulse width / ADC range handling
        if t_start is not None and t_end is not None and t_end > t_start:
            pulse_width = t_end - t_start
        
            start_idx = max(0, int(t_start * self.sample_rate))
            end_idx = min(len(v_raw), int(t_end * self.sample_rate))
        
            if end_idx > start_idx:
                adc_counts_max = np.max(v_raw[start_idx:end_idx])
            else:
                adc_counts_max = np.max(v_raw)
        else:
            pulse_width = 0.0
            adc_counts_max = np.max(v_raw) if len(v_raw) > 0 else 0.0
        
        # Average of last N values
        avg_charge = np.mean(self.charges[-charge_averaging:]) if len(self.charges) > 0 else 0.0
    
        # --- Write CSV ---
        # elapsed_seconds = time.time() - self.start_time_datetime
        elapsed_seconds=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        

        if self.csv_writer:
            charge_to_save = charge if charge is not None else 0.0
            self.csv_writer.writerow([elapsed_seconds, charge_to_save, adc_counts_max, pulse_width, avg_charge, snr_signal])
            if is_new_event and charge is not None:
                # Update the tracking variable so we don't save this one again
                self.last_saved_trig = hw_trig_val

                # --- Save .npy Waveform ---
                if self.wf_saved_count < WAVEFORM_SAVE_LIMIT:
                    v_scaled = v_corrected * WAVEFORM_SCALE_FACTOR
                    charge_str = format(charge, WAVEFORM_PRECISION)
                    ts = datetime.datetime.now().strftime("%H%M%S_%f")
                    
                    filename = os.path.join(
                        selected_dir,
                        datetime.datetime.now().strftime(f"WF_Trig_{hw_trig_val:06d}_{ts}_Q_{charge_str}C.npy")
                    )
                    
                    # filename = f"WF_Trig_{hw_trig_val:06d}_{ts}_Q_{charge_str}C.npy"
                    np.save(filename, v_scaled)
                    self.wf_saved_count += 1
        # --- Schedule next update ---
        # --- Keep cursor at its last known position ---
        
        if self.cursor_x is not None:
            self.cursor_line.set_xdata([self.cursor_x, self.cursor_x])
        
            # Reposition text on top layer if it somehow disappears
            if self.cursor_text not in self.ax_charge.texts:
                self.ax_charge.add_artist(self.cursor_text)


        self._schedule_update()

    def apply_adc_offset_smart(self):
        """
        Improved smart ADC offset compensation.
        Takes multiple baseline samples, computes robust median baseline,
        ignores outliers, and applies offset only if significantly nonzero.
        """
        if self.use_mimic:
            print("ℹ️ Mimic mode active: offset compensation skipped.")
            return
    
        try:
            print("⚙️ Starting smart offset compensation check...")
    
            N_SAMPLES = 3                 # how many baseline acquisitions to average
            BASELINE_WINDOW_FRAC = 0.05   # fraction of waveform used for baseline window
            OFFSET_STATIC_THRESHOLD = 5   # minimum baseline in counts before correction
            OFFSET_MIN_CHANGE = 2         # minimum change before writing new offset
    
            medians, sigmas, desired_offsets = [], [], []
    
            # Collect multiple baseline readings
            for i in range(N_SAMPLES):
                data = read_adc()
                if data is None:
                    print(f"⚠️ Read {i+1}/{N_SAMPLES} failed, skipping...")
                    continue
    
                n_win = max(1, int(BASELINE_WINDOW_FRAC * len(data)))
                baseline_window = data[:n_win]
    
                baseline_median = float(np.median(baseline_window))
                noise_std = float(np.std(baseline_window))
    
                medians.append(baseline_median)
                sigmas.append(noise_std)
                desired_offsets.append(-round(baseline_median))
                time.sleep(0.05)
    
            if len(medians) == 0:
                print("⚠️ No valid baseline readings found.")
                return
    
            # Convert lists to arrays for math
            medians = np.array(medians, dtype=float)
            sigmas = np.array(sigmas, dtype=float)
            desired_offsets = np.array(desired_offsets, dtype=float)
    
            print(f"   medians={medians.tolist()}, sigmas={sigmas.tolist()}, desired_offsets={desired_offsets.tolist()}")
    
            # Robust median-of-medians filtering (ignore outliers)
            median_of_medians = np.median(medians)
            mad_of_medians = np.median(np.abs(medians - median_of_medians))
            mask = np.abs(medians - median_of_medians) < 3 * mad_of_medians if mad_of_medians > 0 else np.ones_like(medians, dtype=bool)
    
            medians_filtered = medians[mask]
            median_final = float(np.median(medians_filtered))
            noise_final = float(np.median(sigmas[mask]))
    
            print(f"   filtered_medians={medians_filtered.tolist()}")
            print(f"   median_final={median_final:.2f} counts, noise_final={noise_final:.2f} counts")
    
            # --- Improved adaptive threshold decision ---
            # Lower limit (noise immunity) and upper limit (prevent blocking due to large noise)
            threshold_min = OFFSET_STATIC_THRESHOLD         # e.g. 5 counts
            threshold_max = 30                              # don’t let noise make threshold >50
            adaptive_threshold = min(max(3.0 * noise_final, threshold_min), threshold_max)
            
            print(f"   → adaptive_threshold = ±{adaptive_threshold:.2f} counts")
            
            if abs(median_final) < adaptive_threshold:
                print(f"   ℹ️ Baseline ({median_final:.2f}) within ±{adaptive_threshold:.2f} counts → no offset applied.")
                return

            desired_offset = int(round(-median_final))
    
            # Read current PV (if available)
            try:
                current_pv = epics.caget(ADC_OFFSET_MON_PV, timeout=1.0)
                current_pv_val = int(round(float(current_pv))) if current_pv is not None else None
            except Exception:
                current_pv_val = None
    
            # Check if we really need to write
            if current_pv_val is not None:
                delta = abs(desired_offset - current_pv_val)
                if delta < OFFSET_MIN_CHANGE:
                    print(f"   ℹ️ Current PV {current_pv_val} within {OFFSET_MIN_CHANGE} counts → no update needed.")
                    return
    
            # Apply offset
            print(f"   ✉️ Applying ADC offset {desired_offset} counts")
            epics.caput(ADC_OFFSET_SP_PV, desired_offset, wait=True, timeout=1.0)
            time.sleep(0.1)
    
            # Verify
            try:
                applied = epics.caget(ADC_OFFSET_MON_PV, timeout=1.0)
                print(f"✅ Offset applied successfully: {desired_offset} (PV now {applied})")
            except Exception as e:
                print(f"✅ Offset {desired_offset} applied, but PV readback failed: {e}")
    
        except Exception as e:
            print(f"⚠️ Offset compensation failed: {e}")

    def on_click(self, event):
        """Start dragging if user clicks near the vertical cursor line."""
        if event.inaxes != self.ax_charge or event.xdata is None:
            return
    
        cursor_x = self.cursor_x  # current cursor float-date
        xlim = self.ax_charge.get_xlim()
        tol = (xlim[1] - xlim[0]) * 0.01  # 1% of axis width tolerance
    
        if abs(event.xdata - cursor_x) < tol:
            self.dragging_cursor = True
    
    
    def on_motion(self, event):
        """Move cursor while dragging."""
        if not self.dragging_cursor or event.inaxes != self.ax_charge or event.xdata is None:
            return
    
        # update position
        self.cursor_x = event.xdata
        self.cursor_line.set_xdata([self.cursor_x, self.cursor_x])
    
        # find nearest data point
        x_data = np.array(self.line_charge.get_xdata())
        y_data = np.array(self.line_charge.get_ydata())
        if len(x_data) > 0:
            idx = np.abs(mdates.date2num(x_data) - self.cursor_x).argmin()
            nearest_time = x_data[idx]
            nearest_charge = y_data[idx]
            time_str = mdates.num2date(self.cursor_x).strftime('%H:%M:%S')
            self.cursor_text.set_text(f"Time: {time_str}\nCharge: {nearest_charge:.3e} C")
    
        self.fig.canvas.draw_idle()
    
    
    def on_release(self, event):
        """Stop dragging but keep cursor in final position."""
        if self.dragging_cursor:
            self.dragging_cursor = False
            # Leave cursor where released
            if event.xdata is not None:
                self.cursor_x = event.xdata

    def record_background(self):
        """Toggle background subtraction ON/OFF during acquisition."""
        try:
            # If already active, turn it OFF
            if getattr(self, "background_active", False):
                self.background_active = False
                self.background_recorded = False
                self.background_data = None
    
                # Button visual reset
                self.btn_record_bg.config(
                    text="Record Background",
                    bg="#BFC9CA", fg="black", relief="raised"
                )
                self.bg_text.set_text("")  # clear on-plot label
                print("🟡 Background subtraction turned OFF.")
                return
    
            # --- Otherwise: record new background ---
            if hasattr(self, "current_waveform_y") and len(self.current_waveform_y) > 0:
                self.background_data = np.array(self.current_waveform_y)
                self.background_recorded = True
                self.background_active = True
                self.background_time = datetime.datetime.now()
    
                # Visual feedback
                self.btn_record_bg.config(
                    text="Background Active",
                    bg="#27AE60", fg="white", relief="sunken"
                )
                print(f"✅ Background recorded at {self.background_time.strftime('%H:%M:%S')}")
            else:
                print("⚠️ No waveform data available to record background.")
        except Exception as e:
            print(f"❌ Error in record_background: {e}")
            
    # def udp_command_listener(self):
    #     """Listens for UDP packets on Port 5006."""
    #     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #     sock.bind(("127.0.0.1", self.command_port))
    #     while True:
    #         data, addr = sock.recvfrom(1024)
    #         msg = data.decode('utf-8').strip()
            
    #         # Logic: Only trigger if we are NOT currently recording
    #         if msg.startswith("START") and not self.streaming:
    #             duration = int(msg.split(":")[1]) if ":" in msg else 9
    #             # Use .after to safely trigger GUI functions from a background thread
    #             self.master.after(0, self.execute_remote_single_shot, duration)
    
    def udp_command_listener(self):
        """Listens for UDP packets with safety checks for the Main Loop."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            sock.bind(("127.0.0.1", self.command_port))
            sock.settimeout(1.0) # Allows the thread to check 'remote_active' status
        except Exception as e:
            print(f"❌ UDP Bind Error: {e}")
            return
    
        while getattr(self, "remote_active", True):
            try:
                data, addr = sock.recvfrom(1024)
                msg = data.decode('utf-8').strip()
                
                # Verify the GUI hasn't been closed/restarted since the thread started
                if msg.startswith("START"):
                    try:
                        if self.master.winfo_exists() and not self.streaming:
                            duration = int(msg.split(":")[1]) if ":" in msg else 9
                            # Inject into the REAL main thread
                            self.master.after(0, self.execute_remote_single_shot, duration)
                    except (tk.TclError, AttributeError):
                        # Master is dead, exit thread
                        break
                        
            except socket.timeout:
                continue 
            except Exception as e:
                if getattr(self, "remote_active", True):
                    print(f"⚠️ UDP Listener Warning: {e}")
                break
                
        sock.close()
        # print("📡 UDP Listener Thread Exited.")
        
    def execute_remote_single_shot(self, duration_sec):
        """Starts acquisition silently. AVOID MESSAGEBOXES HERE."""
        if self.streaming:
            return
    
        # Use the 'remote' flag to bypass the 'Ask to enable logging' popup
        self.src_var.set("Real")
        self.on_source_change()
        self.start_logging(remote=True) # Ensure this skips the yes/no popup!
        
        print(f"🚀 Remote Trigger: Started for {duration_sec}s")
        self.master.after(duration_sec * 1000, self.stop_logging)

    def on_closing(self):
        """Cleanly shut down threads and sockets before closing the window."""
        print("Shutting down Logger GUI...")
        self.remote_active = False  # Signal the UDP loop to stop
        self.streaming = False
        
        # Close CSV if open
        if self.csv_file:
            try:
                self.csv_file.close()
            except:
                pass
                
        # The daemon thread will still take a second to exit due to the 1.0s timeout
        self.master.destroy()

    def toggle_low_charge_mode(self):
            """Toggles the low charge averaging mode and updates button appearance."""
            self.low_charge_mode = not self.low_charge_mode
            
            if self.low_charge_mode:
                self.low_charge_buffer.clear() # Reset buffer when starting
                self.btn_low_charge.config(
                    text="Low Charge: ON", 
                    bg="#F39C12",  # Orange/Warning color
                    fg="white",
                    relief="sunken"
                )
                print("Mode: Low Charge Averaging Enabled")
            else:
                self.btn_low_charge.config(
                    text="Low Charge: OFF", 
                    bg="#BFC9CA", 
                    fg="black",
                    relief="raised"
                )
                print("Mode: Low Charge Averaging Disabled")

    def toggle_remote_listener(self):
            """Toggles the UDP socket listener."""
            self.remote_active = not self.remote_active
            if self.remote_active:
                # If thread isn't running or was never started, start it now
                if not hasattr(self, 'socket_thread') or not self.socket_thread.is_alive():
                    self.socket_thread = threading.Thread(target=self.udp_command_listener, daemon=True)
                    self.socket_thread.start()
                
                self.btn_remote.config(text="Singel shot: ON", bg="#27AE60", fg="white", relief="sunken")
                # print("📡 UDP Listener: STARTED")
            else:
                self.btn_remote.config(text="Single shot: OFF", bg="#BFC9CA", fg="black", relief="raised")
                # print("📡 UDP Listener: STOPPING...")



    
# ---------------- MAIN ----------------
def main():
    root = tk.Tk()
    app = ICTChargeLoggerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
