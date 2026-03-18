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

# ---------------- EPICS CONFIG ----------------
os.environ['EPICS_CA_ADDR_LIST'] = '192.168.0.112'
os.environ['EPICS_CA_AUTO_ADDR_LIST'] = 'NO'


REFRESH_INTERVAL = 1  # seconds

ADC_FULL_SCALE = 1.0
SAMPLE_RATE = 500e6
BUFFER_LEN = 32768
ICT_SENSITIVITY = 1.25
DEFAULT_SMOOTH_WINDOW = 10
USE_MIMIC_DEFAULT = False

# Tuning parameters
OFFSET_STATIC_THRESHOLD = 5        # minimal counts to consider correction
OFFSET_MIN_CHANGE = 2              # minimum PV-change (counts) to perform caput
BASELINE_WINDOW_FRAC = 0.10        # first 10% samples used to estimate baseline
MAD_FACTOR = 1.4826                # convert MAD -> sigma
MAD_K = 5.0                        # multiplier for sigma to form threshold
N_STABILITY_READS = 3              # repeated reads to require consensus
STABILITY_TOLERANCE = 2            # counts tolerance among repeated desired offsets
OFFSET_DEBOUNCE_SEC = 5.0          # don't flip offset more than once every N seconds


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

# def find_pulse_bounds(signal, time_axis, min_peak_frac=0.05):
#     """
#     Return start and end times (s) corresponding to 10% rise and 90% fall
#     using linear interpolation for better accuracy.
#     """
#     peak_idx = np.argmax(signal)
#     peak_val = signal[peak_idx]
#     if peak_val <= 0:
#         return None, None

#     # Ignore very small peaks (noise)
#     if peak_val < min_peak_frac * np.max(signal):
#         return None, None

#     # 10% rise before peak
#     ten_pct = 0.01 * peak_val
#     for i in range(peak_idx-1, 0, -1):
#         if signal[i] < ten_pct <= signal[i+1]:
#             # Linear interpolation
#             t_start = time_axis[i] + (ten_pct - signal[i]) * (time_axis[i+1]-time_axis[i]) / (signal[i+1]-signal[i])
#             break
#     else:
#         t_start = time_axis[0]

#     # 90% fall after peak
#     ninety_pct = 0.99 * peak_val
#     for i in range(peak_idx, len(signal)-1):
#         if signal[i] >= ninety_pct > signal[i+1]:
#             t_end = time_axis[i] + (ninety_pct - signal[i]) * (time_axis[i+1]-time_axis[i]) / (signal[i+1]-signal[i])
#             break
#     else:
#         t_end = time_axis[-1]

#     return t_start, t_end

def find_pulse_bounds_fwhm(signal, time_axis, smooth=True):
    """
    Find pulse start and end times using Full Width at Half Maximum (FWHM).
    
    Parameters
    ----------
    signal : np.ndarray
        The input ADC signal (1D array).
    time_axis : np.ndarray
        Time axis corresponding to the signal (same length as signal).
    smooth : bool
        If True, applies a small moving average to suppress noise before finding FWHM.
        
    Returns
    -------
    t_start : float or None
        Start time at half maximum.
    t_end : float or None
        End time at half maximum.
    fwhm : float or None
        Full width at half maximum (seconds).
    """
    # Optional small smoothing to suppress noise
    if smooth:
        window = 5
        if len(signal) > window:
            signal = np.convolve(signal, np.ones(window)/window, mode='same')

    # Find peak
    peak_idx = np.argmax(signal)
    peak_val = signal[peak_idx]
    if peak_val <= 0:
        return None, None, None

    # Compute half-maximum level
    half_max = 0.5 * peak_val

    # --- Find left crossing ---
    left_idx = None
    for i in range(peak_idx - 1, 0, -1):
        if signal[i] < half_max <= signal[i + 1]:
            # Linear interpolation for better precision
            frac = (half_max - signal[i]) / (signal[i + 1] - signal[i])
            t_start = time_axis[i] + frac * (time_axis[i + 1] - time_axis[i])
            left_idx = i
            break
    else:
        t_start = time_axis[0]

    # --- Find right crossing ---
    right_idx = None
    for i in range(peak_idx, len(signal) - 1):
        if signal[i] >= half_max > signal[i + 1]:
            frac = (half_max - signal[i]) / (signal[i + 1] - signal[i])
            t_end = time_axis[i] + frac * (time_axis[i + 1] - time_axis[i])
            right_idx = i
            break
    else:
        t_end = time_axis[-1]

    # Compute FWHM
    fwhm = t_end - t_start
    # print(f"pulse width {fwhm}")

    # Sanity check
    if fwhm <= 0 or np.isnan(fwhm):
        return None, None, None

    return t_start, t_end, fwhm

def find_pulse_bounds_shortpulse(signal, time_axis,pulse_width_ns, baseline=0.0, smooth=True):
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

    half_max = 0.1 * peak_val
    fall_90 = 0.1 * peak_val  # 90% fall = 10% of peak

    # --- Start time: FWHM rising edge ---
    for i in range(peak_idx - 1, 0, -1):
        if signal[i] < half_max <= signal[i + 1]:
            frac = (half_max - signal[i]) / (signal[i + 1] - signal[i])
            t_start = time_axis[i] + frac * (time_axis[i + 1] - time_axis[i])
            break
    else:
        t_start = time_axis[0]

    # --- New fixed end time ---
    if pulse_width_ns <40:
        FIXED_WIDTH = 70e-9  # 60 ns
        
        t_end = t_start + FIXED_WIDTH
        width = t_end - t_start
        
    else:    
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
    
    # Get pulse bounds
    # Method 1
    t_start, t_end, fwhm = find_pulse_bounds_fwhm(voltage, time_s)
    
    # Method 2 for shortpulses
    # t_start, t_end, fwhm = find_pulse_bounds_shortpulse(voltage, time_s)
    
    
    if t_start is None or t_end is None or t_end <= t_start:
        return None, None, None, None

    # Find indices
    start_idx = np.searchsorted(time_s, t_start)
    end_idx   = np.searchsorted(time_s, t_end)
    if end_idx <= start_idx:
        return None, None, None, None
    
    # additional method for short pulse widths enable this method in line
    
    pulse_width_ns = fwhm * 1e9

    # --- If short pulse (<250 ns), use new method ---
    if pulse_width_ns <300:
        t_start, t_end, duration = find_pulse_bounds_shortpulse(
            voltage, time_s, pulse_width_ns, baseline=0.0
        )
    
    if t_start is None or t_end is None or t_end <= t_start:
        return None, None, None, None
    
    # -----------------------------------------------------------

    # --------- Pulse integration ---------
    pulse_time = time_s[start_idx:end_idx]
    pulse_volt = voltage[start_idx:end_idx]
    Vs_pulse = np.trapz(pulse_volt, pulse_time)

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

    base_start_idx = np.searchsorted(time_s, baseline_start)
    base_end_idx   = np.searchsorted(time_s, baseline_end)

    # Check baseline region availability
    if base_end_idx > base_start_idx and base_end_idx <= len(time_s):
        baseline_time = time_s[base_start_idx:base_end_idx]
        baseline_volt = voltage[base_start_idx:base_end_idx]
        Vs_offset = np.trapz(baseline_volt, baseline_time)
    else:
        # If baseline cannot be computed, default to zero correction
        Vs_offset = 0.0

    # --------- Corrected integral ---------
    Vs_corrected = Vs_pulse - Vs_offset

    # # --------- Charge ---------
    
    charge = Vs_corrected / sensitivity
    
    # if duration <= 251e-9 and duration >= 10e-9:
    #     #  Linear fitting
    #     # correction_factor= -1608236.475*duration + 1.386

    #     # new data collection 18/11/2025
    #     # polynomial of second order  corection_factor = 4E+12*duration**2 - 2E+06*duration + 1.3109

    #     correction_factor = 4E+12*duration**2 - 2E+06*duration + 1.3109

    #     charge= charge*correction_factor
        
    pulse_samples = end_idx - start_idx

    return charge, t_start, t_end, pulse_samples



# ---------------- GUI ----------------
class ICTChargeLoggerGUI:
    def __init__(self, master):
        self.master = master
        master.title("ICT Live Charge Logger")
        master.geometry("1500x900")


        self.background_recorded = False
        self.background_data = None
        self.background_time = None


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
        # --- Fonts ---
        self.font_btn = tkFont.Font(family="Arial", size=14, weight="bold")
        self.font_label = tkFont.Font(family="Arial", size=14, weight="bold")

        # --- Frame ---
        ctl_frame = tk.Frame(master, bg="#2E86C1")
        ctl_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        tk.Label(ctl_frame, text="Source:", font=self.font_label, bg="#2E86C1", fg="white").pack(side=tk.LEFT)
        self.src_var = tk.StringVar(value="Mimic" if self.use_mimic else "Real")
        self.src_combo = ttk.Combobox(ctl_frame, textvariable=self.src_var,
                                      values=["Mimic", "Real"], width=6, font=self.font_label)
        self.src_combo.pack(side=tk.LEFT, padx=4)
        self.src_combo.bind("<<ComboboxSelected>>", self.on_source_change)

        # --- Channel Selection ---
        tk.Label(ctl_frame, text="Channel:", font=self.font_label, bg="#2E86C1", fg="white").pack(side=tk.LEFT, padx=(15, 0))
        self.ch_var = tk.StringVar(value="Ch1")  # default
        self.ch_combo = ttk.Combobox(ctl_frame, textvariable=self.ch_var,
                                     values=["Ch1", "Ch2", "Ch3", "Ch4"], width=6, font=self.font_label)
        self.ch_combo.pack(side=tk.LEFT, padx=4)
        self.ch_combo.bind("<<ComboboxSelected>>", self.on_channel_change)

        # --- Arm Number Input ---
        tk.Label(ctl_frame, text="Arm Number:", font=self.font_label, bg="#2E86C1", fg="white").pack(side=tk.LEFT, padx=4)
        self.arm_number_var = tk.StringVar(value="0")  # default = 0
        self.entry_arm = tk.Entry(ctl_frame, textvariable=self.arm_number_var, width=6, justify='center', font=self.font_label)
        self.entry_arm.pack(side=tk.LEFT, padx=4)

        # --- Attenuation Control ---
        tk.Label(ctl_frame, text="Attenuation [0–31]:", font=self.font_label, bg="#2E86C1", fg="white").pack(side=tk.LEFT, padx=(15, 0))
        self.att_var = tk.StringVar(value="31")  # default attenuation
        self.att_entry = ttk.Entry(ctl_frame, textvariable=self.att_var, width=5, font=self.font_label)
        self.att_entry.pack(side=tk.LEFT, padx=4)
        
        # Shots to Average Input ---
        tk.Label(ctl_frame, text="Avg Shots:", font=self.font_label, bg="#2E86C1", fg="white").pack(side=tk.LEFT, padx=(15, 4))
        self.avg_shots_var = tk.StringVar(value="30") # Default matching your low_charge_avg_shots
        self.entry_avg_shots = tk.Entry(ctl_frame, textvariable=self.avg_shots_var, width=5, justify='center', font=self.font_label)
        self.entry_avg_shots.pack(side=tk.LEFT, padx=4)
        # Bind event to update the value immediately when the user finishes typing
        self.entry_avg_shots.bind("<FocusOut>", self.on_avg_shots_change)
        self.entry_avg_shots.bind("<Return>", self.on_avg_shots_change)
        
        # # Bind Return key and focus-out to trigger attenuation set
        # self.att_entry.bind("<Return>", self.on_attenuation_change)
        # self.att_entry.bind("<FocusOut>", self.on_attenuation_change)
        

        self.btn_record_bg = tk.Button(
        ctl_frame,
        text="Record Background",
        font=self.font_label,
        bg="#BFC9CA",
        fg="black",
        activebackground="#AAB7B8",
        activeforeground="black",
        width=18,
        relief="raised",
        bd=2,
        command=self.record_background
        )
        self.btn_record_bg.pack(side=tk.LEFT, padx=8, pady=4)

                
        #Start and stop buttons 
        self.btn_start = tk.Button(ctl_frame, text="Start", font=self.font_btn,
                                    command=self.start_logging, bg="white")
        self.btn_start.pack(side=tk.LEFT, padx=4)
        self.btn_stop = tk.Button(ctl_frame, text="Stop", font=self.font_btn,
                                  command=self.stop_logging, bg="white", state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)
    


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
        self.low_charge_mode = True
        self.low_charge_avg_shots = int(self.avg_shots_var.get()) # number of shots to average
        self.low_charge_buffer = []    # buffer for multi-shot averaging
        
        # --- Logo (optional) ---
        try:
            logo_image = Image.open(r"G:\My Drive\Libera ADC\logo.png")
            logo_image = logo_image.resize((150, 50), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = tk.Label(ctl_frame, image=self.logo_photo, bg="#2E86C1")
            logo_label.pack(side=tk.RIGHT, padx=10)
        except Exception as e:
            print("Logo load failed:", e)

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




    def start_logging(self):
       
        try:
            att_val = float(self.att_var.get())
            if not (10 <= att_val <= 31):
                messagebox.showerror("Invalid Attenuation", 
                                     f"Attenuation {att_val} is out of bounds (0–31).\n"
                                     "Please fix this before starting.")
                return # Exit the function; does not set self.streaming to True
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a numeric value for attenuation.")
            return
        
        self.streaming = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        
        # Ask user if logging should be enabled
        answer = messagebox.askyesno("Enable Logging", "Do you want to enable data logging?")
        if answer:
            # Open CSV file
            filename = datetime.datetime.now().strftime("charge_log_%Y%m%d_%H%M%S.csv")
            self.csv_file = open(filename, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["Time", "Charge(C)", "ADC_Max_Counts","PulseWidth(s)", "AverageCharge(C)"])
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

        # Open CSV file
        filename = datetime.datetime.now().strftime("charge_log_%Y%m%d_%H%M%S.csv")
        self.csv_file = open(filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Write header
        self.csv_writer.writerow(["Time", "Charge(C)", "ADC_Max_Counts","PulseWidth(s)", "AverageCharge(C)"])

      
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
        
        # ----------------------------
        # TEMPORARY Averaging multiple shots
        # ----------------------------
        if self.low_charge_mode:
            # Append current shot
            self.low_charge_buffer.append(v_smoothed)
        
            # Wait until we have enough shots
            if len(self.low_charge_buffer) < self.low_charge_avg_shots:
                self._schedule_update()
                return
        
            # Peak-align and average
            center_idx = len(v_smoothed)//2
            aligned_shots = []
        
            for shot in self.low_charge_buffer:
                peak_idx = np.argmax(shot)
                shift = center_idx - peak_idx
                shot_aligned = np.roll(shot, shift)
                aligned_shots.append(shot_aligned)
        
            v_corrected = np.mean(aligned_shots, axis=0)
            self.low_charge_buffer.clear()
        
            charge, t_start, t_end, pulse_samples = compute_auto_charge(
                self.time_axis, v_corrected, ICT_SENSITIVITY
            )
            if charge is not None:
                charge /= self.low_charge_avg_shots
        
        # else:
        #     # Normal mode
        #     v_corrected = v_smoothed
        #     t_start, t_end, width = compute_auto_charge(self.time_axis, v_corrected, ICT_SENSITIVITY)[:3]
        #     charge = compute_auto_charge(self.time_axis, v_corrected, ICT_SENSITIVITY)[0]

        # --- Apply background subtraction if active ---
        if getattr(self, "background_active", False) and self.background_data is not None:
            if len(self.background_data) == len(v_smoothed):
                # Align baselines before subtraction
                offset = np.mean(v_smoothed) - np.mean(self.background_data)
                v_corrected = v_smoothed - (self.background_data + offset)
            else:
                v_corrected = v_smoothed
                print("⚠️ Background length mismatch — skipping subtraction")
        else:
            v_corrected = v_smoothed

        
        # Store latest waveform for possible background recording
        self.current_waveform_y = v_smoothed
        self.current_waveform_x = self.time_axis
        
                
        
        peak_adc=np.max(v_raw)
        # --- Compute charge and pulse parameters ---
        charge, t_start, t_end, pulse_samples = compute_auto_charge(
            self.time_axis, v_corrected, ICT_SENSITIVITY
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
            avg_charge=np.mean(self.charges[-100:])
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
                        



            # --- Background subtraction (if recorded) ---
            if getattr(self, "background_recorded", False) and self.background_data is not None:
                # Ensure lengths match
                if len(self.background_data) == len(v_smoothed):
                    v_corrected = v_smoothed - self.background_data
                else:
                    # Handle mismatch gracefully
                    v_corrected = v_smoothed
                    print("⚠️ Background length mismatch — skipping subtraction")
            else:
                v_corrected = v_smoothed
            
            # --- Update right plot (Waveform) ---
            self.line_waveform.set_data(self.time_axis * 1e9, v_corrected)

    
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
        pulse_width = t_end - t_start
        # ADC counts in pulse range
        adc_counts_max = np.max(v_raw[int(t_start*self.sample_rate):int(t_end*self.sample_rate)])
    
        # Average of last 10 values
        avg_charge = np.mean(self.charges[-20:])
    
        # --- Write CSV ---
        # elapsed_seconds = time.time() - self.start_time_datetime
        elapsed_seconds=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.csv_writer:
            self.csv_writer.writerow([elapsed_seconds, charge, adc_counts_max, pulse_width, avg_charge])
       
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

            
    
# ---------------- MAIN ----------------
def main():
    root = tk.Tk()
    app = ICTChargeLoggerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
