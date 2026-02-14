import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from pathlib import Path
from collections import deque

from DSP_tasks.task1.helpers import *
from DSP_tasks.task3.helpers import *
from DSP_tasks.task5.helpers import *
#from DSP_tasks.task6.helpers import *
from DSP_tasks.task7.helpers import *


# =================================================================

def on_closing():
    root.quit()
    root.destroy()


def plot_signal(ax_obj, indices, samples, title, canvas_obj):
    ax_obj.clear()
    if plot_mode_var.get() == "Discrete":
        ax_obj.stem(indices, samples, basefmt=" ")
    else:
        ax_obj.plot(indices, samples, 'b-', linewidth=2)
    ax_obj.set_title(title)
    ax_obj.set_xlabel("Index")
    ax_obj.set_ylabel("Amplitude")
    ax_obj.grid(True)
    canvas_obj.draw()


def load_signal(file_name=None):
    if file_name == None:
        file_name = filedialog.askopenfilename(
            title="Select a Signal File",
            filetypes=[("Text Files", "*.txt")])
    if not file_name:
        return
    indicies, samples = ReadSignalFile(file_name)
    signal_name = Path(file_name).stem
    signals[signal_name] = (indicies, samples)
    options = list(signals.keys())
    for combo_box in combo_boxes:
        combo_box["values"] = options
        combo_box.set(options[-1])


def display_signal(combo_obj, ax_obj, canvas_obj):
    signal = signals[combo_obj.get()]
    plot_signal(ax_obj, signal[0], signal[1], title=combo_obj.get(), canvas_obj=canvas_obj)


def add(signal1, signal2):
    indcies, samples = [], []
    indicies1, samples1 = signal1[0], signal1[1]
    indicies2, samples2 = signal2[0], signal2[1]
    len1, len2 = len(indicies1), len(indicies2)
    i, j = 0, 0
    while i < len1 and j < len2:
        if indicies1[i] < indicies2[j]:
            indcies.append(indicies1[i])
            samples.append(samples1[i])
            i += 1
        elif indicies1[i] > indicies2[j]:
            indcies.append(indicies2[j])
            samples.append(samples2[j])
            j += 1
        else:
            indcies.append(indicies1[i])
            samples.append(samples1[i] + samples2[j])
            i += 1
            j += 1
    while i < len1:
        indcies.append(indicies1[i])
        samples.append(samples1[i])
        i += 1
    while j < len2:
        indcies.append(indicies2[j])
        samples.append(samples2[j])
        j += 1
    return indcies, samples


def add_signals():
    if not combo1.get() or not combo2.get():
        messagebox.showerror("Signal Error", "Please Select a Signal1 and Signal2 first.")
        return
    signal1 = signals[combo1.get()]
    signal2 = signals[combo2.get()]
    indicies, samples = add(signal1, signal2)
    plot_signal(ax, indicies, samples, title="Added Signal", canvas_obj=canvas)
    if combo1.get() == "Signal1" and combo2.get() == "Signal2":
        AddSignalSamplesAreEqual("Signal1.txt", "Signal2.txt", indicies, samples)


def subtract_signals():
    if not combo1.get() or not combo2.get():
        messagebox.showerror("Signal Error", "Please Select a Signal1 and Signal2 first.")
        return
    signal1 = signals[combo1.get()]
    signal2 = signals[combo2.get()]
    negative_signal2 = (signal2[0], [-v for v in signal2[1]])
    indicies, samples = add(signal1, negative_signal2)
    plot_signal(ax, indicies, samples, title="Subtracted Signal", canvas_obj=canvas)
    if combo1.get() == "Signal1" and combo2.get() == "Signal2":
        SubSignalSamplesAreEqual("Signal1.txt", "Signal2.txt", indicies, samples)


def shift_indicies(indicies, k):
    return [i + k for i in indicies]


def multiply_samples(samples, k):
    return [i * k for i in samples]


def multiply_signal_byConst():
    if not combo.get():
        messagebox.showerror("Signal Error", "Please Select a Signal first.")
        return
    constant = simpledialog.askfloat("Input", "Enter a constant value to multiply by:")
    if constant is None: return
    selected_index = combo.get()
    indicies, samples = signals[selected_index]
    new_samples = multiply_samples(samples, constant)
    plot_signal(ax, indicies, new_samples, title=f"{selected_index} Multiplied by {constant}", canvas_obj=canvas)
    if combo.get() == "Signal1" and constant == 5:
        MultiplySignalByConst(5, indicies, new_samples)


def shift_signal():
    if not combo.get():
        messagebox.showerror("Signal Error", "Please Select a Signal first.")
        return
    k = simpledialog.askinteger("Input", "Enter shift amount 'k':\n(k > 0 for delay, k < 0 for advance)")
    if k is None:
        return
    indicies, samples = signals[combo.get()]
    new_indicies = shift_indicies(indicies, -k)
    plot_signal(ax, new_indicies, samples, title=f"{combo.get()} shifted by {k}", canvas_obj=canvas)
    if combo.get() == "Signal1" and abs(k) == 3:
        ShiftSignalByConst(k, new_indicies, samples)


def reverse_signal():
    if not combo.get():
        messagebox.showerror("Signal Error", "Please Select a Signal first.")
        return
    indicies, samples = signals[combo.get()]
    new_indicies = [-i for i in indicies]
    new_indicies.reverse()
    new_samples = list(reversed(samples))
    plot_signal(ax, new_indicies, new_samples, title=f"{combo.get()} Reversed", canvas_obj=canvas)
    if combo.get() == "Signal1":
        Folding(new_indicies, new_samples)


# =================================================================


def open_popup():
    popup = tk.Toplevel(root)
    popup.title("Quantization Settings")
    popup.geometry("300x150+700+500")
    popup.grab_set()
    ttk.Label(popup, text="Select input type and value:").pack(pady=5)
    result = tk.IntVar(value=0)
    frame = ttk.Frame(popup)
    frame.pack(pady=5)
    left_frame = ttk.Frame(frame)
    left_frame.pack(side=tk.LEFT, padx=10)
    choice_var = tk.StringVar(value="levels")
    ttk.Radiobutton(left_frame, text="Levels", variable=choice_var, value="levels").pack(anchor="w")
    ttk.Radiobutton(left_frame, text="Bits", variable=choice_var, value="bits").pack(anchor="w")
    right_frame = ttk.Frame(frame)
    right_frame.pack(side=tk.LEFT, padx=10)
    ttk.Label(right_frame, text="Value:").pack()
    spin_val = ttk.Spinbox(right_frame, from_=1, to=100, width=15)
    spin_val.set(1)
    spin_val.pack()

    def confirm():
        choice = choice_var.get()
        try:
            levels = int(spin_val.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid number.")
            return
        if levels < 1:
            messagebox.showerror("Input Error", "Value must be at least 1.")
            return
        if choice == "bits":
            levels = 2 ** levels
        result.set(levels)
        popup.destroy()

    ttk.Button(popup, text="OK", command=confirm).pack(pady=10)
    popup.wait_window()
    return result.get()


def show_quantization_table(original_indices, interval_indices, encoded_values, quantized_values, error_values):
    win = tk.Toplevel(root)
    win.title("Quantization Details")
    win.geometry("700x400")
    win.grab_set()
    tree_frame = ttk.Frame(win)
    tree_frame.pack(pady=10, padx=10, fill="both", expand=True)
    tree_scroll = ttk.Scrollbar(tree_frame)
    tree_scroll.pack(side="right", fill="y")
    cols = ("Sample Index", "Interval Index", "Encoded Value", "Quantized Value", "Error")
    tree = ttk.Treeview(tree_frame, columns=cols, show="headings", yscrollcommand=tree_scroll.set)
    tree.pack(fill="both", expand=True)
    tree_scroll.config(command=tree.yview)
    for col in cols:
        tree.heading(col, text=col, anchor="center")
        tree.column(col, anchor="center", width=130)
    for i in range(len(original_indices)):
        q_val_str = f"{quantized_values[i]:.3f}"
        err_val_str = f"{error_values[i]:.3f}"
        tree.insert("", "end",
                    values=(original_indices[i], interval_indices[i], encoded_values[i], q_val_str, err_val_str))


def quantize_signal():
    if not combo.get():
        messagebox.showerror("Signal Error", "Please Select a Signal first.")
        return
    no_levels = open_popup()
    if not no_levels:
        return
    indicies, samples = signals[combo.get()]
    min_sample = min(samples)
    max_sample = max(samples)
    delta = (max_sample - min_sample) / no_levels
    levels = [min_sample + (i + 0.5) * delta for i in range(no_levels)]
    quantizied_samples = []
    quantizied_indices = []
    quantizied_encode = []
    quantized_error = []  # Corrected: Initialized here
    bit_width = (no_levels - 1).bit_length()
    for sample in samples:
        sample_index = int(np.floor((sample - min_sample) / delta))
        if sample_index == no_levels:
            sample_index -= 1
        quantizied_samples.append(levels[sample_index])
        quantizied_indices.append(sample_index + 1)
        quantizied_encode.append(format(sample_index, f'0{bit_width}b'))
        quantized_error.append(levels[sample_index] - sample)
    show_quantization_table(indicies, quantizied_indices, quantizied_encode, quantizied_samples, quantized_error)
    if combo.get() == "Quan1_input" and no_levels == 8:
        file_path = "DSP_tasks/task3/Quan1_Out.txt"
        QuantizationTest1(file_path, quantizied_encode, quantizied_samples)
    elif combo.get() == "Quan2_input" and no_levels == 4:
        file_path = "DSP_tasks/task3/Quan2_Out.txt"
        QuantizationTest2(file_path, quantizied_indices, quantizied_encode, quantizied_samples, quantized_error)
    plot_signal(ax, indicies, quantizied_samples, title=f"{combo.get()} Quantized to {no_levels} levels",
                canvas_obj=canvas)


def generate_signal():
    wave_type = wave_type_var.get()
    try:
        A = float(amp_entry.get())
        theta_deg = float(phase_entry.get())
        F = float(freq_entry.get())
        Fs = float(fs_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric samples for all parameters.")
        return
    if Fs < 2 * F:
        messagebox.showerror("Input Error",
                             "Sampling frequency Fs must be greater than twice the analog frequency F (Fs > 2F).")
        return
    theta_rad = np.deg2rad(theta_deg)
    duration = 1
    if F > 0:
        duration = 2 / F
    num_samples = int(duration * Fs)
    n = np.arange(num_samples)
    if wave_type == "Sine":
        samples = A * np.sin(2 * np.pi * F / Fs * n + theta_rad)
    else:
        samples = A * np.cos(2 * np.pi * F / Fs * n + theta_rad)
    selected_plot = plot_selector_gen.get()
    target_ax = ax_gen1 if selected_plot == "Plot 1" else ax_gen2
    plot_signal(target_ax, n, samples, title=f"Generated {wave_type} Signal", canvas_obj=canvas_gen)
    if selected_plot == "Plot 2":
        target_ax.set_xlabel("Sample Index (n)")
    fig_gen.tight_layout()
    canvas_gen.draw()


def save_signal(ax):
    lines = ax.get_lines()
    stems = getattr(ax, 'containers', None)
    x_data, y_data = None, None
    if lines:
        line = lines[0]
        x_data, y_data = line.get_xdata(), line.get_ydata()
    elif stems:
        stemlines = [c for c in ax.containers if isinstance(c, plt.StemContainer)]
        if stemlines:
            stemline = stemlines[0]
            x_data, y_data = stemline.markerline.get_data()
    if x_data is None or y_data is None:
        messagebox.showerror("Error", "No signal found on this plot.")
        return
    signal_name = simpledialog.askstring("Save Signal", "Enter a name for the signal:")
    if not signal_name:
        return
    if signal_name in signals:
        if not messagebox.askyesno("Overwrite?",
                                   f"A signal named '{signal_name}' already exists. Do you want to overwrite it?"):
            return
    signals[signal_name] = (list(x_data), list(y_data))
    messagebox.showinfo("Saved", f"Signal '{signal_name}' saved successfully!")
    options = list(signals.keys())
    for combobox in combo_boxes:
        combobox["values"] = options


# =================================================================


def moving_avg_signal():
    window_size = simpledialog.askinteger("Input", "Enter window size for moving average: ")
    if window_size is None or window_size < 1:
        messagebox.showerror("Input Error", "Window size must be a positive integer.")
        return
    indices, samples = signals[combo_conv.get()]
    new_indicies, new_samples = [], []
    max_indix = indices[-1]
    for i in range(window_size, max_indix + 2):
        new_indicies.append(i - window_size)
        sum = 0
        for j in range(i - window_size, i):
            sum += samples[j]
        new_samples.append(round(sum / window_size, 3))
    plot_signal(ax_conv, new_indicies, new_samples, title=f"Moving Average of {combo_conv.get()}",
                canvas_obj=canvas_conv)
    if combo_conv.get() == 'MovingAvg_input':
        if window_size == 3:
            if CheckSignalsEquality("DSP_tasks/task4/Moving Average testcases/MovingAvg_out1.txt", new_indicies, new_samples):
                print("Moving Average W = 3 Test case passed successfully")
            else:
                print("Moving Average W = 3 Test case Failed")
        elif window_size == 5:
            if CheckSignalsEquality("DSP_tasks/task4/Moving Average testcases/MovingAvg_out2.txt", new_indicies, new_samples):
                print("Moving Average W = 5 Test case passed successfully")
            else:
                print("Moving Average W = 5 Test case Failed")


def first_deriv_signal():
    indices, samples = signals[combo_conv.get()]
    new_indicies, new_samples = [], []
    max_indix = indices[-1]
    for i in range(1, max_indix + 1):
        new_indicies.append(i - 1)
        new_samples.append(int(samples[i] - samples[i - 1]))
    plot_signal(ax_conv, new_indicies, new_samples, title=f"1st Derivative of {combo_conv.get()}",
                canvas_obj=canvas_conv)
    if combo_conv.get() == 'Derivative_input':
        if CheckSignalsEquality("DSP_tasks/task4/Derivative testcases/1st_derivative_out.txt", new_indicies, new_samples):
            print("First Derivative Test case passed successfully")
        else:
            print("First Derivative Test case Failed")


def second_deriv_signal():
    indices, samples = signals[combo_conv.get()]
    new_indicies, new_samples = [], []
    max_indix = indices[-1]
    for i in range(1, max_indix):
        new_indicies.append(i - 1)
        new_samples.append(int(samples[i + 1] - (2 * samples[i]) + samples[i - 1]))
    plot_signal(ax_conv, new_indicies, new_samples, title=f"2nd Derivative of {combo_conv.get()}",
                canvas_obj=canvas_conv)
    if combo_conv.get() == 'Derivative_input':
        if CheckSignalsEquality("DSP_tasks/task4/Derivative testcases/2nd_derivative_out.txt", new_indicies, new_samples):
            print("Second Derivative Test case passed successfully")
        else:
            print("Second Derivative Test case Failed")


def convolute(ind1, samp1, ind2, samp2):
    size = len(ind1)
    new_indicies, new_samples = [], []
    for i in range(size):
        cur_sig = (shift_indicies(ind2, ind1[i]), multiply_samples(samp2, samp1[i]))
        new_indicies, new_samples = add((new_indicies, new_samples), cur_sig)
    return new_indicies, new_samples

def convolute_signals():
    ind1, samp1 = signals[combo1_conv.get()]
    ind2, samp2 = signals[combo2_conv.get()]
    new_indicies, new_samples = convolute(ind1, samp1, ind2, samp2)
    plot_signal(ax_conv, new_indicies, new_samples, title=f"Convolution of {combo1_conv.get()} and {combo2_conv.get()}",
                canvas_obj=canvas_conv)
    if combo1_conv.get() == 'Signal 1' and combo2_conv.get() == 'Signal 2':
        if CheckSignalsEquality("DSP_tasks/task4/Convolution testcases/Conv_output.txt", new_indicies, new_samples):
            print("Convolution Test case passed successfully")
        else:
            print("Convolution Test case Failed")


def CheckSignalsEquality(file_name, Your_indices, Your_samples):
    expected_indices, expected_samples = ReadSignalFile(file_name)
    if (len(expected_samples) != len(Your_samples)) or (len(expected_indices) != len(Your_indices)):
        return False
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            return False
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            return False
    return True


# =================================================================


def load_freq_signal(file_name=None):
    if file_name == None:
        file_name = filedialog.askopenfilename(
            title="Select a Frequency Signal File",
            filetypes=[("Text Files", "*.txt")])
    if not file_name:
        return
    amplitudes, phases = ReadFreqSignalFile(file_name)
    signal_name = Path(file_name).stem
    freq_signals[signal_name] = (amplitudes, phases)
    options = list(freq_signals.keys())
    for combo_box in freq_combo_boxes:
        combo_box["values"] = options
        combo_box.set(options[-1])


def display_freq_signal(combo_obj, ax_obj1, ax_obj2, canvas_obj):
    signal = freq_signals[combo_obj.get()]

    fs_val = simpledialog.askfloat("Input", "Enter Sampling Frequency (Hz):")
    if fs_val is None or fs_val < 0:
        messagebox.showerror("Input Error", "Sampling Frequency must be positive.")
        return
    N = len(signal[0])
    frequencies = [k * fs_val / N for k in range(N)]
    plot_signal(ax_obj1, frequencies, signal[0], "Frequency vs Amplitude", canvas_freq)
    plot_signal(ax_obj2, frequencies, signal[1], "Frequency vs Phase (in Radian)", canvas_freq)


def fourier_series(samples, value):
    N = len(samples)
    if value == "Inverse":
        divisor = N
        omega_fundamental = (2 * np.pi) / N * 1j
    else:
        divisor = 1
        omega_fundamental = (2 * np.pi) / N * (-1j)
    frequencies = []
    for k in range(N):
        X = 0 + 0j
        for n in range(N):
            theta = omega_fundamental * k * n
            X += np.exp(theta) * samples[n]
        frequencies.append(X / divisor)
    return frequencies


def fourier_transform():
    fs_val = simpledialog.askfloat("Input", "Enter Sampling Frequency (Hz):")
    
    if fs_val is None or fs_val < 0:
        messagebox.showerror("Input Error", "Sampling Frequency must be positive.")
        return

    indices, samples = signals[combo_freq.get()]
    N = len(samples)
    series = fourier_series(samples, "Direct")
    frequencies, amplitudes, phases = [], [], []
    for k, X in enumerate(series):
        amplitudes.append(np.sqrt(X.real ** 2 + X.imag ** 2))
        phases.append(np.arctan2(X.imag, X.real))
        frequencies.append(k * fs_val / N)

    if combo_freq.get() == "Input_Signal_DFT":
        expected_amplitudes, expected_phases = ReadFreqSignalFile("DSP_tasks/task5/DFT/Output_Signal_DFT.txt")
        if SignalComapreAmplitude(np.round(expected_amplitudes,10), np.round(amplitudes,10)) and SignalComaprePhaseShift(expected_phases, phases):
            print("DFT Test case passed successfully")
        else:
            print("DFT Test case Failed")

    plot_signal(ax_freq_mag, frequencies, amplitudes, "Frequency vs Amplitude", canvas_freq)
    plot_signal(ax_freq_phase, frequencies, phases, "Frequency vs Phase (in Radian)", canvas_freq)


def inverse_fourier_transform():
    amplitudes, phases = freq_signals[combo1_freq.get()]
    N = len(amplitudes)
    complex_form = [amplitudes[i] * (np.cos(phases[i]) + 1j * np.sin(phases[i])) for i in range(N)]

    indicies = np.arange(N).tolist()
    samples = fourier_series(complex_form, "Inverse")
    samples = [round(sample.real) for sample in samples]
    
    if combo1_freq.get() == "Input_Signal_IDFT":
        if CheckSignalsEquality("DSP_tasks/task5/IDFT/Output_Signal_IDFT.txt", indicies, samples):
            print("IDFT Test case passed successfully")
        else:
            print("IDFT Test case Failed")
    
    ax_freq_phase.clear()
    plot_signal(ax_freq_mag, indicies, samples, "Original Signal", canvas_freq)


# =================================================================


def correlate(samples1, samples2, normalized=True):
    if len(samples1) != len(samples2):
        print("Error: Cannot correlate signals of diffrent lengths")
        return
    samples2 = deque(samples2)
    if normalized:
        divisor = np.sqrt(sum([x*x for x in samples1]) * sum([x*x for x in samples2]))
    else:
        divisor = 1
    corr_samples = []
    for i in range(len(samples1)):
        corr_samples.append(np.dot(samples1, samples2) / divisor)
        samples2.append(samples2[0])
        samples2.popleft()
    return corr_samples


def correlate_signals():
    indicies1, samples1 = signals[combo1_corr.get()]
    indicies2, samples2 = signals[combo2_corr.get()]
    corr_indicies = indicies1.copy()
    corr_samples = correlate(samples1, samples2, isNormalized.get())
    if combo1_corr.get() == "Corr_input signal1" and combo2_corr.get() == "Corr_input signal2" and isNormalized.get():
        Compare_Signals("DSP_tasks/task6/Point1 Correlation/CorrOutput.txt", corr_indicies, corr_samples)

    plot_signal(ax_corr, corr_indicies, corr_samples, title=f"Correlation of {combo1_corr.get()} and {combo2_corr.get()}", canvas_obj=canvas_corr)


def calc_delay():
    indicies1, samples1 = signals[combo1_corr.get()]
    indicies2, samples2 = signals[combo2_corr.get()]
    corr_indicies = indicies1.copy()
    corr_samples = correlate(samples1, samples2)
    Fs = simpledialog.askfloat("Input", "Enter Sampling Frequency (Hz):")
    if Fs is None or Fs < 0:
        messagebox.showerror("Input Error", "Sampling Frequency must be positive.")
        return
    time_delay = np.argmax(corr_samples) / Fs # max Index * Ts
    if combo1_corr.get() == "TD_input signal1" and combo2_corr.get() == "TD_input signal2" and Fs == 100:
        if time_delay == 5/100:
            print("Time Delay Test case passed successfully")
        else:
            print("Time Delay Test case Failed")
    messagebox.showinfo("Time Delay", f"There is {time_delay:.2f} sec Delay.")


def load_samples(file_name=None, save_signal=True):
    if file_name == None:
        file_name = filedialog.askopenfilename(
            title="Select a Signal File",
            filetypes=[("Text Files", "*.txt")])
    if not file_name:
        return
    samples = read_samples(file_name)
    indicies = np.arange(len(samples))
    if not save_signal:
        return (indicies, samples)
    else:
        signal_name = Path(file_name).stem
        signals[signal_name] = (indicies, samples)
        options = list(signals.keys())
        for combo_box in combo_boxes:
            combo_box["values"] = options
            combo_box.set(options[-1])


def load_class(dir_name=None):
    if dir_name == None:
        dir_name = filedialog.askdirectory(title="Select Class Folder")
    if not dir_name:
        print("No folder selected.")
        return
    directory = Path(dir_name)
    curr_class = []
    for file_path in directory.glob("*.txt"):
        curr_class.append(load_samples(file_path, save_signal=False))

    if len(curr_class) == 0:
        return
    class_name = directory.name
    classes[class_name] = curr_class
    combo_classes['values'] = list(classes.keys())
    combo_classes.set(class_name)


def classify_signal():
    indicies, samples = signals[combo_corr.get()]
    classes_corr = {}
    for class_name, class_signals in classes.items():
        corr_sum = 0
        for cur_ind, cur_samp in class_signals:
            corr_sum += max(correlate(samples, cur_samp))
        classes_corr[class_name] = corr_sum / len(class_signals)
        # print(f"Correlation = {classes_corr[class_name]}")
    messagebox.showinfo("Class Name", f"{combo_corr.get()} signal belongs to {max(classes_corr, key=classes_corr.get)}.")


# =================================================================

def generate_filter():
    try:
        filter_type = combo_filters.get()
        sampling_freq = float(sample_freq_entry.get())
        trans_band = float(tb_entry.get()) / sampling_freq
        cutoff_freq1 = float(cuttOff_freq1_entry.get()) / sampling_freq + 0.5 * trans_band
        cutoff_freq2 = float(cuttOff_freq2_entry.get()) / sampling_freq + 0.5 * trans_band
        stop_anttenuation = float(ann_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric samples for all parameters.")
        return

    if stop_anttenuation <= 21 :
        N = int(np.ceil(0.9 / trans_band)) 
        N = N if N % 2 != 0 else N + 1
        n = int((N - 1) / 2)
        w = [1 for i in range(N)]
    elif stop_anttenuation <= 44 :
        N = int(np.ceil(3.1 / trans_band))
        N = N if N % 2 != 0 else N + 1
        n = int((N - 1) / 2)
        w = [0.5 + 0.5 * np.cos((2 * np.pi * i) / N) for i in range(-n, n + 1)]
    elif stop_anttenuation <= 53 :
        N = int(np.ceil(3.3 / trans_band))
        N = N if N % 2 != 0 else N + 1
        n = int((N - 1) / 2)
        w = [0.54 + 0.46 * np.cos((2 * np.pi * i) / N) for i in range(-n, n + 1)]
    elif stop_anttenuation <= 74 :
        N = np.ceil(5.5 / trans_band) 
        N = N if N % 2 != 0 else N + 1
        n = int((N - 1) / 2)
        w = [0.42 + 0.5 * np.cos((2 * np.pi * i) / (N-1) ) + 0.08 * np.cos((4 * np.pi * i) / (N-1)) for i in range(-n, n + 1)]
    else:
        messagebox.showerror("KDA KFAYA","SHIT")
        return
    
    if filter_type == "Low Pass":
        hD = []
        for i in range(-n, n + 1):
            if i == 0:
                hD.append(2 * cutoff_freq1)
            else:
                hD.append(np.sin(2 * np.pi * cutoff_freq1 * i) / (np.pi * i))

    elif filter_type == "High Pass":
        hD = []
        for i in range(-n, n + 1):
            if i == 0:
                hD.append(1 - 2 * cutoff_freq1)
            else:
                hD.append(- np.sin(2 * np.pi * cutoff_freq1 * i) / (np.pi * i))
    
    elif filter_type == "Band Pass":
        hD = []
        for i in range(-n, n + 1):
            if i == 0:
                hD.append(2 * (cutoff_freq2 - cutoff_freq1))
            else:
                hD.append(np.sin(2 * np.pi * cutoff_freq2 * i) / (i * np.pi) - np.sin(2 * np.pi * cutoff_freq1 * i) / (i * np.pi))
    
    elif filter_type == "Band Stop":
        hD = []
        for i in range(-n, n + 1):
            if i == 0:
                hD.append(1 - 2 * (cutoff_freq2 - cutoff_freq1))
            else:
                hD.append(np.sin(2 * np.pi * cutoff_freq1 * i) / (i * np.pi) -  np.sinc(2 * np.pi * cutoff_freq2 * i) / (i * np.pi))

    h = [hD[i] * w[i] for i in range(N)]
    global current_filter
    current_filter = h
    indicies = np.arange(-n, n + 1).tolist()
    #Compare_Signals("DSP_tasks/task7/FIR test cases/Testcase 1/LPFCoefficients.txt", list(range(-n,n+1)), h)

current_filter = []

def apply_filter():
    indicies, samples = signals[combo_sigfilter.get()]
    generate_filter()
    global current_filter
    n = int((len(current_filter) - 1) / 2)
    filter_indicies = np.arange(-n,n + 1).tolist()
    conv_indicies, conv_samples = convolute(indicies, samples, filter_indicies, current_filter)
    
    Compare_Signals("DSP_tasks/task7/FIR test cases/Testcase 2/ecg_low_pass_filtered.txt", conv_indicies, conv_samples)

    


# =================================================================
signals = {}
freq_signals = {}

root = tk.Tk()
root.geometry("1200x800")
root.title("Signal Processing GUI")
root.protocol("WM_DELETE_WINDOW", on_closing)
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both", padx=10, pady=10)
plot_mode_var = tk.StringVar(value="Discrete")

# main tabs
operation_tab = ttk.Frame(notebook)
notebook.add(operation_tab, text="Signal Operations")
control_frame = ttk.Frame(operation_tab)
control_frame.pack(side="left", fill="y", padx=(0, 5))
plot_frame = ttk.Frame(operation_tab)
plot_frame.pack(side="right", fill="both", expand=True)
fig, ax = plt.subplots(figsize=(7, 5))
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)
ttk.Button(control_frame, text="Load Signal", command=load_signal).pack(fill=tk.X, padx=5, pady=5)
ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10, padx=5)
ttk.Label(control_frame, text="Plot Type:").pack()
plot_mode_ops = ttk.Combobox(control_frame, values=["Discrete", "Continuous"], textvariable=plot_mode_var,
                             state="readonly")
plot_mode_ops.pack(pady=5)
ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10, padx=5)
ttk.Label(control_frame, text="Select Signal for Operations:").pack(pady=(5, 0))
combo = ttk.Combobox(control_frame, values=[], state="readonly")
combo.pack(pady=10)
ttk.Button(control_frame, text="Display Signal", command=lambda: display_signal(combo, ax, canvas)).pack(fill=tk.X,
                                                                                                         padx=5, pady=5)
ttk.Button(control_frame, text="Multiply by constant", command=multiply_signal_byConst).pack(fill=tk.X, padx=5, pady=5)
ttk.Button(control_frame, text="Shift by constant", command=shift_signal).pack(fill=tk.X, padx=5, pady=5)
ttk.Button(control_frame, text="Reverse Signal", command=reverse_signal).pack(fill=tk.X, padx=5, pady=5)
ttk.Button(control_frame, text="Quantize Signal", command=quantize_signal).pack(fill=tk.X, padx=5, pady=5)
ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10, padx=5)
ttk.Label(control_frame, text="Signal 1:").pack()
combo1 = ttk.Combobox(control_frame, values=[], state="readonly")
combo1.pack(pady=10)
ttk.Label(control_frame, text="Signal 2:").pack()
combo2 = ttk.Combobox(control_frame, values=[], state="readonly")
combo2.pack(pady=10)
ttk.Button(control_frame, text="Add Signals", command=add_signals).pack(fill=tk.X, padx=5, pady=5)
ttk.Button(control_frame, text="Subtract Signals", command=subtract_signals).pack(fill=tk.X, padx=5, pady=5)
ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10, padx=5)
ttk.Button(control_frame, text="Save Signal", command=lambda: save_signal(ax)).pack(side="bottom", fill=tk.X, padx=10,
                                                                                    pady=10)

# --- Generate Signals Tab ---
gen_tab = ttk.Frame(notebook)
notebook.add(gen_tab, text="Generate Signals")
control_frame_gen = ttk.Frame(gen_tab)
control_frame_gen.pack(side="left", fill="y", padx=(0, 5))
plot_frame_gen = ttk.Frame(gen_tab)
plot_frame_gen.pack(side="right", fill="both", expand=True)
fig_gen, (ax_gen1, ax_gen2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 5))
ax_gen1.set_title("Plot 1")
ax_gen2.set_title("Plot 2")
fig_gen.tight_layout()
canvas_gen = FigureCanvasTkAgg(fig_gen, master=plot_frame_gen)
canvas_gen.get_tk_widget().pack(fill="both", expand=True)
ttk.Label(control_frame_gen, text="Plot Type:").pack(pady=(10, 0))
plot_type_frame = ttk.Frame(control_frame_gen)
ttk.Radiobutton(plot_type_frame, text="Discrete", variable=plot_mode_var, value="Discrete").pack(side="left", padx=5)
ttk.Radiobutton(plot_type_frame, text="Continuous", variable=plot_mode_var, value="Continuous").pack(side="left",
                                                                                                     padx=5)
plot_type_frame.pack()
ttk.Separator(control_frame_gen, orient='horizontal').pack(fill='x', pady=10, padx=5)
ttk.Label(control_frame_gen, text="Wave Type:").pack()
wave_type_var = tk.StringVar(value="Sine")
wave_type_frame = ttk.Frame(control_frame_gen)
ttk.Radiobutton(wave_type_frame, text="Sine", variable=wave_type_var, value="Sine").pack(side="left", padx=5)
ttk.Radiobutton(wave_type_frame, text="Cosine", variable=wave_type_var, value="Cosine").pack(side="left", padx=5)
wave_type_frame.pack()
param_frame = ttk.Frame(control_frame_gen)
param_frame.pack(pady=10, padx=10)
ttk.Label(param_frame, text="Amplitude (A):").grid(row=0, column=0, sticky="w", pady=2)
amp_entry = ttk.Spinbox(param_frame, from_=1, to=100000)
amp_entry.grid(row=0, column=1, pady=2)
ttk.Label(param_frame, text="Phase Shift θ (°):").grid(row=1, column=0, sticky="w", pady=2)
phase_entry = ttk.Spinbox(param_frame, from_=0, to=100000)
phase_entry.grid(row=1, column=1, pady=2)
ttk.Label(param_frame, text="Analog Freq F (Hz):").grid(row=2, column=0, sticky="w", pady=2)
freq_entry = ttk.Spinbox(param_frame, from_=0, to=100000)
freq_entry.grid(row=2, column=1, pady=2)
ttk.Label(param_frame, text="Sampling Freq Fs (Hz):").grid(row=3, column=0, sticky="w", pady=2)
fs_entry = ttk.Spinbox(param_frame, from_=0, to=100000)
fs_entry.grid(row=3, column=1, pady=2)
ttk.Separator(control_frame_gen, orient='horizontal').pack(fill='x', pady=10, padx=5)
ttk.Label(control_frame_gen, text="Target Plot:").pack()
plot_selector_gen = tk.StringVar(value="Plot 1")
target_plot_frame = ttk.Frame(control_frame_gen)
ttk.Radiobutton(target_plot_frame, text="Plot 1", variable=plot_selector_gen, value="Plot 1").pack(side="left", padx=5)
ttk.Radiobutton(target_plot_frame, text="Plot 2", variable=plot_selector_gen, value="Plot 2").pack(side="left", padx=5)
target_plot_frame.pack()
ttk.Button(control_frame_gen, text="Generate Signal", command=generate_signal).pack(fill=tk.X, padx=10, pady=10)
ttk.Button(control_frame_gen, text="Save Signal",
           command=lambda: save_signal(ax_gen1 if plot_selector_gen.get() == "Plot 1" else ax_gen2)).pack(fill=tk.X,
                                                                                                          padx=10,
                                                                                                          pady=10)

##  --- Convolution Tab ---
conv_tab = ttk.Frame(notebook)
notebook.add(conv_tab, text="Correlation & Derivatives")
control_frame_conv = ttk.Frame(conv_tab)
control_frame_conv.pack(side="left", fill="y", padx=(0, 5))
plot_frame_conv = ttk.Frame(conv_tab)
plot_frame_conv.pack(side="right", fill="both", expand=True)
fig_conv, ax_conv = plt.subplots(figsize=(7, 5))
ax_conv.set_title("Convolution / Derivative Plot")
fig_conv.tight_layout()
canvas_conv = FigureCanvasTkAgg(fig_conv, master=plot_frame_conv)
canvas_conv.get_tk_widget().pack(fill="both", expand=True)
ttk.Button(control_frame_conv, text="Load Signal", command=load_signal).pack(fill=tk.X, padx=5, pady=5)
ttk.Separator(control_frame_conv, orient='horizontal').pack(fill='x', pady=10, padx=5)
ttk.Label(control_frame_conv, text="Plot Type:").pack()
plot_mode_conv = ttk.Combobox(control_frame_conv, values=["Discrete", "Continuous"], textvariable=plot_mode_var,
                              state="readonly")
plot_mode_conv.pack(pady=5)
ttk.Separator(control_frame_conv, orient='horizontal').pack(fill='x', pady=10, padx=5)
ttk.Label(control_frame_conv, text="Select Signal for Operations:").pack(pady=(5, 0))
combo_conv = ttk.Combobox(control_frame_conv, values=[], state="readonly")
combo_conv.pack(pady=10)
ttk.Button(control_frame_conv, text="Display Signal",
           command=lambda: display_signal(combo_conv, ax_conv, canvas_conv)).pack(fill=tk.X, padx=5, pady=5)
ttk.Button(control_frame_conv, text="1st Derivative", command=first_deriv_signal).pack(fill=tk.X, padx=5, pady=5)
ttk.Button(control_frame_conv, text="2nd Derivative", command=second_deriv_signal).pack(fill=tk.X, padx=5, pady=5)
ttk.Button(control_frame_conv, text="Moving Average", command=moving_avg_signal).pack(fill=tk.X, padx=5, pady=5)
ttk.Separator(control_frame_conv, orient='horizontal').pack(fill='x', pady=10, padx=5)
ttk.Label(control_frame_conv, text="Signal 1:").pack()
combo1_conv = ttk.Combobox(control_frame_conv, values=[], state="readonly")
combo1_conv.pack(pady=10)
ttk.Label(control_frame_conv, text="Signal 2:").pack()
combo2_conv = ttk.Combobox(control_frame_conv, values=[], state="readonly")
combo2_conv.pack(pady=10)
ttk.Button(control_frame_conv, text="Convolute Signals", command=convolute_signals).pack(fill=tk.X, padx=5, pady=5)
ttk.Button(control_frame_conv, text="Correlate Signals", command=correlate_signals).pack(fill=tk.X, padx=5, pady=5)
ttk.Button(control_frame_conv, text="Calculate Time Delay", command=calc_delay).pack(fill=tk.X, padx=5, pady=5)
ttk.Separator(control_frame_conv, orient='horizontal').pack(fill='x', pady=10, padx=5)
ttk.Button(control_frame_conv, text="Save Signal", command=lambda: save_signal(ax_conv)).pack(side="bottom", fill=tk.X,
                                                                                              padx=10, pady=10)

# --- Frequency Domain Tab ---
freq_tab = ttk.Frame(notebook)
notebook.add(freq_tab, text="Frequency Domain")

control_frame_freq = ttk.Frame(freq_tab)
control_frame_freq.pack(side="left", fill="y", padx=(0, 5))

plot_frame_freq = ttk.Frame(freq_tab)
plot_frame_freq.pack(side="right", fill="both", expand=True)

fig_freq, (ax_freq_mag, ax_freq_phase) = plt.subplots(nrows=2, ncols=1, figsize=(7, 5))
fig_freq.tight_layout()

canvas_freq = FigureCanvasTkAgg(fig_freq, master=plot_frame_freq)
canvas_freq.get_tk_widget().pack(fill="both", expand=True)

ttk.Button(control_frame_freq, text="Load Signal", command=load_signal).pack(fill=tk.X, padx=5, pady=5)
ttk.Button(control_frame_freq, text="Load Frequency Signal", command=load_freq_signal).pack(fill=tk.X, padx=5, pady=5)
ttk.Separator(control_frame_freq, orient='horizontal').pack(fill='x', pady=10, padx=5)

ttk.Label(control_frame_freq, text="Select Signal:").pack(pady=(5, 0))
combo_freq = ttk.Combobox(control_frame_freq, values=[], state="readonly")
combo_freq.pack(pady=10)

ttk.Button(control_frame_freq, text="Display Time Signal",
           command=lambda: (ax_freq_phase.clear(), display_signal(combo_freq, ax_freq_mag, canvas_freq))).pack(fill=tk.X, padx=5, pady=5)

ttk.Button(control_frame_freq, text="Fourier Transform",
           command=fourier_transform).pack(fill=tk.X, padx=5, pady=5)


ttk.Separator(control_frame_freq, orient='horizontal').pack(fill='x', pady=10, padx=5)

ttk.Label(control_frame_freq, text="Select Frequency Signal :").pack()
combo1_freq = ttk.Combobox(control_frame_freq, values=[], state="readonly")
combo1_freq.pack(pady=10)

ttk.Button(control_frame_freq, text="Display Frequency Signal",
           command=lambda: display_freq_signal(combo1_freq, ax_freq_mag, ax_freq_phase, canvas_freq)).pack(fill=tk.X, padx=5, pady=5)

ttk.Button(control_frame_freq, text="Inverse Fourier Transform",
           command=inverse_fourier_transform).pack(fill=tk.X, padx=5, pady=5)

ttk.Separator(control_frame_freq, orient='horizontal').pack(fill='x', pady=10, padx=5)


##### Correlation tab  ####
corr_tab = ttk.Frame(notebook)
notebook.add(corr_tab, text="Correlation")
control_frame_corr = ttk.Frame(corr_tab)
control_frame_corr.pack(side="left", fill="y", padx=(0, 5))
plot_frame_corr = ttk.Frame(corr_tab)
plot_frame_corr.pack(side="right", fill="both", expand=True)
fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
fig_corr.tight_layout()
canvas_corr = FigureCanvasTkAgg(fig_corr, master=plot_frame_corr)
canvas_corr.get_tk_widget().pack(fill="both", expand=True)
ttk.Button(control_frame_corr, text="Load Signal", command=load_signal).pack(fill=tk.X, padx=5, pady=5)
ttk.Button(control_frame_corr, text="Load Samples", command=load_samples).pack(fill=tk.X, padx=5, pady=5)
ttk.Separator(control_frame_corr, orient='horizontal').pack(fill='x', pady=10, padx=5)
ttk.Label(control_frame_corr, text="Plot Type:").pack()
plot_mode_corr = ttk.Combobox(control_frame_corr, values=["Discrete", "Continuous"], textvariable=plot_mode_var, state="readonly")
plot_mode_corr.pack(pady=5)
ttk.Separator(control_frame_corr, orient='horizontal').pack(fill='x', pady=10, padx=5)
ttk.Label(control_frame_corr, text="Signal 1:").pack()
combo1_corr = ttk.Combobox(control_frame_corr, values=[], state="readonly")
combo1_corr.pack(pady=10)
ttk.Label(control_frame_corr, text="Signal 2:").pack()
combo2_corr = ttk.Combobox(control_frame_corr, values=[], state="readonly")
combo2_corr.pack(pady=10)
ttk.Button(control_frame_corr, text="Correlate Signals", command=correlate_signals).pack(fill=tk.X, padx=5, pady=5)
ttk.Button(control_frame_corr, text="Calculate Time Delay", command=calc_delay).pack(fill=tk.X, padx=5, pady=5)
isNormalized = tk.BooleanVar()
isNormalized.set(True)
ttk.Checkbutton(control_frame_corr, text="Normalized", variable=isNormalized).pack(pady=5)

ttk.Separator(control_frame_corr, orient='horizontal').pack(fill='x', pady=10, padx=5)
ttk.Label(control_frame_corr, text="Select Signal to Classify:").pack(pady=(5, 0))
combo_corr = ttk.Combobox(control_frame_corr, values=[], state="readonly")
combo_corr.pack(pady=10)
ttk.Button(control_frame_corr, text="Classify", command=classify_signal).pack(fill=tk.X, padx=5, pady=5)
ttk.Label(control_frame_corr, text="Available Classes:").pack(pady=(5, 0))
combo_classes = ttk.Combobox(control_frame_corr, values=[], state="readonly")
combo_classes.pack(pady=10)
ttk.Button(control_frame_corr, text="Load Class", command=load_class).pack(fill=tk.X, padx=5, pady=5)
ttk.Separator(control_frame_corr, orient='horizontal').pack(fill='x', pady=10, padx=5)
ttk.Button(control_frame_corr, text="Save Signal", command=lambda: save_signal(ax_corr)).pack(side="bottom", fill=tk.X,
                                                                                              padx=10, pady=10)
#############


# --- Filters Tab Setup ---
filters_tab = ttk.Frame(notebook)
notebook.add(filters_tab, text="Filters")

# --- Main Frames ---
control_frame_filters = ttk.Frame(filters_tab)
control_frame_filters.pack(side="left", fill="y", padx=(0, 5))

plot_frame_filters = ttk.Frame(filters_tab)
plot_frame_filters.pack(side="right", fill="both", expand=True)

# --- Plotting Area ---
fig_filters, ax_filters= plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
ax_filters.set_title("Plot 1")
fig_filters.tight_layout()

canvas_filters = FigureCanvasTkAgg(fig_filters, master=plot_frame_filters)
canvas_filters.get_tk_widget().pack(fill="both", expand=True)

# --- Control Panel Elements ---

# 1. Load Signal
ttk.Button(control_frame_filters, text="Load Signal", command=load_signal).pack(fill=tk.X, padx=5, pady=5)

ttk.Separator(control_frame_filters, orient='horizontal').pack(fill='x', pady=10, padx=5)

# 2. Plot Type
ttk.Label(control_frame_filters, text="Plot Type:").pack()
plot_mode_corr = ttk.Combobox(control_frame_filters, values=["Discrete", "Continuous"], textvariable=plot_mode_var, state="readonly")
plot_mode_corr.pack(pady=5)

ttk.Separator(control_frame_filters, orient='horizontal').pack(fill='x', pady=10, padx=5)

# 4. Filter Selection (Moved above parameters for better UX)
ttk.Label(control_frame_filters, text="Select Filter:").pack()
combo_filters = ttk.Combobox(control_frame_filters, values=["Low Pass", "High Pass", "Band Pass", "Band Stop"], state="readonly")
combo_filters.pack(pady=10)

# 5. Filter Parameters Frame
param_frame_filters = ttk.Frame(control_frame_filters)
param_frame_filters.pack(pady=10, padx=10)

# -- Sampling Frequency (Row 0)
ttk.Label(param_frame_filters, text="Sampling Frequency:").grid(row=0, column=0, sticky="w", pady=2)
sample_freq_entry = ttk.Spinbox(param_frame_filters, from_=1, to=100000)
sample_freq_entry.grid(row=0, column=1, pady=2)

# -- Cutoff 1 (Row 1)
ttk.Label(param_frame_filters, text="Cut-off Frequency (1):").grid(row=1, column=0, sticky="w", pady=2)
cuttOff_freq1_entry = ttk.Spinbox(param_frame_filters, from_=0, to=100000)
cuttOff_freq1_entry.grid(row=1, column=1, pady=2)

# -- Cutoff 2 (Row 2) - Fixed Row Index
ttk.Label(param_frame_filters, text="Cut-off Frequency (2):").grid(row=2, column=0, sticky="w", pady=2)
cuttOff_freq2_entry = ttk.Spinbox(param_frame_filters, from_=0, to=100000)
cuttOff_freq2_entry.grid(row=2, column=1, pady=2)

# -- Stop Attenuation (Row 3) - Fixed Row Index
ttk.Label(param_frame_filters, text="Stop Attenuation:").grid(row=3, column=0, sticky="w", pady=2)
ann_entry = ttk.Spinbox(param_frame_filters, from_=0, to=100000)
ann_entry.grid(row=3, column=1, pady=2)

# -- Transition Band (Row 4) - Fixed Row Index
ttk.Label(param_frame_filters, text="Transition Band:").grid(row=4, column=0, sticky="w", pady=2)
tb_entry = ttk.Spinbox(param_frame_filters, from_=0, to=100000)
tb_entry.grid(row=4, column=1, pady=2)

ttk.Separator(control_frame_filters, orient='horizontal').pack(fill='x', pady=10, padx=5)

ttk.Button(control_frame_filters, text=" Generate Filter", command=generate_filter).pack(fill=tk.X, padx=5, pady=5)


# . Select Signal
ttk.Label(control_frame_filters, text="Signal to filter:").pack()
combo_sigfilter = ttk.Combobox(control_frame_filters, values=[], state="readonly")
combo_sigfilter.pack(pady=10)

ttk.Separator(control_frame_filters, orient='horizontal').pack(fill='x', pady=10, padx=5)

ttk.Button(control_frame_filters, text="Apply Filter", command=apply_filter).pack(fill=tk.X, padx=5, pady=5)




############




combo_boxes = [combo, combo1, combo2, combo_conv, combo1_conv, combo2_conv, combo_freq, combo_corr, combo1_corr, combo2_corr,combo_sigfilter]
freq_combo_boxes = [combo1_freq]
classes = {}

# Default Settings
# load_signal("DSP_tasks/task6/Point1 Correlation/Corr_input signal1.txt")
# load_signal("DSP_tasks/task6/Point1 Correlation/Corr_input signal2.txt")
# load_signal("DSP_tasks/task6/Point2 Time analysis/TD_input signal1.txt")
# load_signal("DSP_tasks/task6/Point2 Time analysis/TD_input signal2.txt")

# load_samples("DSP_tasks/task6/point3 Files/Test Signals/Test1.txt")
# load_samples("DSP_tasks/task6/point3 Files/Test Signals/Test2.txt")

load_class("DSP_tasks/task6/point3 Files/Class 1")
load_class("DSP_tasks/task6/point3 Files/Class 2")

sample_freq_entry.insert(0, "8000")
cuttOff_freq1_entry.insert(0, "1500")
cuttOff_freq2_entry.insert(0, "200")
ann_entry.insert(0, "50")
tb_entry.insert(0, "500")

root.mainloop()