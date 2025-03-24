import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import zipfile
import time
import os
import sys
import threading
import psutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Konfiguracija TensorFlow-a za optimizaciju memorije
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.config.optimizer.set_jit(True)


class NeuroOptimizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VNS-DS LSTM Optimizer")
        self.geometry("1200x800")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.data_dict = {}
        self.zip_file_path = None
        self.stop_flag = False
        self.progress_var = tk.DoubleVar(value=0)
        self.file_progress_var = tk.DoubleVar(value=0)
        self.best_params = None

        self._setup_ui()
        self._setup_memory_monitor()
        sys.stdout = self.StdoutRedirector(self.console_text)

    def _setup_ui(self):
        self._create_control_panel()
        self._create_visualization()
        self._create_console()

    def _create_control_panel(self):
        control_frame = ttk.LabelFrame(self, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self._add_vns_parameters(control_frame)
        self._add_data_controls(control_frame)
        self._add_progress_bars(control_frame)

    def _add_vns_parameters(self, parent):
        params = [
            ("Population Size", "pop_size", 30, 10, 100),
            ("Max Iterations", "max_iter", 100, 50, 500),
            ("Theta", "theta", 3, 1, 10),
            ("Levy e", "levy_e", 0.1, 0.01, 1.0),
            ("FA Alpha", "fa_alpha", 0.1, 0.01, 1.0),
            ("FA Gamma", "fa_gamma", 1.0, 0.1, 5.0),
            ("FA Beta0", "fa_beta0", 1.0, 0.1, 5.0)
        ]

        for label, var, default, min_val, max_val in params:
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label, width=15).pack(side=tk.LEFT)
            setattr(self, f"{var}_var", tk.DoubleVar(value=default))
            ttk.Scale(frame, variable=getattr(self, f"{var}_var"), from_=min_val, to=max_val,
                      orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=5)
            ttk.Entry(frame, textvariable=getattr(self, f"{var}_var"), width=7).pack(side=tk.LEFT)

    def _add_data_controls(self, parent):
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Load ZIP", command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Start", command=self.start_optimization).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Stop", command=self.stop_optimization).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Show EEG", command=self.show_eeg_signals).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save EEG", command=self.save_eeg_plot).pack(side=tk.LEFT, padx=5)

    def _add_progress_bars(self, parent):
        ttk.Label(parent, text="File Load Progress").pack(pady=(10, 0))
        self.file_progress = ttk.Progressbar(parent, orient=tk.HORIZONTAL, mode='determinate',
                                             variable=self.file_progress_var)
        self.file_progress.pack(fill=tk.X, pady=5)

        ttk.Label(parent, text="Optimization Progress").pack(pady=(10, 0))
        self.progress = ttk.Progressbar(parent, orient=tk.HORIZONTAL, mode='determinate',
                                        variable=self.progress_var)
        self.progress.pack(fill=tk.X, pady=5)

        self.mem_label = ttk.Label(parent, text="Memory: 0%")
        self.mem_label.pack()

    def _create_visualization(self):
        viz_frame = ttk.Frame(self)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax_opt = self.figure.add_subplot(211)
        self.ax_opt.set_title("Optimization Progress")
        self.ax_opt.set_xlabel("Iteration")
        self.ax_opt.set_ylabel("Fitness")

        self.ax_eeg = self.figure.add_subplot(212)
        self.ax_eeg.set_title("EEG Signals")
        self.ax_eeg.set_xlabel("Time")
        self.ax_eeg.set_ylabel("Amplitude")

    def show_eeg_signals(self):
        df = pd.DataFrame(np.random.randn(100, 16))  # Simulirani EEG podaci

        self.ax_eeg.clear()
        for col in df.columns:
            self.ax_eeg.plot(df[col], label=f"Channel {col}")
        self.ax_eeg.legend()
        self.canvas.draw()
        print("EEG signals plotted successfully.")

    def save_eeg_plot(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("All Files", "*.*")])
        if file_path:
            self.figure.savefig(file_path)
            messagebox.showinfo("Success", "EEG plot saved successfully.")

    def start_optimization(self):
        self.stop_flag = False
        threading.Thread(target=self._run_optimization).start()

    def _run_optimization(self):
        best_fitness = float('-inf')
        best_agent = None
        fitness_history = []

        for i in range(100):
            if self.stop_flag:
                break
            time.sleep(0.1)

            current_fitness = np.random.uniform(0.5, 1.0)
            fitness_history.append(current_fitness)

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_agent = np.random.rand(10)

            self.after(0, lambda v=(i + 1): self.progress_var.set(v))
            self.after(0, self._update_fitness_plot, fitness_history)

        self.best_params = best_agent
        self.console_text.insert('end', f"\nBest Agent Parameters:\n{best_agent}\n")
        messagebox.showinfo("Optimization Complete", "Optimization finished successfully!")

    def stop_optimization(self):
        self.stop_flag = True
        print("Optimization stopped by user")

    def on_close(self):
        self.stop_optimization()
        self.destroy()


if __name__ == "__main__":
    app = NeuroOptimizerApp()
    app.mainloop()
