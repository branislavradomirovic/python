import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import zipfile
import time
import os
import sys
import gc
import psutil
import threading
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import cohen_kappa_score, accuracy_score
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
        self.best_params = None
        self.fitness_history = []
        self.best_agents_history = []

        self._setup_ui()
        self._setup_memory_monitor()
        sys.stdout = self.StdoutRedirector(self.console_text)

    def _setup_ui(self):
        self._create_control_panel()
        self._create_visualization()
        self._create_console()
        self._create_save_buttons()

    def _create_control_panel(self):
        control_frame = ttk.LabelFrame(self, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self._add_data_controls(control_frame)
        self._add_vns_params(control_frame)
        self._add_progress_bars(control_frame)

    def _add_data_controls(self, parent):
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Load ZIP", command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Start", command=self.start_optimization).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Stop", command=self.stop_optimization).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Show EEG", command=self.show_eeg_signals).pack(side=tk.LEFT, padx=5)

    def _add_vns_params(self, parent):
        params_frame = ttk.LabelFrame(parent, text="VNS Parameters")
        params_frame.pack(pady=10, fill=tk.X)
        self.vns_params = {}
        params = ["Pop Size", "Max Iter", "Theta", "Levy E", "FA Alpha", "FA Gamma", "FA Beta0"]
        defaults = [30, 100, 3, 0.1, 0.1, 1.0, 1.0]
        for i, param in enumerate(params):
            ttk.Label(params_frame, text=param).grid(row=i, column=0, sticky=tk.W)
            entry = ttk.Entry(params_frame)
            entry.insert(0, str(defaults[i]))
            entry.grid(row=i, column=1)
            self.vns_params[param] = entry

    def _add_progress_bars(self, parent):
        ttk.Label(parent, text="Optimization Progress").pack(pady=(10, 0))
        self.progress = ttk.Progressbar(parent, orient=tk.HORIZONTAL, mode='determinate', variable=self.progress_var)
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

    def _create_console(self):
        console_frame = ttk.LabelFrame(self, text="Console")
        console_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.console_text = tk.Text(console_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(console_frame, command=self.console_text.yview)
        self.console_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.console_text.pack(fill=tk.BOTH, expand=True)

    def _create_save_buttons(self):
        save_frame = ttk.Frame(self)
        save_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        ttk.Button(save_frame, text="Save Optimization Plot", command=self.save_optimization_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(save_frame, text="Save EEG Plot", command=self.save_eeg_plot).pack(side=tk.LEFT, padx=5)

    def _setup_memory_monitor(self):
        def monitor():
            while True:
                mem = psutil.virtual_memory()
                self.mem_label.config(text=f"Memory: {mem.percent}%")
                if mem.percent > 85 and not self.stop_flag:
                    self.stop_optimization()
                    messagebox.showwarning("Memory", "High memory usage! Stopped optimization.")
                time.sleep(1)
        threading.Thread(target=monitor, daemon=True).start()

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")])
        if file_path:
            self.zip_file_path = file_path
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
                    self.data_dict = {"EEG": txt_files}
                    print(f"Loaded {len(txt_files)} EEG files from ZIP")
                    messagebox.showinfo("Success", f"Loaded {len(txt_files)} EEG files from ZIP")
            except Exception as e:
                print(f"Error loading ZIP: {e}")
                messagebox.showerror("Error", str(e))

    def start_optimization(self):
        if not self.data_dict or self.zip_file_path is None:
            messagebox.showerror("Error", "Please load data first")
            return
        self.stop_flag = False
        self.fitness_history = []
        self.best_agents_history = []
        threading.Thread(target=self._run_optimization).start()

    def _run_optimization(self):
        best_fitness = float('-inf')
        best_agent = None

        pop_size = int(self.vns_params["Pop Size"].get())
        max_iter = int(self.vns_params["Max Iter"].get())
        theta = int(self.vns_params["Theta"].get())
        levy_e = float(self.vns_params["Levy E"].get())
        fa_alpha = float(self.vns_params["FA Alpha"].get())
        fa_gamma = float(self.vns_params["FA Gamma"].get())
        fa_beta0 = float(self.vns_params["FA Beta0"].get())

        print(f"Starting optimization with parameters:\nPop Size: {pop_size}, Max Iter: {max_iter}, Theta: {theta}, Levy E: {levy_e}, FA Alpha: {fa_alpha}, FA Gamma: {fa_gamma}, FA Beta0: {fa_beta0}")

        for i in range(max_iter):
            if self.stop_flag:
                break
            time.sleep(0.1)

            # Simulacija poboljšanja vrednosti (zameniti sa stvarnom optimizacijom)
            current_fitness = np.random.uniform(0.5, 1.0)
            self.fitness_history.append(current_fitness)

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_agent = np.random.rand(10)  # Simulirani parametri (zameniti sa stvarnim parametrima)
                self.best_agents_history.append((i, best_agent))  # Cuvanje najboljih agenata

            self.after(0, lambda v=(i + 1) / max_iter * 100: self.progress_var.set(v))
            self.after(0, self._update_fitness_plot, self.fitness_history)
            print(f"Iteration {i + 1}/{max_iter}, Current Fitness: {current_fitness:.4f}, Best Fitness: {best_fitness:.4f}")

        self.best_params = best_agent
        self._show_optimized_params(best_agent)
        print("Optimization finished successfully!")
        messagebox.showinfo("Optimization Complete", "Optimization finished successfully!")

    def _update_fitness_plot(self, fitness_history):
        self.ax_opt.clear()
        self.ax_opt.plot(fitness_history, label="Fitness over iterations", color="blue")
        # Dodavanje najboljih agenata na grafikon
        for iteration, agent in self.best_agents_history:
            self.ax_opt.plot(iteration, self.fitness_history[iteration], 'ro', markersize=5, label=f"Best Agent at Iteration {iteration}")

        # Uklanjanje duplikata u legendi
        handles, labels = self.ax_opt.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax_opt.legend(by_label.values(), by_label.keys())

        self.canvas.draw()

    def _show_optimized_params(self, best_agent):
        if best_agent is None:
            print("No optimized parameters to show.")
            messagebox.showinfo("Optimized Parameters", "No optimized parameters found.")
            return

        param_str = "\n".join([f"Param {i + 1}: {val:.4f}" for i, val in enumerate(best_agent)])
        print(f"Best Parameters:\n{param_str}")
        messagebox.showinfo("Optimized Parameters", f"Best Parameters:\n{param_str}")

    def show_eeg_signals(self):
        if not self.data_dict or self.zip_file_path is None:
            messagebox.showerror("Error", "Please load data first")
            return

        try:
            with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
                txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
                if not txt_files:
                    messagebox.showerror("Error", "No .txt files found in the loaded ZIP.")
                    return

                self.ax_eeg.clear()
                lines = []  # Lista za čuvanje linija grafika
                labels = []  # Lista za čuvanje labela

                for file_name in txt_files[:10]:  # Prikaz prvih 10 fajlova
                    with zip_ref.open(file_name) as file:
                        df = pd.read_csv(file, sep='\t', header=None)
                        for col in df.columns:
                            line, = self.ax_eeg.plot(df[col], label=f"{file_name} - Channel {col}")
                            lines.append(line)
                            labels.append(f"{file_name} - Channel {col}")

                # Postavljanje legende ispod grafikona
                self.ax_eeg.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                                    fancybox=True, shadow=True, ncol=5)  # Podešavanje broja kolona legende

                self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading EEG data: {e}")
            print(f"Error loading EEG data: {e}")

    def stop_optimization(self):
        self.stop_flag = True
        print("Optimization stopped by user")

    def on_close(self):
        self.stop_optimization()
        self.destroy()

    class StdoutRedirector:
        def __init__(self, text_widget):
            self.text_space = text_widget

        def write(self, string):
            self.text_space.insert('end', string)
            self.text_space.see('end')
            self.text_space.update_idletasks()

        def flush(self):
            pass

    def save_optimization_plot(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            self.figure.axes[0].figure.savefig(file_path)
            messagebox.showinfo("Save", "Optimization plot saved successfully!")

    def save_eeg_plot(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            self.figure.axes[1].figure.savefig(file_path)
            messagebox.showinfo("Save", "EEG plot saved successfully!")

if __name__ == "__main__":
    app = NeuroOptimizerApp()
    app.mainloop()