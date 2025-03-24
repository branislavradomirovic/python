import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import zipfile
import time
import sys
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
        self.population = None  # Populacija agenata

        self._setup_ui()
        self._setup_memory_monitor()
        sys.stdout = self.StdoutRedirector(self.console_text)

    # --- UI deo ---
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

    # --- Funkcije za optimizaciju ---
    def chaotic_initialization(self, pop_size, dim, c0=0.7):
        # Generiše populaciju X koristeći logističku mapu
        population = np.zeros((pop_size, dim))
        c = c0
        for i in range(pop_size):
            for j in range(dim):
                c = 4 * c * (1 - c)  # logistička mapa
                population[i, j] = c
        return population

    def evaluate_fitness(self, agent):
        # Funkcija koja ima globalni optimum kada su sve komponente 0.5
        # Optimalno rešenje: agent = [0.5, 0.5, ..., 0.5]
        return -np.sum((agent - 0.5)**2)

    def vns_search(self, agent):
        # Shaking i protokol poboljšanja: mala slučajna perturbacija
        perturbation = np.random.uniform(-0.1, 0.1, size=agent.shape)
        return agent + perturbation

    def fa_search(self, agent, best_agent, fa_params, dim):
        # Firefly mehanizam: agent se pomera ka najboljem uz dodatak slučajnosti
        beta0 = fa_params.get("FA Beta0", 1.0)
        gamma = fa_params.get("FA Gamma", 1.0)
        alpha = fa_params.get("FA Alpha", 0.1)
        distance = np.linalg.norm(agent - best_agent)
        attractiveness = beta0 * np.exp(-gamma * distance**2)
        return agent + attractiveness * (best_agent - agent) + alpha * (np.random.rand(dim) - 0.5)

    def start_optimization(self):
        if not self.data_dict or self.zip_file_path is None:
            messagebox.showerror("Error", "Please load data first")
            return
        self.stop_flag = False
        self.fitness_history = []
        self.best_agents_history = []
        threading.Thread(target=self._run_optimization).start()

    def _run_optimization(self):
        # Učitavanje parametara iz GUI-a
        pop_size = int(self.vns_params["Pop Size"].get())
        max_iter = int(self.vns_params["Max Iter"].get())
        theta = int(self.vns_params["Theta"].get())
        levy_e = float(self.vns_params["Levy E"].get())  # Trenutno se ne koristi, ali je rezervisan
        fa_alpha = float(self.vns_params["FA Alpha"].get())
        fa_gamma = float(self.vns_params["FA Gamma"].get())
        fa_beta0 = float(self.vns_params["FA Beta0"].get())
        dim = 10  # Dimenzija rešenja

        fa_params = {"FA Alpha": fa_alpha, "FA Gamma": fa_gamma, "FA Beta0": fa_beta0}

        print(f"Starting optimization with parameters:\nPop Size: {pop_size}, Max Iter: {max_iter}, Theta: {theta}, Levy E: {levy_e}, FA Alpha: {fa_alpha}, FA Gamma: {fa_gamma}, FA Beta0: {fa_beta0}")

        # 1: Generiši populaciju X putem haotične inicijalizacije
        self.population = self.chaotic_initialization(pop_size, dim, c0=0.7)
        chaotic_ES = 0.7  # početna haotična vrednost

        best_fitness = -np.inf
        best_agent = None

        # 3: Glavna petlja dok (t < T) radi
        for t in range(max_iter):
            if self.stop_flag:
                break

            # 4: Odredi fitnes svih agenata i sačuvaj najboljeg
            fitnesses = np.array([self.evaluate_fitness(agent) for agent in self.population])
            iteration_best_index = np.argmax(fitnesses)
            iteration_best_fitness = fitnesses[iteration_best_index]
            iteration_best_agent = self.population[iteration_best_index].copy()

            if iteration_best_fitness > best_fitness:
                best_fitness = iteration_best_fitness
                best_agent = iteration_best_agent.copy()
                self.best_agents_history.append((t, best_agent))

            self.fitness_history.append(best_fitness)

            # 5: Ponovo izračunaj ES pomoću logističke mape
            chaotic_ES = 4 * chaotic_ES * (1 - chaotic_ES)

            # 6: Za svakog agenta u populaciji:
            for i in range(pop_size):
                current_agent = self.population[i]
                if t > theta:
                    # 7: Ako je t > θ, koristi VNS mehanizam
                    new_agent = self.vns_search(current_agent)
                else:
                    # 9: Inače, generiši nasumičnu vrednost R
                    R = np.random.uniform(0, 1)
                    if R < 0.5:
                        # 11: Ako je R < 0.5, koristi VNS mehanizam
                        new_agent = self.vns_search(current_agent)
                    else:
                        # 14: Inače, koristi FA mehanizam
                        new_agent = self.fa_search(current_agent, best_agent, fa_params, dim)
                # Ažuriranje rešenja korišćenjem haotičnog elitnog učenja (jednačina 28)
                new_agent = new_agent + np.random.rand(dim) * (chaotic_ES - 1)
                self.population[i] = new_agent

            # 18: t = t + 1 (u petlji je implicitno)
            self.after(0, lambda v=(t + 1) / max_iter * 100: self.progress_var.set(v))
            self.after(0, self._update_fitness_plot, self.fitness_history)
            print(f"Iteration {t + 1}/{max_iter}, Best Fitness: {best_fitness:.4f}")
            time.sleep(0.1)

        # 20: Vrati agenta sa najboljim performansama
        self.best_params = best_agent
        self._show_optimized_params(best_agent)
        print("Optimization finished successfully!")
        messagebox.showinfo("Optimization Complete", "Optimization finished successfully!")

    def _update_fitness_plot(self, fitness_history):
        self.ax_opt.clear()
        self.ax_opt.plot(fitness_history, label="Best Fitness", color="blue")
        # Prikaz tačaka najboljih agenata
        for iteration, agent in self.best_agents_history:
            self.ax_opt.plot(iteration, fitness_history[iteration], 'ro', markersize=5, label=f"Best at {iteration}")
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
                lines = []
                labels = []
                for file_name in txt_files[:10]:
                    with zip_ref.open(file_name) as file:
                        df = pd.read_csv(file, sep='\t', header=None)
                        for col in df.columns:
                            line, = self.ax_eeg.plot(df[col], label=f"{file_name} - Channel {col}")
                            lines.append(line)
                            labels.append(f"{file_name} - Channel {col}")
                self.ax_eeg.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                                     fancybox=True, shadow=True, ncol=5)
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
