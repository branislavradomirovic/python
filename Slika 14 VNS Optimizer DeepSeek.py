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
from scipy.special import gamma
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Конфигурација TensorFlow за GPU оптимизацију
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.config.optimizer.set_jit(True)

class VNSOptimizer:
    def __init__(self, params):
        self.pop_size = params['pop_size']
        self.max_iter = params['max_iter']
        self.theta = params['theta']
        self.levy_e = params['levy_e']
        self.fa_alpha = params['fa_alpha']
        self.fa_gamma = params['fa_gamma']
        self.fa_beta0 = params['fa_beta0']
        self.problem_dim = params['problem_dim']
        self.lstm_shape = params['lstm_shape']
        self.chaotic_values = self.generate_chaotic_sequence()
        self.scaler = MinMaxScaler()
        self.model = self.build_lstm_model()

    def generate_chaotic_sequence(self):
        seq = np.zeros(self.pop_size)
        c = np.random.rand()
        for i in range(self.pop_size):
            c = 4 * c * (1 - c)
            seq[i] = c
        return seq

    def levy_flight(self, size):
        tau = 1.5
        phi = (gamma(1 + tau) * np.sin(np.pi * tau / 2)) / \
              (gamma((1 + tau)/2) * tau * 2**((tau-1)/2)) ** (1/tau)
        mu = np.random.normal(0, phi, size)
        v = np.random.normal(0, 1, size)
        return 0.01 * mu / (np.abs(v)**(1/tau))

    def build_lstm_model(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=self.lstm_shape, return_sequences=True))
        model.add(LSTM(32))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def evaluate_fitness(self, X, data):
        X_train, X_test, y_train, y_test = data
        self.model.set_weights(X)
        _, accuracy = self.model.evaluate(X_train, y_train, verbose=0)
        return accuracy

    def chaotic_elite_learning(self, population):
        chaotic = np.random.rand(*population.shape) * self.chaotic_values[-1]
        return population * (1 + chaotic)

    def firefly_move(self, population, fitness):
        new_pop = np.zeros_like(population)
        for i in range(len(population)):
            for j in range(len(population)):
                if fitness[j] > fitness[i]:
                    r = np.linalg.norm(population[i] - population[j])
                    beta = self.fa_beta0 * np.exp(-self.fa_gamma * r**2)
                    new_pop[i] += beta * (population[j] - population[i]) + \
                                  self.fa_alpha * (np.random.rand() - 0.5)
        return new_pop

    def optimize(self, data):
        population = np.array([np.random.randn(*self.lstm_shape) for _ in range(self.pop_size)])
        best_fitness = -np.inf
        best_agent = None
        fitness_history = []

        for t in range(self.max_iter):
            population = self.chaotic_elite_learning(population)
            fitness = np.array([self.evaluate_fitness(agent, data) for agent in population])
            
            current_best = np.max(fitness)
            if current_best > best_fitness:
                best_fitness = current_best
                best_agent = population[np.argmax(fitness)]
            
            if t > self.theta:
                levy = self.levy_flight(population.shape)
                population += levy * self.levy_e
            else:
                if np.random.rand() < 0.5:
                    population = self.firefly_move(population, fitness)
                else:
                    population += self.levy_flight(population.shape) * self.levy_e
            
            fitness_history.append(best_fitness)
            
        return best_agent, fitness_history

class NeuroOptimizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VNS-DS LSTM Optimizer Pro")
        self.geometry("1400x900")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.optimizer = None
        self.data_dict = {}
        self.zip_file_path = None
        self.stop_flag = False
        self.progress_var = tk.DoubleVar(value=0)
        self.fitness_history = []
        self.best_agents_history = []
        
        self._setup_ui()
        self._setup_memory_monitor()
        sys.stdout = self.StdoutRedirector(self.console_text)

    def _setup_ui(self):
        control_frame = ttk.LabelFrame(self, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Data controls
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Load ZIP", command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Start", command=self.start_optimization).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Stop", command=self.stop_optimization).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Show EEG", command=self.show_eeg_signals).pack(side=tk.LEFT, padx=5)
        
        # VNS parameters
        params_frame = ttk.LabelFrame(control_frame, text="VNS Parameters")
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
        
        # Progress bars
        ttk.Label(control_frame, text="Optimization Progress").pack(pady=(10, 0))
        self.progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, 
                                      mode='determinate', variable=self.progress_var)
        self.progress.pack(fill=tk.X, pady=5)
        self.mem_label = ttk.Label(control_frame, text="Memory: 0%")
        self.mem_label.pack()
        
        # Visualization
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
        
        # Console
        console_frame = ttk.LabelFrame(self, text="Console")
        console_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.console_text = tk.Text(console_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(console_frame, command=self.console_text.yview)
        self.console_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.console_text.pack(fill=tk.BOTH, expand=True)
        
        # Save buttons
        save_frame = ttk.Frame(self)
        save_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        ttk.Button(save_frame, text="Save Optimization Plot", 
                 command=self.save_optimization_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(save_frame, text="Save EEG Plot", 
                 command=self.save_eeg_plot).pack(side=tk.LEFT, padx=5)

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
        try:
            X_train, X_test, y_train, y_test = self.prepare_data()
            data = (X_train, X_test, y_train, y_test)
            
            params = {
                'pop_size': int(self.vns_params["Pop Size"].get()),
                'max_iter': int(self.vns_params["Max Iter"].get()),
                'theta': int(self.vns_params["Theta"].get()),
                'levy_e': float(self.vns_params["Levy E"].get()),
                'fa_alpha': float(self.vns_params["FA Alpha"].get()),
                'fa_gamma': float(self.vns_params["FA Gamma"].get()),
                'fa_beta0': float(self.vns_params["FA Beta0"].get()),
                'problem_dim': (64, 64, 3),
                'lstm_shape': (X_train.shape[1], X_train.shape[2])
            }
            
            self.optimizer = VNSOptimizer(params)
            best_agent, fitness_history = self.optimizer.optimize(data)
            
            self.best_params = best_agent
            self.fitness_history = fitness_history
            self._show_optimized_params(best_agent)
            
            self.after(0, self._update_fitness_plot, fitness_history)
            messagebox.showinfo("Optimization Complete", "Optimization finished successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.stop_optimization()

    def prepare_data(self):
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
            data = []
            labels = []
            for file_name in txt_files:
                with zip_ref.open(file_name) as f:
                    df = pd.read_csv(f, sep='\t', header=None)
                    data.append(df.values)
                    labels.append(0 if 'normal' in file_name else 1)
            
            data = np.array(data)
            labels = np.array(labels)
            data = self.optimizer.scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
            
            return train_test_split(data, labels, test_size=0.2, random_state=42)

    def _update_fitness_plot(self, fitness_history):
        self.ax_opt.clear()
        self.ax_opt.plot(fitness_history, label="Fitness", color="blue")
        if self.best_agents_history:
            iterations = [iter for iter, _ in self.best_agents_history]
            fitness = [self.fitness_history[iter] for iter in iterations]
            self.ax_opt.scatter(iterations, fitness, color='red', label="Best Agents")
        self.ax_opt.legend()
        self.canvas.draw()

    def _show_optimized_params(self, best_agent):
        param_str = "\n".join([f"Param {i+1}: {val:.4f}" for i, val in enumerate(best_agent.flatten()[:10])])
        messagebox.showinfo("Optimized Parameters", f"Best Parameters (first 10):\n{param_str}")

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
                for file_name in txt_files[:3]:
                    with zip_ref.open(file_name) as file:
                        df = pd.read_csv(file, sep='\t', header=None)
                        for col in df.columns[:5]:
                            self.ax_eeg.plot(df[col], label=f"{file_name} Ch{col}")
                self.ax_eeg.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
                self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading EEG data: {e}")

    def stop_optimization(self):
        self.stop_flag = True
        print("Optimization stopped")

    def on_close(self):
        self.stop_optimization()
        self.destroy()

    def save_optimization_plot(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                filetypes=[("PNG files", "*.png")])
        if file_path:
            self.figure.savefig(file_path)
            messagebox.showinfo("Save", "Plot saved successfully!")

    def save_eeg_plot(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                filetypes=[("PNG files", "*.png")])
        if file_path:
            self.figure.savefig(file_path)
            messagebox.showinfo("Save", "EEG plot saved successfully!")

    class StdoutRedirector:
        def __init__(self, text_widget):
            self.text_space = text_widget

        def write(self, string):
            self.text_space.insert('end', string)
            self.text_space.see('end')
            self.text_space.update_idletasks()

        def flush(self):
            pass

if __name__ == "__main__":
    app = NeuroOptimizerApp()
    app.mainloop()