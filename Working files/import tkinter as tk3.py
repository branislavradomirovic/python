import tkinter as tk
from tkinter import filedialog, messagebox, Entry, Label, ttk, scrolledtext
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
import io
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.metrics import cohen_kappa_score, accuracy_score
import sys
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from typing import List, Dict, Tuple

class StdoutRedirector(object):
    def __init__(self, text_widget):
        self.text_space = text_widget

    def write(self, string):
        self.text_space.insert('end', string)
        self.text_space.see('end')

    def flush(self):
        pass

def fitness_function_lstm(agent: np.ndarray, experiment_data: List[pd.DataFrame], target_metric: str, 
                          progress_bar: ttk.Progressbar, master: tk.Tk, vns_ds_lstm: 'VNS_DS_LSTM') -> float:
    print("Pozivanje fitness funkcije")
    total_metric = 0.0
    count = 0
    total_segments = sum(len(data) // 158 for data in experiment_data)
    progress_bar["maximum"] = total_segments
    processed_segments = 0

    try:
        model = vns_ds_lstm.create_lstm_model()
        layer_weights_shapes = [layer.get_weights() for layer in model.layers]
        total_weights_size = sum(np.prod(w.shape) for layer_weights in layer_weights_shapes for w in layer_weights)
        if len(agent) != total_weights_size:
            print(f"Agent size mismatch: {len(agent)} vs {total_weights_size}")
        return float('-inf')

        if len(agent) != total_weights_size:
            print(f"Agent size mismatch: {len(agent)} vs {total_weights_size}")
            return float('-inf')

        for data in experiment_data:
            segments = [data[i:i+158] for i in range(0, len(data), 158)]
            for segment in segments:
                if len(segment) < 158:
                    continue
                
                segment = segment.values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(segment)

                X, y = [], []
                for i in range(60, len(scaled_data)):
                    X.append(scaled_data[i-60:i, 0])
                    y.append(scaled_data[i, 0])
                X, y = np.array(X), np.array(y)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                start = 0
                new_weights = []
                for layer_weights in layer_weights_shapes:
                    layer_weights_arr = []
                    for w_shape in layer_weights:
                        size = np.prod(w_shape.shape)
                        layer_weights_arr.append(agent[start:start+size].reshape(w_shape.shape))
                        start += size
                    new_weights.append(layer_weights_arr)
                    
                model.set_weights(new_weights)
                model.compile(optimizer='adam', loss='mean_squared_error')

                model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
                
                y_pred = model.predict(X_test)
                y_pred = np.round(y_pred).flatten()
                y_test = np.round(y_test).flatten()

                if target_metric == 'accuracy':
                    metric = accuracy_score(y_test, y_pred)
                elif target_metric == 'kappa':
                    metric = cohen_kappa_score(y_test, y_pred)
                
                total_metric += metric
                count += 1
                processed_segments += 1
                progress_bar["value"] = processed_segments
                master.update_idletasks()
                
                if vns_ds_lstm.gui.stop_flag:
                    return None

        return total_metric / count if count > 0 else 0.0
    except Exception as e:
        print(f"Error in fitness function: {str(e)}")
        return float('-inf')

class VNS_DS_LSTM_GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VNS-DS LSTM Optimizacija")
        self._setup_ui()
        self.data_dict: Dict[str, List[pd.DataFrame]] = {}
        self.stop_flag = False
        sys.stdout = StdoutRedirector(self.console_text)

    def _setup_ui(self):
        self._create_parameter_controls()
        self._create_data_controls()
        self._create_visualization_components()
        self._create_console()
        self._create_result_visualization()

    def _create_parameter_controls(self):
        params_frame = tk.LabelFrame(self, text="Parametri")
        params_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self._add_parameter_row(params_frame, "Dimenzija (LSTM):", "dim", 50, 10, 100)
        self._add_parameter_row(params_frame, "Veličina populacije:", "pop_size", 30, 10, 100)
        self._add_parameter_row(params_frame, "Maksimalno iteracija:", "max_iter", 100, 50, 500)
        self._add_parameter_row(params_frame, "Theta:", "theta", 3, 1, 10)
        self._add_parameter_row(params_frame, "Levy e:", "levy_e", 0.1, 0.01, 1.0)
        self._add_parameter_row(params_frame, "FA alpha:", "fa_alpha", 0.1, 0.01, 1.0)
        self._add_parameter_row(params_frame, "FA gamma:", "fa_gamma", 1.0, 0.1, 5.0)
        self._add_parameter_row(params_frame, "FA beta0:", "fa_beta0", 1.0, 0.1, 5.0)

    def _add_parameter_row(self, frame, label, param, default, min_val, max_val):
        row = frame.grid_size()[1]
        tk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
        slider = ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, length=300)
        slider.set(default)
        setattr(self, f"{param}_slider", slider)
        slider.grid(row=row, column=1)
        
        entry = Entry(frame)
        entry.insert(0, str(default))
        setattr(self, f"{param}_entry", entry)
        entry.grid(row=row, column=2)

    def _create_data_controls(self):
        data_frame = tk.LabelFrame(self, text="Učitavanje podataka")
        data_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.load_data_button = tk.Button(data_frame, text="Učitaj UPF ZIP", command=self.load_upf_zip)
        self.load_data_button.grid(row=0, column=0, columnspan=3, pady=5)

        self.progress_bar = ttk.Progressbar(data_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.grid(row=1, column=0, columnspan=3, pady=5, sticky='ew')
        
        self.progress_bar_label = Label(data_frame, text="Progres učitavanja")
        self.progress_bar_label.grid(row=2, column=0, columnspan=3)

        self.loaded_label = Label(data_frame, text="Učitano: 0")
        self.loaded_label.grid(row=3, column=0, sticky="w")
        self.remaining_label = Label(data_frame, text="Preostalo: 0")
        self.remaining_label.grid(row=3, column=1, sticky="w")
        self.time_label = Label(data_frame, text="Vreme: -")
        self.time_label.grid(row=4, column=0, columnspan=3, pady=5)

        controls_frame = tk.LabelFrame(self, text="Kontrole")
        controls_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        self.run_button = tk.Button(controls_frame, text="Pokreni VNS-DS", command=self.run_vns_ds)
        self.run_button.grid(row=0, column=0, columnspan=2, pady=5)

        self.stop_button = tk.Button(controls_frame, text="Zaustavi", command=self.stop_vns_ds, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=2, pady=5)

        self.vns_progress_bar = ttk.Progressbar(controls_frame, orient="horizontal", length=300, mode="determinate")
        self.vns_progress_bar.grid(row=1, column=0, columnspan=3, pady=5, sticky='ew')
        self.vns_progress_bar_label = Label(controls_frame, text="Progres VNS-DS")
        self.vns_progress_bar_label.grid(row=2, column=0, columnspan=3)

        self.results_frame = tk.LabelFrame(self, text="Rezultati")
        self.results_frame.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")

        self.result_text = tk.Text(self.results_frame, height=10, width=50)
        self.result_text.grid(row=0, column=0, columnspan=3, pady=5)

        self.lstm_train_progress_bar = ttk.Progressbar(self.results_frame, orient="horizontal", length=300, mode="determinate")
        self.lstm_train_progress_bar.grid(row=1, column=0, columnspan=3, pady=5, sticky='ew')
        self.lstm_train_progress_bar_label = Label(self.results_frame, text="Progres LSTM treninga")
        self.lstm_train_progress_bar_label.grid(row=2, column=0, columnspan=3)

        self.file_count_label = Label(self.results_frame, text="Broj fajlova: 0")
        self.file_count_label.grid(row=4, column=0, sticky="w")
        self.file_size_label = Label(self.results_frame, text="Veličina (MB): 0")
        self.file_size_label.grid(row=4, column=1, sticky="w")

        self.global_progress_bar = ttk.Progressbar(self.results_frame, orient="horizontal", length=300, mode="determinate")
        self.global_progress_bar.grid(row=5, column=0, columnspan=3, pady=5, sticky='ew')
        self.global_progress_bar_label = Label(self.results_frame, text="Progres izvršenja")
        self.global_progress_bar_label.grid(row=6, column=0, columnspan=3)

        self.start_time = 0
        self.elapsed_time_label = Label(self.results_frame, text="Protekao: 00:00:00")
        self.elapsed_time_label.grid(row=7, column=0, sticky="w")
        self.estimated_time_label = Label(self.results_frame, text="Preostalo: 00:00:00")
        self.estimated_time_label.grid(row=7, column=1, sticky="w")

        self.save_results_button = tk.Button(self.results_frame, text="Sačaj rezultate kao sliku", command=self.save_results_image)
        self.save_results_button.grid(row=8, column=0, columnspan=3, pady=5)

    def _create_visualization_components(self):
        eeg_frame = tk.LabelFrame(self, text="EEG signal")
        eeg_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        self.eeg_figure = plt.Figure(figsize=(10, 4), dpi=100)
        self.eeg_canvas = FigureCanvasTkAgg(self.eeg_figure, master=eeg_frame)
        self.eeg_canvas.draw()
        self.eeg_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_console(self):
        self.console_text = scrolledtext.ScrolledText(self.results_frame, height=10, width=80)
        self.console_text.grid(row=3, column=0, columnspan=3, pady=5)

    def _create_result_visualization(self):
        self.fitness_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.fitness_canvas = FigureCanvasTkAgg(self.fitness_figure, master=self.results_frame)
        self.fitness_canvas.get_tk_widget().grid(row=9, column=0, columnspan=3, pady=5)

    def load_upf_zip(self):
        file_path = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")])
        if file_path:
            self.zip_file_path = file_path
            try:
                self.data_dict = {}
                with zipfile.ZipFile(self.zip_file_path, 'r') as zip_file:
                    file_list = [f for f in zip_file.namelist() if f.endswith('.txt')]
                    total_files = len(file_list)
                    self.progress_bar["maximum"] = total_files
                    start_time = time.time()
                    loaded_files = 0
                    total_size = 0

                    for folder_name in sorted(zip_file.namelist()):
                        if folder_name.endswith('/'):
                            continue
                        parts = folder_name.split('/')
                        if len(parts) >= 1 and parts[-1].endswith('.txt'):
                            parent_folder = parts[-2]
                            print(f"Pronađen fajl: {folder_name}, folder: {parent_folder}")
                            if not parent_folder in self.data_dict:
                                self.data_dict[parent_folder] = []
                            with zip_file.open(folder_name) as file:
                                df = pd.read_csv(io.BytesIO(file.read()), sep='\t', header=None)
                                self.data_dict[parent_folder].append(df)
                                loaded_files += 1
                                total_size += file.seek(0, os.SEEK_END)
                                self.progress_bar["value"] = loaded_files
                                self.loaded_label["text"] = f"Učitano: {loaded_files}"
                                self.remaining_label["text"] = f"Preostalo: {total_files - loaded_files}"
                                elapsed_time = time.time() - start_time
                                if loaded_files > 0:
                                    avg_time_per_file = elapsed_time / loaded_files
                                    remaining_time = avg_time_per_file * (total_files - loaded_files)
                                    self.time_label["text"] = f"Vreme: {remaining_time:.2f}s"
                                self.update_idletasks()

                    print(f"Sadržaj data_dict: {self.data_dict}")
                    if messagebox.askokcancel("Uspeh", "Podaci učitani iz ZIP datoteke."):
                        self.file_count_label["text"] = f"Broj fajlova: {loaded_files}"
                        self.file_size_label["text"] = f"Veličina (MB): {total_size / (1024 * 1024):.2f}"

            except Exception as e:
                messagebox.showerror("Greška", f"Došlo je do greške pri učitavanju ZIP datoteke: {e}")
                print(f"Greška pri učitavanju ZIP datoteke: {e}")

    def run_vns_ds(self):
        threading.Thread(target=self.run_vns_ds_thread).start()

    def run_vns_ds_thread(self):
        print("Pokrenuta funkcija run_vns_ds")
        if not self.data_dict:
            messagebox.showerror("Greška", "Prvo učitajte podatke.")
            return

        try:
            pop_size = int(self.pop_size_entry.get())
            max_iter = int(self.max_iter_entry.get())
            theta = int(self.theta_entry.get())
            levy_e = float(self.levy_e_entry.get())
            fa_alpha = float(self.fa_alpha_entry.get())
            fa_gamma = float(self.fa_gamma_entry.get())
            fa_beta0 = float(self.fa_beta0_entry.get())
        except ValueError:
            messagebox.showerror("Greška", "Neispravan format parametara.")
            return

        vns_params = {'levy_e': levy_e}
        fa_params = {'alpha': fa_alpha, 'gamma': fa_gamma, 'beta0': fa_beta0}

        self.stop_flag = False
        self.stop_button.config(state=tk.NORMAL)
        self.run_button.config(state=tk.DISABLED)

        experiments = {
            'SS2_NF_txt': (['SS2_NF_txt'], 'accuracy'),
            'REM_F_txt': (['REM_F_txt'], 'kappa'),
            'Awake_NF_txt': (['Awake_NF_txt'], 'accuracy')
        }

        results = {}
        self.vns_progress_bar["maximum"] = max_iter

        total_experiments = len(experiments)
        self.global_progress_bar["maximum"] = total_experiments
        completed_experiments = 0

        self.start_time = time.time()
        fitness_history = {}

        for experiment, experiment_data in experiments.items():
            print(f"Obrada eksperimenta: {experiment}")
            folders, metric = experiment_data
            valid_folders = [f for f in folders if f in self.data_dict]
            if len(valid_folders) != len(folders):
                missing_folders = set(folders) - set(valid_folders)
                messagebox.showerror("Greška", f"Nedostaju podaci za foldere: {', '.join(missing_folders)}")
                return
            experiment_data = [pd.concat(self.data_dict[f]) for f in valid_folders]
            print(f"Broj DataFrame-ova za {experiment}: {len(experiment_data)}")

            for data in experiment_data:
                print("Ažuriranje EEG plot-a")
                self.update_eeg_plot(data, valid_folders[experiment_data.index(data)])

            print("Kreiranje LSTM modela")
            X = np.random.rand(60, 1)
            model = Sequential()
            model.add(Input(shape=(X.shape[0], 1)))
            model.add(LSTM(units=100, return_sequences=True))
            model.add(LSTM(units=50))
            model.add(Dense(units=1))
            dim = sum(np.prod(w.shape) for w in model.get_weights())
            print(f"Dimenzija LSTM modela: {dim}")

            try:
                print("Kreiranje VNS_DS_LSTM instance")
                vns_ds_lstm = VNS_DS_LSTM(dim, pop_size, max_iter, theta, lambda agent: fitness_function_lstm(agent, experiment_data, metric, self.lstm_train_progress_bar, self, vns_ds_lstm), vns_params, fa_params, self.vns_progress_bar, self)
                print("Pokretanje VNS_DS_LSTM algoritma")
                best_agent, best_metric = vns_ds_lstm.run()
                if self.stop_flag:
                    break
                if best_metric is not None:  # Dodata provera
                    results[experiment] = (best_agent, best_metric)
                    fitness_history[experiment] = vns_ds_lstm.fitness_history

                    completed_experiments += 1
                    self.global_progress_bar["value"] = completed_experiments

                    elapsed_time = time.time() - self.start_time
                    avg_time_per_experiment = elapsed_time / completed_experiments
                    remaining_time = avg_time_per_experiment * (total_experiments - completed_experiments)
                    self.elapsed_time_label["text"] = f"Protekao: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
                    self.estimated_time_label["text"] = f"Preostalo: {time.strftime('%H:%M:%S', time.gmtime(remaining_time))}"
                else:
                    print(f"Eksperiment {experiment} je zaustavljen.")
                break #Zaustavlja se i dalji rad, jer je eksperiment prekinut
             
            except Exception as e:
                print(f"Greška u eksperimentu {experiment}: {e}")
                messagebox.showerror("Greška", f"Greška u eksperimentu {experiment}: {e}")
                break

        self.lstm_train_progress_bar.grid_forget()

        self.result_text.delete(1.0, tk.END)
        for experiment, result in results.items():
            best_agent, best_metric = result
            metric_name = experiments[experiment][1]
            self.result_text.insert(tk.END, f"Eksperiment {experiment}: Najbolja {metric_name}: {best_metric}\n")

        try:
            for experiment, data in fitness_history.items():
                self.fitness_figure.clear()
                ax = self.fitness_figure.add_subplot(111)
                ax.plot(data)
                ax.set_title(f"Fitness tokom iteracija - {experiment}")
                ax.set_xlabel("Iteracija")
                ax.set_ylabel("Fitness")
                self.fitness_canvas.draw()
        except Exception as e:
            print(f"Greška pri vizualizaciji fitness-a: {e}")

        try:
            for experiment, result in results.items():
                best_agent, best_metric = result
                plt.figure()
                plt.plot(best_agent)
                plt.title(f"Najbolji LSTM parametri - {experiment}")
                plt.xlabel("Parametar")
                plt.ylabel("Vrednost")
                plt.show()
        except Exception as e:
            print(f"Greška pri vizualizaciji: {e}")

        try:
            for folder, dataframes in self.data_dict.items():
                for df in dataframes:
                    plt.figure()
                    for column in df.columns:
                        plt.plot(df[column], label=f"Kanal {column}")
                    plt.title(f"EEG signal - {folder}")
                    plt.xlabel("Vreme (uzorci)")
                    plt.ylabel("Amplituda")
                    plt.legend()
                    plt.show()
        except Exception as e:
            print(f"Greška pri vizualizaciji EEG signala: {e}")

        self.stop_button.config(state=tk.DISABLED)
        self.run_button.config(state=tk.NORMAL)

    def stop_vns_ds(self):
        self.stop_flag = True
        self.stop_button.config(state=tk.DISABLED)
        self.run_button.config(state=tk.NORMAL)

    def update_eeg_plot(self, data, folder_name):
        self.eeg_figure.clear()
        ax = self.eeg_figure.add_subplot(111)
        for column in data.columns:
            ax.plot(data[column], label=f"Kanal {column}")
        ax.set_title(f"EEG signal - {folder_name}")
        ax.set_xlabel("Vreme (uzorci)")
        ax.set_ylabel("Amplituda")
        ax.legend()
        self.eeg_canvas.draw()

    def save_results_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png")
        if file_path:
            self.eeg_figure.savefig(file_path)

class VNS_DS_LSTM:
    def __init__(self, dim: int, pop_size: int, max_iter: int, theta: int,
                 fitness_func: callable, vns_params: dict, fa_params: dict,
                 progress_bar: ttk.Progressbar, gui: VNS_DS_LSTM_GUI):
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.theta = theta
        self.fitness_func = fitness_func
        self.vns_params = vns_params
        self.fa_params = fa_params
        self.gui = gui
        self.progress_bar = progress_bar

        self.population = self.chaotic_initialization()
        self.best_agent: np.ndarray = None
        self.best_fitness: float = -np.inf
        self.chaotic_sequence: np.ndarray = self._generate_chaotic_sequence()
        self.fitness_history = []

    def _generate_chaotic_sequence(self) -> np.ndarray:
        c = np.zeros(self.max_iter)
        c[0] = random.uniform(0, 1)
        for i in range(1, self.max_iter):
            c[i] = 4 * c[i-1] * (1 - c[i-1])
        return c

    def chaotic_initialization(self) -> np.ndarray:
        population = np.zeros((self.pop_size, self.dim))
        for i in range(self.pop_size):
            c = random.uniform(0, 1)
            for j in range(self.dim):
                c = 4 * c * (1 - c)
                population[i,j] = c * 2 - 1  # Scale to [-1, 1]
        return population

    def chaotic_elite_learning(self, best_agent: np.ndarray, t: int) -> np.ndarray:
        c = self.chaotic_sequence[t]
        perturbation = np.random.rand(self.dim) * (c - 1)
        return best_agent + perturbation

    def levy_flight(self, agent: np.ndarray, t: int) -> np.ndarray:
        beta = 1.5  # Levy index
        sigma = (math.gamma(1+beta) * math.sin(math.pi*beta/2) / 
                 (math.gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
        
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v)**(1/beta))

        # Dynamic scaling factor from equation 11
        a = 1 + math.cos(math.pi * t / self.max_iter)
        e = a / 2
        
        return agent + e * self.vns_params['levy_e'] * step

    def firefly_move(self, agent: np.ndarray) -> np.ndarray:
        other = self.population[np.random.randint(self.pop_size)]
        r = np.linalg.norm(agent - other)
        beta = self.fa_params['beta0'] * math.exp(-self.fa_params['gamma'] * r**2)
        alpha = self.fa_params['alpha']
        return agent + beta * (other - agent) + alpha * (np.random.rand(self.dim) - 0.5)

    def vns_local_search(self, agent: np.ndarray) -> np.ndarray:
        return agent + np.random.uniform(-0.1, 0.1, self.dim)

    def run(self) -> Tuple[np.ndarray, float]:
        self.progress_bar["maximum"] = self.max_iter
        
        for t in range(self.max_iter):
            # Evaluate population
            fitness = np.array([self.fitness_func(agent) for agent in self.population])
            best_idx = np.argmax(fitness)
            
            if fitness[best_idx] > self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_agent = self.population[best_idx].copy()
            
            # Chaotic elite learning
            self.best_agent = self.chaotic_elite_learning(self.best_agent, t)
            
            # Update population
            for i in range(self.pop_size):
                if t < self.theta and np.random.rand() < 0.5:
                    # Firefly movement
                    self.population[i] = self.firefly_move(self.population[i])
                else:
                    # VNS local search
                    self.population[i] = self.vns_local_search(self.population[i])
                
                # Apply Levy flight
                self.population[i] = self.levy_flight(self.population[i], t)
            
            # Clipping to valid range
            self.population = np.clip(self.population, -1, 1)
            
            self.progress_bar["value"] = t + 1
            self.gui.update()
            
            print(f"Iteration {t+1}/{self.max_iter}, Best Fitness: {self.best_fitness:.4f}")
            self.fitness_history.append(self.best_fitness)
            
            if self.gui.stop_flag:
                break

        return self.best_agent, self.best_fitness

    def create_lstm_model(self):
        X = np.random.rand(60, 1)
        model = Sequential()
        model.add(Input(shape=(X.shape[0], 1)))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        return model

if __name__ == "__main__":
    root = VNS_DS_LSTM_GUI()
    root.mainloop()