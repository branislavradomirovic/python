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

class StdoutRedirector(object):
    def __init__(self, text_widget):
        self.text_space = text_widget

    def write(self, string):
        self.text_space.insert('end', string)
        self.text_space.see('end')

    def flush(self):
        pass

def fitness_function_lstm(agent, experiment_data, target_metric, progress_bar, master, vns_ds_lstm):
    print("Ulazak u fitness funkciju")
    total_metric = 0
    count = 0
    total_segments = sum(len(data) // 158 for data in experiment_data)
    progress_bar["maximum"] = total_segments
    processed_segments = 0

    for i, data in enumerate(experiment_data):
        print(f"Tip experiment_data[{i}]: {type(data)}")
        if isinstance(data, pd.DataFrame):
            print(f"Oblik experiment_data[{i}]: {data.shape}")
        else:
            print(f"experiment_data[{i}] nije pandas.DataFrame")

    try:
        for data in experiment_data:
            print(f"Obrada podataka: {data.shape}")
            segments = [data[i:i + 158] for i in range(0, len(data), 158)]
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

                layer_weights_shapes = [layer.get_weights() for layer in vns_ds_lstm.model.layers]
                layer_weights_sizes = [sum(np.prod(w.shape) for w in weights) for weights in layer_weights_shapes]
                total_weights_size = sum(layer_weights_sizes)

                if len(agent) != total_weights_size:
                    print(f"Agent ima {len(agent)} težina, a model očekuje {total_weights_size}.")
                    return float('-inf')

                start = 0
                for i, layer in enumerate(vns_ds_lstm.model.layers):
                    layer_weights = []
                    for weights_shape in layer_weights_shapes[i]:
                        size = np.prod(weights_shape.shape)
                        layer_weights.append(np.array(agent[start:start + size]).reshape(weights_shape.shape))
                        start += size
                    layer.set_weights(layer_weights)

                vns_ds_lstm.model.compile(optimizer='adam', loss='mean_squared_error')

                print(f"X_train oblik: {X_train.shape}")
                print(f"y_train oblik: {y_train.shape}")
                print(f"X_train tip: {X_train.dtype}")
                print(f"y_train tip: {y_train.dtype}")

                vns_ds_lstm.model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, callbacks=[vns_ds_lstm.gui])
                print("Zavrsen fit")
                y_pred = vns_ds_lstm.model.predict(X_test)
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
        print("Izlazak iz fitness funkcije")
        return total_metric / count if count > 0 else 0
    except AttributeError as e:
        print(f"Greška u fitness funkciji: {e}")
        return float('-inf')

class VNS_DS_LSTM_GUI(tf.keras.callbacks.Callback):
    def __init__(self, master):
        super().__init__()
        self.master = master
        master.title("VNS-DS LSTM Optimizacija")

        params_frame = tk.LabelFrame(master, text="Parametri")
        params_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        tk.Label(params_frame, text="Dimenzija (LSTM parametri):").grid(row=0, column=0, sticky="w")
        self.dim_slider = tk.Scale(params_frame, from_=10, to=100, orient=tk.HORIZONTAL, length=300)
        self.dim_slider.set(50)
        self.dim_slider.grid(row=0, column=1)
        self.dim_entry = Entry(params_frame)
        self.dim_entry.grid(row=0, column=2)
        self.dim_entry.insert(0, "50")

        tk.Label(params_frame, text="Veličina populacije:").grid(row=1, column=0, sticky="w")
        self.pop_size_slider = tk.Scale(params_frame, from_=10, to=100, orient=tk.HORIZONTAL, length=300)
        self.pop_size_slider.set(30)
        self.pop_size_slider.grid(row=1, column=1)
        self.pop_size_entry = Entry(params_frame)
        self.pop_size_entry.grid(row=1, column=2)
        self.pop_size_entry.insert(0, "30")

        tk.Label(params_frame, text="Maksimalno iteracija:").grid(row=2, column=0, sticky="w")
        self.max_iter_slider = tk.Scale(params_frame, from_=50, to=500, orient=tk.HORIZONTAL, length=300)
        self.max_iter_slider.set(100)
        self.max_iter_slider.grid(row=2, column=1)
        self.max_iter_entry = Entry(params_frame)
        self.max_iter_entry.grid(row=2, column=2)
        self.max_iter_entry.insert(0, "100")

        tk.Label(params_frame, text="Theta:").grid(row=3, column=0, sticky="w")
        self.theta_slider = tk.Scale(params_frame, from_=10, to=100, orient=tk.HORIZONTAL, length=300)
        self.theta_slider.set(30)
        self.theta_slider.grid(row=3, column=1)
        self.theta_entry = Entry(params_frame)
        self.theta_entry.grid(row=3, column=2)
        self.theta_entry.insert(0, "30")

        tk.Label(params_frame, text="Levy e:").grid(row=4, column=0, sticky="w")
        self.levy_e_slider = ttk.Scale(params_frame, from_=0.01, to=1.0, orient=tk.HORIZONTAL, length=300)
        self.levy_e_slider.set(0.1)
        self.levy_e_slider.grid(row=4, column=1)
        self.levy_e_entry = Entry(params_frame)
        self.levy_e_entry.grid(row=4, column=2)
        self.levy_e_entry.insert(0, "0.1")

        tk.Label(params_frame, text="FA alpha:").grid(row=5, column=0, sticky="w")
        self.fa_alpha_slider = ttk.Scale(params_frame, from_=0.01, to=1.0, orient=tk.HORIZONTAL, length=300)
        self.fa_alpha_slider.set(0.1)
        self.fa_alpha_slider.grid(row=5, column=1)
        self.fa_alpha_entry = Entry(params_frame)
        self.fa_alpha_entry.grid(row=5, column=2)
        self.fa_alpha_entry.insert(0, "0.1")

        tk.Label(params_frame, text="FA gamma:").grid(row=6, column=0, sticky="w")
        self.fa_gamma_slider = ttk.Scale(params_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, length=300)
        self.fa_gamma_slider.set(1.0)
        self.fa_gamma_slider.grid(row=6, column=1)
        self.fa_gamma_entry = Entry(params_frame)
        self.fa_gamma_entry.grid(row=6, column=2)
        self.fa_gamma_entry.insert(0, "1.0")

        tk.Label(params_frame, text="FA beta0:").grid(row=7, column=0, sticky="w")
        self.fa_beta0_slider = ttk.Scale(params_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, length=300)
        self.fa_beta0_slider.set(1.0)
        self.fa_beta0_slider.grid(row=7, column=1)
        self.fa_beta0_entry = Entry(params_frame)
        self.fa_beta0_entry.grid(row=7, column=2)
        self.fa_beta0_entry.insert(0, "1.0")

        data_frame = tk.LabelFrame(master, text="Učitavanje podataka")
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

        controls_frame = tk.LabelFrame(master, text="Kontrole")
        controls_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        self.run_button = tk.Button(controls_frame, text="Pokreni VNS-DS", command=self.run_vns_ds)
        self.run_button.grid(row=0, column=0, columnspan=2, pady=5)

        self.stop_button = tk.Button(controls_frame, text="Zaustavi", command=self.stop_vns_ds, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=2, pady=5)

        self.vns_progress_bar = ttk.Progressbar(controls_frame, orient="horizontal", length=300, mode="determinate")
        self.vns_progress_bar.grid(row=1, column=0, columnspan=3, pady=5, sticky='ew')
        self.vns_progress_bar_label = Label(controls_frame, text="Progres VNS-DS")
        self.vns_progress_bar_label.grid(row=2, column=0, columnspan=3)

        results_frame = tk.LabelFrame(master, text="Rezultati")
        results_frame.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")

        self.result_text = tk.Text(results_frame, height=10, width=50)
        self.result_text.grid(row=0, column=0, columnspan=3, pady=5)

        self.data_dict = {}
        self.data_loaded = False
        self.lstm_train_progress_bar = ttk.Progressbar(results_frame, orient="horizontal", length=300, mode="determinate")
        self.lstm_train_progress_bar.grid(row=1, column=0, columnspan=3, pady=5, sticky='ew')
        self.lstm_train_progress_bar_label = Label(results_frame, text="Progres LSTM treninga")
        self.lstm_train_progress_bar_label.grid(row=2, column=0, columnspan=3)

        self.stop_flag = False

        self.console_text = scrolledtext.ScrolledText(results_frame, height=10, width=80)
        self.console_text.grid(row=3, column=0, columnspan=3, pady=5)
        sys.stdout = StdoutRedirector(self.console_text)

        self.file_count_label = Label(results_frame, text="Broj fajlova: 0")
        self.file_count_label.grid(row=4, column=0, sticky="w")
        self.file_size_label = Label(results_frame, text="Veličina (MB): 0")
        self.file_size_label.grid(row=4, column=1, sticky="w")

        self.global_progress_bar = ttk.Progressbar(results_frame, orient="horizontal", length=300, mode="determinate")
        self.global_progress_bar.grid(row=5, column=0, columnspan=3, pady=5, sticky='ew')
        self.global_progress_bar_label = Label(results_frame, text="Progres izvršenja")
        self.global_progress_bar_label.grid(row=6, column=0, columnspan=3)

        self.start_time = 0
        self.elapsed_time_label = Label(results_frame, text="Protekao: 00:00:00")
        self.elapsed_time_label.grid(row=7, column=0, sticky="w")
        self.estimated_time_label = Label(results_frame, text="Preostalo: 00:00:00")
        self.estimated_time_label.grid(row=7, column=1, sticky="w")

        self.save_results_button = tk.Button(results_frame, text="Sačaj rezultate kao sliku", command=self.save_results_image)
        self.save_results_button.grid(row=8, column=0, columnspan=3, pady=5)

        eeg_frame = tk.LabelFrame(master,text="EEG signal")
        eeg_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        self.eeg_figure = plt.Figure(figsize=(10, 4), dpi=100)
        self.eeg_canvas = FigureCanvasTkAgg(self.eeg_figure, master=eeg_frame)
        self.eeg_canvas.draw()
        self.eeg_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
                                self.master.update_idletasks()

                    print(f"Sadržaj data_dict: {self.data_dict}")
                    if messagebox.askokcancel("Uspeh", "Podaci učitani iz ZIP datoteke."):
                        self.data_loaded = True
                        self.file_count_label["text"] = f"Broj fajlova: {loaded_files}"
                        self.file_size_label["text"] = f"Veličina (MB): {total_size / (1024 * 1024):.2f}"

            except Exception as e:
                messagebox.showerror("Greška", f"Došlo je do greške pri učitavanju ZIP datoteke: {e}")
                print(f"Greška pri učitavanju ZIP datoteke: {e}")

    def run_vns_ds(self):
        threading.Thread(target=self.run_vns_ds_thread).start()

    def run_vns_ds_thread(self):
        print("Pokrenuta funkcija run_vns_ds")
        if not self.data_loaded:
            messagebox.showerror("Greška", "Prvo učitajte podatke.")
            return
        print("Podaci su ucitani")

        try:
            pop_size = int(self.pop_size_entry.get())
            max_iter = int(self.max_iter_entry.get())
            theta = int(self.theta_entry.get())
            levy_e = float(self.levy_e_entry.get())
            fa_alpha = float(self.fa_alpha_entry.get())
            fa_gamma = float(self.fa_gamma_entry.get())
            fa_beta0 = float(self.fa_beta0_entry.get())
        except ValueError:
            print("Parametri su u redu")
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

        for experiment, experiment_data in experiments.items():
            print(f"Obrada eksperimenta: {experiment}")
            folders, metric = experiment_data
            valid_folders = [f for f in folders if f in self.data_dict]
            if len(valid_folders) != len(folders):
                missing_folders = set(folders) - set(valid_folders)
                messagebox.showerror("Greška", f"Nedostaju podaci za foldere: {', '.join(missing_folders)}")
                return
            experiment_data = [pd.concat(self.data_dict[f]) for f in valid_folders]

            for data in experiment_data:
                self.update_eeg_plot(data, valid_folders[experiment_data.index(data)])

            X = np.random.rand(60, 1)
            model = Sequential()
            model.add(Input(shape=(X.shape[0], 1)))
            model.add(LSTM(units=100, return_sequences=True))
            model.add(LSTM(units=50))
            model.add(Dense(units=1))
            dim = model.count_params()

            try:
                vns_ds_lstm = VNS_DS_LSTM(dim, pop_size, max_iter, theta, lambda agent: fitness_function_lstm(agent, experiment_data, metric, self.lstm_train_progress_bar, self.master, vns_ds_lstm), vns_params, fa_params, self.vns_progress_bar, self.master, self)
                print(f"vns_ds_lstm u lambda funkciji: {vns_ds_lstm}")
                best_agent, best_metric = vns_ds_lstm.run()
                print(f"Eksperiment {experiment} završen")
                if self.stop_flag:
                    break
                results[experiment] = (best_agent, best_metric)

                completed_experiments += 1
                self.global_progress_bar["value"] = completed_experiments

                elapsed_time = time.time() - self.start_time
                avg_time_per_experiment = elapsed_time / completed_experiments
                remaining_time = avg_time_per_experiment * (total_experiments - completed_experiments)
                self.elapsed_time_label["text"] = f"Protekao: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
                self.estimated_time_label["text"] = f"Preostalo: {time.strftime('%H:%M:%S', time.gmtime(remaining_time))}"
            except Exception as e:
                print(f"Greška u eksperimentu {experiment}: {e}")
                messagebox.showerror("Greška", f"Greška u eksperimentu {experiment}: {e}")
                break

        self.lstm_train_progress_bar.grid_forget()
        self.result_text.delete(1.0, tk.END)
        for experiment, result in results.items():
            if len(result) == 2:
                best_agent, best_metric = result
                metric_name = experiments[experiment][1]
                self.result_text.insert(tk.END, f"Eksperiment {experiment}: Najbolja {metric_name}: {best_metric}\n")
            else:
                print(f"Upozorenje: Rezultat za eksperiment {experiment} nema 2 elementa.")

        try:
            for experiment, result in results.items():
                if len(result) == 2:
                    best_agent, best_metric = result
                    plt.figure()
                    plt.plot(best_agent)
                    plt.title(f"Najbolji LSTM parametri - {experiment}")
                    plt.xlabel("Parametar")
                    plt.ylabel("Vrednost")
                    plt.show()
                else:
                    print(f"Upozorenje: Rezultat za eksperiment {experiment} nema 2 elementa.")
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

    def on_epoch_end(self, epoch, logs=None):
        print(f"Trenutna epoha: {epoch + 1}")
        total_epochs = 10
        self.lstm_train_progress_bar["maximum"] = total_epochs
        self.lstm_train_progress_bar["value"] = epoch + 1
        self.master.update_idletasks()
        print(f"update_lstm_progress pozvan sa epoch={epoch}, logs={logs}")

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
    def __init__(self, dim, pop_size, max_iter, theta, fitness_func, vns_params, fa_params, vns_progress_bar, master, gui):
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.theta = theta
        self.fitness_func = fitness_func
        self.vns_params = vns_params
        self.fa_params = fa_params
        self.population = self.chaotic_initialization()
        self.best_agent = None
        self.best_fitness = float('-inf')
        self.vns_progress_bar = vns_progress_bar
        self.master = master
        self.gui = gui
        self.model = self.create_lstm_model()

    def create_lstm_model(self):
        X = np.random.rand(60, 1)
        model = Sequential()
        model.add(Input(shape=(X.shape[0], 1)))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        print(f"Kreiran model: {model}")
        return model

    def chaotic_initialization(self):
        population = np.zeros((self.pop_size, self.dim))
        for i in range(self.pop_size):
            c = random.uniform(0, 1)
            for j in range(self.dim):
                c = 4 * c * (1 - c)
                population[i, j] = random.uniform(-1, 1) + c * (random.uniform(-1, 1) - population[i, j])
        return population

    def calculate_fitness(self, agent):
        return self.fitness_func(agent)

    def update_best_agent(self):
        for agent in self.population:
            fitness = self.calculate_fitness(agent)
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_agent = agent.copy()

    def levy_flight(self, agent, best_agent, t):
        e = self.vns_params['levy_e']
        T = self.max_iter
        beta = 3/2
        if best_agent is None:
            return agent
        alpha = e * (best_agent - agent) * (t/T)**beta

        lambda_val = 1.5
        sigma_u = (math.gamma(1 + lambda_val) * math.sin(math.pi * lambda_val / 2)) / \
                  (math.gamma((1 + lambda_val) / 2) * lambda_val * (2**((lambda_val - 1) / 2)))
        sigma_v = 1
        u = np.random.normal(0, sigma_u** (1/lambda_val), self.dim)
        v = np.random.normal(0, sigma_v, self.dim)
        levy = u / abs(v)**(1/lambda_val)

        return agent + alpha * levy

    def firefly_move(self, agent, other_agent):
        alpha = self.fa_params['alpha']
        gamma = self.fa_params['gamma']
        beta0 = self.fa_params['beta0']
        r = np.linalg.norm(agent - other_agent)
        beta = beta0 * math.exp(-gamma * r**2)
        kappa = np.random.normal(0, 1, self.dim)
        return agent + beta * (other_agent - agent) + alpha * kappa

    def vns_search(self, agent):
        for i in range(self.dim):
            agent[i] += random.uniform(-0.1, 0.1)
        return agent

    def fa_search(self, agent):
        other_agent = self.population[random.randint(0, self.pop_size - 1)]
        return self.firefly_move(agent, other_agent)

    def run(self):
        self.vns_progress_bar["maximum"] = self.max_iter
        for t in range(self.max_iter):
            self.update_best_agent()
            for i in range(self.pop_size):
                if t > self.theta:
                    self.population[i] = self.vns_search(self.population[i])
                else:
                    r = random.random()
                    if r < 0.5:
                        self.population[i] = self.vns_search(self.population[i])
                    else:
                        self.population[i] = self.fa_search(self.population[i])
                self.population[i] = self.levy_flight(self.population[i], self.best_agent, t)
            self.vns_progress_bar["value"] = t + 1
            self.master.update_idletasks()
            print(f"Iteracija {t+1}, Najbolja fitnes vrednost: {self.best_fitness}")
        return self.best_agent, self.best_fitness

if __name__ == "__main__":
    root = tk.Tk()
    app = VNS_DS_LSTM_GUI(root)
    root.mainloop()