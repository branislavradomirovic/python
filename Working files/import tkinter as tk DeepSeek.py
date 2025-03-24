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
import random
import math

# Globalne optimizacije za TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.config.optimizer.set_jit(True)
tf.config.set_soft_device_placement(True)

class MemoryEfficientNeuroOptimizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VNS-DS LSTM Optimizer Pro")
        self.geometry("1200x800")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.data_chunks = []
        self.current_chunk = 0
        self.stop_flag = False
        self.progress_var = tk.DoubleVar(value=0)
        self.best_params = None
        self.fitness_history = []
        self.scaler = MinMaxScaler()
        self.figure = None
        
        self._setup_ui()
        self._setup_memory_monitor()
        sys.stdout = self.StdoutRedirector(self.console_text)

    def _setup_ui(self):
        self._create_control_panel()
        self._create_visualization()
        self._create_console()

    # Остали UI делови остају исти...

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("ZIP datoteke", "*.zip")])
        if file_path:
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
                    
                    progress = self._show_loading_progress(len(txt_files))
                    
                    # Читање података у мањим деловима
                    chunk_size = 5000  # Прилагодите према расположивој меморији
                    for i, file in enumerate(txt_files):
                        with zip_ref.open(file) as f:
                            # Инкрементално учитавање и процесирање
                            chunk_reader = pd.read_csv(f, sep='\t', header=None, 
                                                     chunksize=chunk_size,
                                                     dtype=np.float32)
                            for chunk in chunk_reader:
                                self.data_chunks.append(chunk.values)
                                
                        progress['value'] = i+1
                        progress['text'] = f"Učitano {i+1}/{len(txt_files)}"
                        self.update()
                    
                    messagebox.showinfo("Uspeh", f"Uspješno učitano {len(txt_files)} EEG fajlova")
                    
            except Exception as e:
                messagebox.showerror("Greška", str(e))

    def _get_next_batch(self):
        # Генератор за добијање података у мањим деловима
        while self.current_chunk < len(self.data_chunks):
            chunk = self.data_chunks[self.current_chunk]
            self.current_chunk += 1
            yield chunk
        self.current_chunk = 0  # Ресет за нову епоху

    def _create_lstm_model(self):
        # Управољен модел са мањим меморијским захтевима
        model = Sequential([
            LSTM(32, input_shape=(60, 1), return_sequences=False,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal'),
            Dense(1, activation='sigmoid')
        ])
        model.build(input_shape=(None, 60, 1))
        return model

    def _calculate_fitness(self, agent):
        try:
            model = self._create_lstm_model()
            ptr = 0
            new_weights = []
            
            # Оптимизовано постављање тежина
            for layer in model.layers:
                layer_weights = []
                for w in layer.get_weights():
                    size = np.prod(w.shape)
                    layer_weights.append(agent[ptr:ptr+size].reshape(w.shape).astype(np.float32)
                    ptr += size
                new_weights.extend(layer_weights)
            
            model.set_weights(new_weights)
            
            total_metric = 0
            count = 0
            batch_size = 32  # Смањите ако је потребно
            
            # Инкрементално тренирање на деловима података
            for chunk in self._get_next_batch():
                X, y = self._prepare_training_data(chunk)
                if len(X) == 0:
                    continue
                
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                             loss='mse',
                             run_eagerly=False)
                
                # Тренирање на делу података
                history = model.fit(X, y, batch_size=batch_size,
                                  epochs=1, verbose=0,
                                  validation_split=0.2)
                
                total_metric += history.history['val_loss'][0]
                count += 1
                
                # Раније ослобађање меморије
                del X, y, history
                tf.keras.backend.clear_session()
                gc.collect()
            
            return total_metric / count if count > 0 else 0.0
            
        except Exception as e:
            print(f"Грешка у fitness funkciји: {str(e)}")
            return 0.0

    def _prepare_training_data(self, chunk):
        scaled = self.scaler.fit_transform(chunk)
        
        X, y = [], []
        for i in range(60, len(scaled)):
            if i % 10 == 0:  # Узорак сваки 10ти податак
                X.append(scaled[i-60:i, 0])
                y.append(scaled[i, 0])
        
        if len(X) == 0:
            return [], []
        
        return np.array(X, dtype=np.float32).reshape(-1, 60, 1), np.array(y, dtype=np.float32)

    def show_eeg_signals(self):
        if not self.data_chunks:
            messagebox.showerror("Greška", "Nema EEG podataka za prikaz")
            return
        
        # Приказ само дела података
        sample = self.data_chunks[0][:500]  # Првих 500 узорака
        
        self.ax_eeg.clear()
        for col in range(min(3, sample.shape[1])):  # Прва 3 канала
            self.ax_eeg.plot(sample[:, col], label=f"Kanal {col}")
        self.ax_eeg.legend()
        self.canvas.draw()

    def _cleanup_memory(self):
        # Агресивно чишћење меморије
        tf.keras.backend.clear_session()
        gc.collect()
        if tf.executing_eagerly():
            tf.compat.v1.reset_default_graph()
        
        # Ослобађање листа података
        if len(self.data_chunks) > 10:  # Чувамо само последњих 10 делова
            self.data_chunks = self.data_chunks[-10:]
        
        # Принудно ослобађање GPU меморије
        if tf.config.list_physical_devices('GPU'):
            tf.tpu.experimental.initialize_tpu_system()
            tf.config.experimental_initialize_multi_host()
            tf.config.experimental_connect_to_cluster()
            tf.tpu.experimental.shutdown_tpu_system()

    # Остале методе остају исте...

class VNSOptimizer:
    def __init__(self, dim, pop_size, max_iter, theta, fitness_func, vns_params, fa_params):
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.theta = theta
        self.fitness_func = fitness_func
        self.vns_params = vns_params
        self.fa_params = fa_params
        self.best_agent = None
        self.best_fitness = -np.inf
        self.chaotic_sequence = self._generate_chaotic_sequence()
        self.population = self.chaotic_initialization()

    def chaotic_initialization(self):
        # Оптимизована иницијализација са мањим меморијским захтевима
        population = np.empty((self.pop_size, self.dim), dtype=np.float32)
        for i in range(self.pop_size):
            c = random.uniform(0, 1)
            for j in range(self.dim):
                c = 4 * c * (1 - c)
                population[i,j] = c * 2 - 1
        return population

    def optimize(self):
        for t in range(self.max_iter):
            fitness = np.zeros(self.pop_size, dtype=np.float32)
            for i in range(self.pop_size):
                fitness[i] = self.fitness_func(self.population[i])
                # Ослобађање меморије између јединки
                if i % 10 == 0:
                    gc.collect()
            
            best_idx = np.argmax(fitness)
            
            if fitness[best_idx] > self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_agent = self.population[best_idx].copy()
            
            self._update_population(t)
            
            yield t, self.best_fitness

    # Остале методе остају исте...

if __name__ == "__main__":
    app = MemoryEfficientNeuroOptimizer()
    app.mainloop()