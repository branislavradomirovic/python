import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
import time
from tqdm import tqdm

def process_eeg_data(zip_file_path):
    """Obrada EEG podataka iz ZIP fajla sa gauge metrom."""
    data_frames = []
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_list = [f for f in zip_ref.namelist() if f.endswith('.txt')]
            total_files = len(file_list)
            start_time = time.time()

            for i, file_path in enumerate(tqdm(file_list, desc="Obrada fajlova")):
                try:
                    with zip_ref.open(file_path) as file:
                        data = np.loadtxt(file)
                        df = pd.DataFrame(data, columns=[f'kanal_{i+1}' for i in range(16)])

                        # Ekstrakcija informacija iz naziva fajla
                        file_name = os.path.basename(file_path)
                        if file_name:
                            phase = file_name[0]  # W, R, S1, S2

                            # Ekstrakcija indeksa signala
                            try:
                                signal_index = int(file_name.split('_')[-1].split('.')[0])
                            except ValueError:
                                print(f"Upozorenje: Neispravan format indeksa u nazivu fajla: {file_name}")
                                continue  # Preskakanje fajla sa neispravnim indeksom

                            df['faza'] = phase
                            df['indeks_signala'] = signal_index

                            # Dodajemo tip signala (fokalni/nefokalni)
                            if 'F' in file_name:
                                df['tip_signala'] = 'fokalni'
                            elif 'NF' in file_name:
                                df['tip_signala'] = 'nefokalni'
                            else:
                                df['tip_signala'] = 'nepoznato'

                            data_frames.append(df)
                        else:
                            print(f"Upozorenje: Naziv fajla je prazan: {file_path}")
                except Exception as e:
                    print(f"Greška pri obradi fajla {file_path}: {e}")

                # Procena vremena završetka
                elapsed_time = time.time() - start_time
                avg_time_per_file = elapsed_time / (i + 1)
                remaining_files = total_files - (i + 1)
                estimated_remaining_time = avg_time_per_file * remaining_files
                print(f"Procenjeno vreme završetka: {estimated_remaining_time:.2f} sekundi")

    except Exception as e:
        print(f"Greška pri otvaranju ZIP fajla: {e}")

    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        return combined_df
    else:
        return None

def visualize_eeg_signals(combined_df, num_samples=1000):
    """Vizualizacija fokalnih i nefokalnih signala."""

    if combined_df is None:
        print("Nema podataka za vizualizaciju.")
        return

    focal_signals = combined_df[combined_df['tip_signala'] == 'fokalni']
    nonfocal_signals = combined_df[combined_df['tip_signala'] == 'nefokalni']

    # Vizualizacija fokalnih signala
    if not focal_signals.empty:
        plt.figure(figsize=(15, 8))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.plot(focal_signals[f'kanal_{i+1}'][:num_samples])
            plt.title(f'Kanal {i+1} (fokalni)')
        plt.tight_layout()
        plt.show()
    else:
        print("Nema fokalnih signala u skupu podataka.")

    # Vizualizacija nefokalnih signala
    if not nonfocal_signals.empty:
        plt.figure(figsize=(15, 8))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.plot(nonfocal_signals[f'kanal_{i+1}'][:num_samples])
            plt.title(f'Kanal {i+1} (nefokalni)')
        plt.tight_layout()
        plt.show()
    else:
        print("Nema nefokalnih signala u skupu podataka.")

def select_zip_file():
    """Omogućava korisniku da odabere ZIP fajl."""
    root = tk.Tk()
    root.withdraw()  # Sakrivanje glavnog prozora
    file_path = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")])
    return file_path

if __name__ == "__main__":
    zip_file_path = select_zip_file()
    if zip_file_path:
        data = process_eeg_data(zip_file_path)
        if data is not None and not data.empty:  # Dodata provera da li je DataFrame prazan
            visualize_eeg_signals(data)
        else:
            print("Obrada podataka nije uspela ili nema podataka za vizualizaciju.")
    else:
        print("Nije odabran ZIP fajl.")