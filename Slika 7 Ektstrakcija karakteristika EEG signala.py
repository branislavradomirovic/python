import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Parametri simulacije
fs = 250  # Frekvencija uzorkovanja (Hz)
duration = 10  # Trajanje signala u sekundama
t = np.linspace(0, duration, fs * duration, endpoint=False)

# Kreiranje sintetičkog EEG signala:
# Simuliramo signal sa dominantnom alfa frekvencijom (10 Hz) i dodatnim šumom
alpha_wave = 50 * np.sin(2 * np.pi * 10 * t)
noise = 15 * np.random.randn(len(t))
eeg_signal = alpha_wave + noise

# Izračunavanje spektrograma
f, t_spec, Sxx = spectrogram(eeg_signal, fs=fs, window='hann', nperseg=fs//2, noverlap=fs//4)

# Konvertovanje snage u decibele
Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Dodajemo malu konstantu da izbegnemo log(0)

# Kreiranje grafikona
plt.figure(figsize=(14, 8))
plt.pcolormesh(t_spec, f, Sxx_db, shading='gouraud', cmap='viridis')
plt.title('Spektrogram EEG signala sa označenim frekvencijskim pojasevima')
plt.ylabel('Frekvencija (Hz)')
plt.xlabel('Vreme (s)')
cbar = plt.colorbar()
cbar.set_label('Snaga (dB)')
plt.ylim([0, 40])  # Fokus na frekvencijski opseg do 40 Hz, gde se nalaze delta, theta, alfa i beta pojasevi
plt.tight_layout()
plt.show()
