import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Parametri simulacije
fs = 250  # Frekvencija uzorkovanja (Hz)
duration = 5  # Ukupno trajanje signala (s)
t = np.linspace(0, duration, fs * duration)  # Vremenski vektor

# Kreiranje sintetičkog EEG signala: sinusoidni signal sa dominantnom frekvencijom alfa talasa (10 Hz) + dodatni šum
eeg_clean = 50 * np.sin(2 * np.pi * 10 * t)
noise = 10 * np.random.randn(len(t))
eeg_signal = eeg_clean + noise

# Definisanje trajanja epohe u sekundama
epoch_duration = 1  # 1 sekunda po epozi
epoch_samples = int(fs * epoch_duration)
num_epochs = len(t) // epoch_samples

# Generisanje boja za svaki segment
colors = ['blue', 'green', 'red', 'purple', 'orange']

# Kreiranje figure
plt.figure(figsize=(14, 8))
plt.plot(t, eeg_signal, color='lightgray', label='Originalni EEG signal')  # crni sirovi signal u pozadini

# Segmentacija signala i prikazivanje segmentovanih epoha sa različitim bojama
for i in range(num_epochs):
    start = i * epoch_samples
    end = start + epoch_samples
    epoch_t = t[start:end]
    epoch_signal = eeg_signal[start:end]
    plt.plot(epoch_t, epoch_signal, color=colors[i % len(colors)], linewidth=2, label=f'Epoha {i+1}')
    plt.axvline(x=t[end-1], color='k', linestyle='--', linewidth=0.8)  # vertikalna linija na kraju epohe

plt.xlabel('Vreme (s)')
plt.ylabel('Amplituda (µV)')
plt.title('Segmentacija EEG signala u epohe')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
