import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Definisanje funkcije za kreiranje pip 
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs  # Nyquist frekvencija
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Funkcija za primenu filtera
def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Parametri simulacije
fs = 500         # Frekvencija uzorkovanja u Hz
t = np.linspace(0, 5, fs * 5)  # Vremenski vektor za 5 sekundi

# Kreiranje sintetičkog EEG signala: osnovni signal + 50 Hz šum
# Osnovni EEG signal: sinusoidni signal koji predstavlja alfa talase (10 Hz)
eeg_signal = 100 * np.sin(2 * np.pi * 10 * t)
# Šum: sinusoidna komponenta na 50 Hz koja simulira mrežni šum
noise = 50 * np.sin(2 * np.pi * 50 * t)
# Kombinovani signal: EEG signal sa šumom
raw_signal = eeg_signal + noise

# Primena Butterworth niskopropusnog filtera sa cutoff frekvencijom 40 Hz
filtered_signal = lowpass_filter(raw_signal, cutoff=40, fs=fs, order=4)

# Kreiranje grafikona
plt.figure(figsize=(14, 8))

# Gornji panel: Sirovi EEG signal sa šumom
plt.subplot(2, 1, 1)
plt.plot(t, raw_signal, label='Sirovi EEG signal')
plt.xlabel('Vreme (s)')
plt.ylabel('Amplituda (µV)')
plt.title('Sirovi EEG signal sa 50 Hz šumom')
plt.legend()
plt.grid(True)

# Donji panel: Filtrirani EEG signal
plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal, color='orange', label='Filtrirani EEG signal')
plt.xlabel('Vreme (s)')
plt.ylabel('Amplituda (µV)')
plt.title('EEG signal nakon primene Butterworth niskopropusnog filtera (cutoff=40 Hz)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
