import numpy as np
import matplotlib.pyplot as plt

# Generisanje vremenskog vektora (0 do 5 sekundi)
t = np.linspace(0, 5, 500)

# Simulacija EEG signala:
# Kanali Fz i Cz imaju veće amplitude (multiplikator 150)
Fz = 150 * np.sin(2 * np.pi * 1 * t)
Cz = 150 * np.cos(2 * np.pi * 1 * t)
# Kanali Pz i Oz imaju niže amplitude (multiplikator 50)
Pz = 50 * np.sin(2 * np.pi * 1 * t)
Oz = 50 * np.cos(2 * np.pi * 1 * t)

# Kreiranje levog panela: Nenormalizovani EEG signali
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(t, Fz, label='Fz')
plt.plot(t, Cz, label='Cz')
plt.plot(t, Pz, label='Pz')
plt.plot(t, Oz, label='Oz')
plt.xlabel('Vreme (s)')
plt.ylabel('Amplituda (µV)')
plt.title('Nenormalizovani EEG signali')
plt.ylim([-150, 150])
plt.legend()

# Funkcija za Min-Max normalizaciju
def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# Normalizacija signala za svaki kanal
Fz_norm = min_max_normalize(Fz)
Cz_norm = min_max_normalize(Cz)
Pz_norm = min_max_normalize(Pz)
Oz_norm = min_max_normalize(Oz)

# Kreiranje desnog panela: Normalizovani EEG signali
plt.subplot(1, 2, 2)
plt.plot(t, Fz_norm, label='Fz')
plt.plot(t, Cz_norm, label='Cz')
plt.plot(t, Pz_norm, label='Pz')
plt.plot(t, Oz_norm, label='Oz')
plt.xlabel('Vreme (s)')
plt.ylabel('Normalizovana amplituda')
plt.title('Min-Max Normalizovani EEG signali')
plt.ylim([0, 1])
plt.legend()

plt.tight_layout()
plt.show()
