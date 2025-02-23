import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Simulacija podataka
np.random.seed(42)
n_channels = 32
# Generisanje slučajnih pozicija elektroda na površini glave (u 2D prostoru)
# Pretpostavljamo jednostavan model gde se pozicije kreću unutar kruga
angles = np.linspace(0, 2*np.pi, n_channels, endpoint=False)
radii = np.random.uniform(0.5, 1.0, n_channels)
electrode_x = radii * np.cos(angles)
electrode_y = radii * np.sin(angles)

# Simulacija sirovih EEG amplituda (µV) za svaki kanal
raw_amplitudes = np.random.uniform(-150, 150, n_channels)

# Simulacija CSP komponente (prva komponenta koja naglašava razlike između klasa)
# Pretpostavimo da CSP algoritam izdvoji pozitivne vrednosti za kanale koji imaju veći značaj
csp_component = raw_amplitudes * np.random.uniform(0.5, 1.5, n_channels)

# Kreiranje grida za interpolaciju (za topografsku mapu)
grid_x, grid_y = np.mgrid[-1:1:200j, -1:1:200j]
# Interpolacija sirovih EEG amplituda
grid_raw = griddata((electrode_x, electrode_y), raw_amplitudes, (grid_x, grid_y), method='cubic')
# Interpolacija CSP komponente
grid_csp = griddata((electrode_x, electrode_y), csp_component, (grid_x, grid_y), method='cubic')

# Plotovanje grafikona
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Levii panel: Topografska mapa sirovih EEG amplituda
c1 = axes[0].imshow(grid_raw, extent=(-1, 1, -1, 1), origin='lower', cmap='jet')
axes[0].scatter(electrode_x, electrode_y, c='k', s=50, label='Elektrode')
axes[0].set_title('Topografska mapa sirovih EEG amplituda')
axes[0].set_xlabel('X pozicija')
axes[0].set_ylabel('Y pozicija')
axes[0].legend()
fig.colorbar(c1, ax=axes[0], orientation='vertical', label='Amplituda (µV)')

# Desni panel: CSP komponenta nakon obuke
c2 = axes[1].imshow(grid_csp, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
axes[1].scatter(electrode_x, electrode_y, c='k', s=50, label='Elektrode')
axes[1].set_title('CSP komponenta - Prva komponenta')
axes[1].set_xlabel('X pozicija')
axes[1].set_ylabel('Y pozicija')
axes[1].legend()
fig.colorbar(c2, ax=axes[1], orientation='vertical', label='CSP vrednost')

plt.suptitle('Prostorna filtracija korišćenjem CSP algoritma')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
