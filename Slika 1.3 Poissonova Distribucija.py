import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def poissonova_raspodela(lambda_vrednost):
    """
    Izračunava i prikazuje Poissonovu raspodelu.

    Args:
        lambda_vrednost (float): Prosečan broj događaja.
    """

    # Generisanje niza k vrednosti (broj događaja)
    k_vrednosti = np.arange(0, 20)  # Možete prilagoditi opseg

    # Izračunavanje verovatnoća za svaku k vrednost
    verovatnoce = stats.poisson.pmf(k_vrednosti, lambda_vrednost)

    # Crtanje grafikona
    plt.bar(k_vrednosti, verovatnoce)
    plt.xlabel('Broj događaja (k)')
    plt.ylabel('Verovatnoća')
    plt.title(f'Poissonova raspodela (λ={lambda_vrednost})')
    plt.xticks(k_vrednosti)  # Prikaz celih brojeva na x-osi
    plt.show()

# Primer upotrebe

lambda_vrednost = 7 # Prosečan broj događaja
poissonova_raspodela(lambda_vrednost)