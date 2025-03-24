import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def binomna_distribucija(n, p):
    """
    Izračunava i prikazuje binomnu distribuciju.

    Args:
        n (int): Broj pokušaja.
        p (float): Verovatnoća uspeha u svakom pokušaju.
    """

    # Generisanje niza k vrednosti (broj uspeha)
    k_vrednosti = np.arange(0, n + 1)

    # Izračunavanje verovatnoća za svaku k vrednost
    verovatnoce = stats.binom.pmf(k_vrednosti, n, p)

    # Crtanje grafikona
    plt.bar(k_vrednosti, verovatnoce)
    plt.xlabel('Broj uspeha (k)')
    plt.ylabel('Verovatnoća')
    plt.title(f'Binomna distribucija (n={n}, p={p})')
    plt.xticks(k_vrednosti)  # Prikaz celih brojeva na x-osi
    plt.show()

n = 20 # drugi primer
p = 0.7 # drugi primer
binomna_distribucija(n,p)