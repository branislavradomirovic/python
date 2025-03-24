import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def prikazi_normalnu_distribuciju(podaci):
    """
    Prikazuje normalnu distribuciju skupa podataka.

    Args:
        podaci (list or numpy.ndarray): Skup podataka.
    """

    # Izračunaj srednju vrednost i standardnu devijaciju
    srednja_vrednost = np.mean(podaci)
    standardna_devijacija = np.std(podaci)

    # Generiši x vrednosti za grafikon
    x = np.linspace(min(podaci), max(podaci), 100)

    # Izračunaj y vrednosti za normalnu distribuciju
    y = stats.norm.pdf(x, srednja_vrednost, standardna_devijacija)

    # Kreiraj histogram podataka
    plt.hist(podaci, density=True, alpha=0.6, color='g', label='Histogram podataka')

    # Nacrtaj normalnu distribuciju
    plt.plot(x, y, 'r', label='Normalna distribucija')

    # Dodaj oznake i legendu
    plt.xlabel('Vrednosti')
    plt.ylabel('Verovatnoća')
    plt.title('Normalna distribucija skupa podataka')
    plt.legend()

    # Prikaži grafikon
    plt.show()

# Primer upotrebe
podaci = np.random.normal(0, 1, 1000)  # Generisanje 1000 slučajnih brojeva iz normalne distribucije
prikazi_normalnu_distribuciju(podaci)