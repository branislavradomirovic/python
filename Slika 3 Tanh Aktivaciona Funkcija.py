import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
  """Hiperbolička tangens funkcija."""
  return np.tanh(x)

# Stvaranje niza ulaznih vrijednosti
x = np.linspace(-5, 5, 100)  # Raspon od -5 do 5 za bolji prikaz tanh funkcije

# Izračunavanje izlaznih vrijednosti primjenom tanh funkcije
y = tanh(x)

# Crtanje grafikona
plt.plot(x, y)
plt.xlabel("Ulaz (x)")
plt.ylabel("Izlaz (y)")
plt.title("Hiperbolička tangens funkcija (tanh)")
plt.grid(True)
plt.show()