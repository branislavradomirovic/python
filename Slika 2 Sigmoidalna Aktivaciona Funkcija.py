import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  """Sigmoid aktivacijska funkcija."""
  return 1 / (1 + np.exp(-x))

# Stvaranje niza ulaznih vrijednosti
x = np.linspace(-10, 10, 100)  # Raspon od -10 do 10 za bolji prikaz sigmoid funkcije

# Izračunavanje izlaznih vrijednosti primjenom Sigmoid funkcije
y = sigmoid(x)

# Crtanje grafikona
plt.plot(x, y)
plt.xlabel("Ulaz (x)")
plt.ylabel("Izlaz (y)")
plt.title("Sigmoid aktivacijska funkcija")
plt.grid(True)
plt.show()