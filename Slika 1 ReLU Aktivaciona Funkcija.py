import numpy as np
import matplotlib.pyplot as plt

def relu(x):
  """ReLU aktivacijska funkcija."""
  return np.maximum(0, x)

# Stvaranje niza ulaznih vrijednosti
x = np.linspace(-5, 5, 100)

# Izračunavanje izlaznih vrijednosti primjenom ReLU funkcije
y = relu(x)

# Crtanje grafikona
plt.plot(x, y)
plt.xlabel("Ulaz (x)")
plt.ylabel("Izlaz (y)")
plt.title("ReLU aktivacijska funkcija")
plt.grid(True)
plt.show()