import numpy as np
import matplotlib.pyplot as plt

# Generišemo slučajne ulazne podatke
np.random.seed(42)
x = np.linspace(-10, 10, 400)
# Simulacija linearne transformacije: y = wx + b, gde su w=1.5 i b=-2
w = 1.5
b = -2
z = w * x + b

# Primena ReLU aktivacione funkcije: f(z) = max(0, z)
def relu(x):
    return np.maximum(0, x)

a = relu(z)

# Kreiranje grafikona
plt.figure(figsize=(14, 6))

# Prvi panel: Linearna transformacija
plt.subplot(1, 2, 1)
plt.plot(x, z, color='blue', label='z = 1.5x - 2')
plt.title('Rezultat linearne transformacije')
plt.xlabel('Ulaz x')
plt.ylabel('z = 1.5x - 2')
plt.legend()
plt.grid(True)

# Drugi panel: Rezultat ReLU aktivacije
plt.subplot(1, 2, 2)
plt.plot(x, a, color='red', label='ReLU(z)')
plt.title('Rezultat ReLU aktivacione funkcije')
plt.xlabel('Ulaz x')
plt.ylabel('Aktivirani izlaz')
plt.legend()
plt.grid(True)

plt.suptitle('Transformacija podataka kroz skriveni sloj sa ReLU aktivacijom')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
