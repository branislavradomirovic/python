import numpy as np
import matplotlib.pyplot as plt

# Definisanje softmax funkcije
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # stabilizacija eksponencijalne funkcije
    return exp_x / exp_x.sum(axis=0)

# Primer ulaznog vektora iz izlaznog sloja (npr. logiti)
x = np.array([2.0, 1.0, 0.1])
softmax_output = softmax(x)

# Kreiranje grafikona
fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.5
index = np.arange(len(x))

# Prikaz originalnih ulaznih vrednosti (logiti) i softmax izlaza
ax.bar(index, x, bar_width, label='Logiti', color='skyblue')
ax.bar(index + bar_width, softmax_output, bar_width, label='Softmax izlaz', color='salmon')

ax.set_xlabel('Klasni indeksi')
ax.set_ylabel('Vrednosti')
ax.set_title('Transformacija izlaznih vrednosti u verovatnoće pomoću softmax funkcije')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(['Klasa 1', 'Klasa 2', 'Klasa 3'])
ax.legend()

plt.tight_layout()
plt.show()
