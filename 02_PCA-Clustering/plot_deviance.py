import pandas as pd
import matplotlib.pyplot as plt
import os

# Percorso del CSV di riepilogo
csv_path = "deviance.csv"  
output_png = "images/deviance_graph.png"

# Leggi il CSV
df = pd.read_csv(csv_path)

# Raggruppa per numero di componenti principali (PCA)
grouped = df.groupby("PCA")

plt.figure(figsize=(10, 6))

# Colori fissi: rosso, blu, verde
color_map = {7: 'red', 8: 'blue', 9: 'green'}

for pcs, group in grouped:
    # Ordina per numero di cluster
    group_sorted = group.sort_values("Cluster")
    color = color_map.get(pcs, 'black')  # default nero se PCS non presente
    plt.plot(group_sorted["Cluster"], group_sorted["total_dev_lost"], 
             marker='o', label=f'{pcs} PCs', color=color)

plt.xlabel("Numero di cluster")
plt.ylabel("Devianza totale persa")
plt.title("Devianza totale persa per numero di cluster e componenti principali")
plt.grid(True)
plt.legend(title="Componenti Principali")
plt.tight_layout()

# Salva il grafico come PNG
plt.savefig(output_png, dpi=300)
plt.show()

print(f"Grafico salvato in: {output_png}")
