import pandas as pd

# Input e output
input_file = "data/workload_nbody.txt"
output_file = "data/workload_nbody.csv"

# Definizione dei nomi estesi per le colonne vmstat
col_map = {
    "r":   "runnable_processes",
    "b":   "blocked_processes",
    "swpd": "swap_used",
    "free": "free_memory",
    "buff": "buffer_memory",
    "cache": "page_cache",
    "si":   "swap_in",
    "so":   "swap_out",
    "bi":   "blocks_in",
    "bo":   "blocks_out",
    "in":   "interrupts",
    "cs":   "context_switches",
    "us":   "user_cpu",
    "sy":   "system_cpu",
    "id":   "idle_cpu",
    "wa":   "io_wait",
    "st":   "steal_time",
    "gu":   "guest_time"
}

# -------------------------------------------
# LETTURA DEL FILE
# -------------------------------------------

rows = []
with open(input_file, "r") as f:
    for line in f:
        # ignora intestazioni e separatori
        if line.startswith("procs") or line.startswith(" r ") or line.startswith("r "):
            continue
        
        # filtra solo le righe di dati (numeriche)
        parts = line.split()
        if len(parts) == 0:
            continue
        
        rows.append(parts)

# Carica nel DataFrame
# Nota: la prima riga utile dopo l’header contiene l'ordine delle colonne
columns = list(col_map.keys())
df = pd.DataFrame(rows, columns=columns)

# Converti valori da stringa a numeri
df = df.apply(pd.to_numeric, errors='coerce')

# Rinominazione colonne con nomi completi
df = df.rename(columns=col_map)

# Salva in CSV
df.to_csv(output_file, index=False)

print("Conversione completata! File salvato come:", output_file)
