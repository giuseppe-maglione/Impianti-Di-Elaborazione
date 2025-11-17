import os
import re
import pandas as pd
import numpy as np

# Colonne delle componenti principali (mantenute per compatibilità con lo script originale)
PCA_COLS = [f"Principale{i}" for i in range(1, 10)]
# Colonne da ignorare per il calcolo della devianza
UNUSED_COLS = ["steal_time", "guest_time", "Cluster"]


def intracluster_deviance_no_pca(csv_path):
    """
    Calcola la devianza intra-cluster utilizzando le colonne originali
    normalizzate Z-score (caso senza PCA).
    Restituisce la devianza totale intracluster e la devianza totale delle feature originali.
    """
    df = pd.read_csv(csv_path, sep=',', quotechar='"', decimal=',', skipinitialspace=True)
    
    if 'Cluster' not in df.columns:
        raise ValueError("CSV must contain 'Cluster' column")

    df.columns = df.columns.str.strip().str.replace("'", "")
    
    # Filtra e pulisce i cluster
    df = df[df['Cluster'].notna() & (df['Cluster'].astype(str).str.strip() != '')]
    if df.empty:
        raise ValueError("No valid 'Cluster' rows")

    # Identifica le colonne numeriche originali
    numeric_cols = df.select_dtypes(include="number").columns
    original_cols = [col for col in numeric_cols if col not in PCA_COLS and col not in UNUSED_COLS]

    if not original_cols:
        raise ValueError("Missing original feature columns")

    # Normalizzazione z-score sulle colonne originali (passo cruciale)
    # Calcola la media e la deviazione standard solo sui dati validi (non-NaN)
    df_norm = df[original_cols].apply(lambda x: (x - x.mean()) / x.std(ddof=0))
    
    # Rimuovi le righe con NaN (risultanti dalla normalizzazione su colonne con dev.st. zero o righe mancanti)
    df_norm.dropna(inplace=True) 
    
    if df_norm.empty:
         raise ValueError("DataFrame empty after z-score normalization and NaN removal")

    # Calcolo della devianza totale delle feature originali normalizzate (DENOMINATORE)
    # Poiché i dati sono normalizzati Z-score, la media è 0.
    dev_original_total = (df_norm ** 2).sum().sum() 

    # Calcolo della devianza intra-cluster (SST) sulle colonne originali normalizzate (NUMERATORE)
    intra_cluster_dev_total = 0
    # Assicurati che l'indice e il cluster corrispondano a df_norm
    df_with_norm = pd.concat([df_norm, df['Cluster']], axis=1).dropna(subset=original_cols)

    for cluster, group in df_with_norm.groupby("Cluster"):
        # Calcola la media del cluster (centroidi)
        mean_vec = group[original_cols].mean().values
        # Calcola la somma dei quadrati delle distanze intra-cluster
        sq_dists = np.sum((group[original_cols].values - mean_vec) ** 2, axis=1)
        intra_cluster_dev_total += np.sum(sq_dists)

    return dev_original_total, intra_cluster_dev_total


if __name__ == "__main__":
    # La parte di gestione dei percorsi è mantenuta identica
    csv_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clustering_nopca")
    results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deviance_nopca2.csv") # Nome file aggiornato

    if not os.path.isdir(csv_folder):
        print(f"No folder '{csv_folder}' found. Nothing to process.")
        exit()

    csv_files = [os.path.join(csv_folder, f) for f in sorted(os.listdir(csv_folder)) if f.lower().endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in '{csv_folder}'.")
        exit()

    for csv_file in csv_files:
        error_msg = ""
        total_dev_lost_no_pca = float('nan')
        intra_cluster_total = float('nan')
        
        try:
            # Calcolo della devianza intracluster sulle feature originali normalizzate
            dev_original, intra_cluster_total = intracluster_deviance_no_pca(csv_file)
            
            # La devianza totale persa è la devianza intra-cluster normalizzata
            # rispetto alla devianza totale delle feature originali (SST/SS_Total)
            total_dev_lost_no_pca = (intra_cluster_total / dev_original) if dev_original else float('nan')
            
        except Exception as e:
            error_msg = f"intracluster_deviance_no_pca error: {e}"

        # Estrai Cluster dal nome file (es: 7pcs_15cluster.csv, ignora 'pcs')
        base = os.path.basename(csv_file)
        m = re.search(r"nopca_(\d+)cluster", base)
        cluster_count = int(m.group(1)) if m else float('nan')
        if not m:
            error_msg = (error_msg + "; " if error_msg else "") + f"filename parse error: {base}"
            
        row = {
            'PCA': 0,
            'Cluster': cluster_count,
            'intra_cluster_dev': intra_cluster_total,
            'total_feature_dev': dev_original if 'dev_original' in locals() else float('nan'),
            'total_dev_lost': total_dev_lost_no_pca,
            'error': error_msg
        }

        header = not os.path.exists(results_file)
        pd.DataFrame([row]).to_csv(results_file, mode='a', header=header, index=False, float_format='%.6f')
        print(f"Processed: {base} -> total_dev_lost_no_pca={total_dev_lost_no_pca:.6f}")