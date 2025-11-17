import os
import re
import pandas as pd
import numpy as np

# Colonne delle componenti principali
PCA_COLS = [f"Principale{i}" for i in range(1, 10)]
UNUSED_COLS = ["steal_time", "guest_time", "Cluster"]

def deviance_lost_after_pca(csv_path):
    """
    Calcola la devianza persa e trattenuta dopo PCA.
    """
    df = pd.read_csv(csv_path, sep=',', quotechar='"', decimal=',', skipinitialspace=True)
    
    if 'Cluster' in df.columns:
        df = df[df['Cluster'].notna() & (df['Cluster'].astype(str).str.strip() != '')]
        if df.empty:
            raise ValueError("No valid 'Cluster' rows")

    df.columns = df.columns.str.strip().str.replace("'", "")

    # Converte le colonne PCA in numerico
    for col in PCA_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    numeric_cols = df.select_dtypes(include="number").columns
    pca_cols = [col for col in PCA_COLS if col in numeric_cols]
    original_cols = [col for col in numeric_cols if col not in pca_cols and col not in UNUSED_COLS]

    if not pca_cols or not original_cols:
        raise ValueError("Missing PCA or original columns")

    # Normalizzazione z-score sulle colonne originali
    df_norm = df[original_cols].apply(lambda x: (x - x.mean()) / x.std(ddof=0))

    dev_original = ((df_norm - df_norm.mean()) ** 2).sum().sum()
    dev_pca = ((df[pca_cols] - df[pca_cols].mean()) ** 2).sum().sum()

    deviance_retained = dev_pca / dev_original
    deviance_lost = 1 - deviance_retained

    return deviance_lost, deviance_retained

def intracluster_deviance(csv_path):
    """
    Calcola la devianza intra-cluster (SST per ciascun cluster e totale)
    """
    df = pd.read_csv(csv_path, sep=',', quotechar='"', decimal=',', skipinitialspace=True)
    if 'Cluster' not in df.columns:
        raise ValueError("CSV must contain 'Cluster' column")

    df = df[df['Cluster'].notna() & (df['Cluster'].astype(str).str.strip() != '')]
    feature_cols = [col for col in df.columns if col in PCA_COLS]

    results = {"total": 0}
    for cluster, group in df.groupby("Cluster"):
        mean_vec = group[feature_cols].mean().values
        sq_dists = np.sum((group[feature_cols].values - mean_vec) ** 2, axis=1)
        results[str(cluster)] = np.sum(sq_dists)
        results["total"] += results[str(cluster)]

    return results

if __name__ == "__main__":
    csv_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clustering")
    results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deviance.csv")

    if not os.path.isdir(csv_folder):
        print(f"No folder '{csv_folder}' found. Nothing to process.")
        exit()

    csv_files = [os.path.join(csv_folder, f) for f in sorted(os.listdir(csv_folder)) if f.lower().endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in '{csv_folder}'.")
        exit()

    for csv_file in csv_files:
        error_msg = ""
        try:
            pca_lost, pca_retained = deviance_lost_after_pca(csv_file)
        except Exception as e:
            pca_lost = pca_retained = float('nan')
            error_msg = f"deviance_lost_after_pca error: {e}"

        try:
            intra_total = intracluster_deviance(csv_file).get("total", 0)
        except Exception as e:
            intra_total = float('nan')
            error_msg = (error_msg + "; " if error_msg else "") + f"intracluster_deviance error: {e}"

        # Normalizzazione intra-cluster rispetto PCA
        try:
            df_main = pd.read_csv(csv_file, sep=',', quotechar='"', decimal=',', skipinitialspace=True)
            pca_cols_present = [c for c in PCA_COLS if c in df_main.columns]
            total_pca_dev = ((df_main[pca_cols_present] - df_main[pca_cols_present].mean()) ** 2).sum().sum()
            normalized_intra = (intra_total / total_pca_dev) if total_pca_dev else 0
        except Exception:
            normalized_intra = 0

        total_dev_lost = float('nan')
        if not np.isnan(pca_lost) and not np.isnan(pca_retained):
            total_dev_lost = pca_lost + normalized_intra * pca_retained

        # Estrai PCA e Cluster dal nome file (es: 7pcs_15cluster.csv)
        base = os.path.basename(csv_file)
        m = re.search(r"(\d+)pcs_(\d+)cluster", base)
        pca_count = int(m.group(1)) if m else float('nan')
        cluster_count = int(m.group(2)) if m else float('nan')
        if not m:
            error_msg = (error_msg + "; " if error_msg else "") + f"filename parse error: {base}"

        row = {
            'PCA': pca_count,
            'Cluster': cluster_count,
            'deviance_retained': pca_retained,
            'deviance_lost': pca_lost,
            'intra_cluster_total': intra_total,
            'total_dev_lost': total_dev_lost,
            'error': error_msg
        }

        header = not os.path.exists(results_file)
        pd.DataFrame([row]).to_csv(results_file, mode='a', header=header, index=False, float_format='%.6f')
        print(f"Processed: {base} -> total_dev_lost={total_dev_lost:.6f}")
