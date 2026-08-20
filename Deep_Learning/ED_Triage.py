import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("📊 DISTRIBUTIONAL VALIDATION & STATS: SYNTHETIC VS MENDELEY")
print("="*80)

# 1. Chemins vers les fichiers
BASE_DIR = Path(__file__).resolve().parent
synthetic_path = BASE_DIR / "Synthetic_Data" / "super_hospital_file.csv"
mendeley_path = BASE_DIR / "Real_Data" / "An open‑access dataset of emergency department admissions at a large teaching hospital in Iran" / "ED_triage.csv"
graph_folder = BASE_DIR / "Graph_Folder" / "Dashboard_Ultimate"
graph_folder.mkdir(parents=True, exist_ok=True)

# 2. Chargement des datasets
print("-> Chargement des données...")
try:
    df_synth = pd.read_csv(synthetic_path)
    df_real = pd.read_csv(mendeley_path)
except FileNotFoundError as e:
    print(f"[!] Erreur de fichier: {e}")
    exit()

# 3. Alignement des colonnes
column_mapping = {
    "SpO2": "O2Saturation",
    "Age": "age", 
    "HR": "PulseRate", 
    "RR": "RespiratoryRate",
    "Dia_BP": "BlooddpressurDiastol", 
    "Sys_BP": "BlooddpressurSystol", 
    "Temp": "Temperature"
}

features_to_compare = [(s, r) for s, r in column_mapping.items() if r in df_real.columns and s in df_synth.columns]

# 4. Calculs Statistiques et Tracé (Grille 2x3 compacte)
print("-> Calcul des statistiques (Kolmogorov-Smirnov) et génération des graphiques 2x3...")
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
axes = axes.flatten()

stats_results = []
plot_idx = 0

for synth_col, real_col in features_to_compare:
    # Nettoyage
    synth_data = df_synth[synth_col].replace([np.inf, -np.inf], np.nan).dropna()
    real_data = pd.to_numeric(df_real[real_col], errors='coerce').dropna()
    
    # Filtres médicaux pour les données réelles (Mendeley)
    if real_col == "PulseRate":
        real_data = real_data[(real_data > 0) & (real_data < 300)]
    elif real_col == "BlooddpressurSystol":
        real_data = real_data[(real_data > 0) & (real_data < 300)]
    elif real_col == "O2Saturation":
        real_data = real_data[(real_data > 0) & (real_data <= 100)]
        
    mean_s, std_s = synth_data.mean(), synth_data.std()
    mean_r, std_r = real_data.mean(), real_data.std()
    
    # Calcul réel du Test de Kolmogorov-Smirnov
    ks_stat, p_value = ks_2samp(synth_data, real_data)
    
    stats_results.append({
        "Variable": synth_col,
        "Moyenne Synth.": f"{mean_s:.2f} ± {std_s:.2f}",
        "Moyenne Réelle": f"{mean_r:.2f} ± {std_r:.2f}",
        "N (Réel)": len(real_data),
        "D (KS Stat)": f"{ks_stat:.4f}",
        "P-value": f"{p_value:.4e}"
    })
    
    # Tracé 2x3 (On saute Dia_BP pour gagner de la place visuellement, mais on le garde dans les stats)
    if synth_col != "Dia_BP" and plot_idx < 6:
        ax = axes[plot_idx]
        sns.kdeplot(synth_data, ax=ax, fill=True, color="#457B9D", label="Synthétique", alpha=0.5)
        sns.kdeplot(real_data, ax=ax, fill=True, color="#E63946", label="Réel (Mendeley)", alpha=0.5)
        ax.set_title(f"Distribution: {synth_col}", fontsize=13, pad=10)
        ax.set_xlabel("Valeur")
        ax.set_ylabel("Densité")
        ax.legend()
        plot_idx += 1

plt.tight_layout()
plt.savefig(graph_folder / "11_Distribution_Validation.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Export du tableau récapitulatif
df_stats = pd.DataFrame(stats_results)
print("\nRésultats de la comparaison (Calculs Bruts : Synthétique vs Mendeley réel)")
print(df_stats.to_markdown(index=False))