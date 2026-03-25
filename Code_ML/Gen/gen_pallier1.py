import pandas as pd
import numpy as np
import os
from pathlib import Path # <-- Ajout de l'import

# Fixer la graine pour avoir un résultat reproductible
np.random.seed(42)
n_samples = 100000

print(f"Génération de {n_samples} patients en cours...")

# 1. Génération des Constantes Vitales (Capteurs IoMT)
donnees = {
    "ID_Patient": [f"PAT_{i:05d}" for i in range(1, n_samples + 1)],
    "FC": np.random.normal(85, 20, n_samples).astype(int), # Fréquence Cardiaque moy=85
    "Tension_Sys": np.random.normal(125, 25, n_samples).astype(int),
    "Tension_Dia": np.random.normal(80, 15, n_samples).astype(int),
    "FR": np.random.normal(16, 5, n_samples).astype(int), # Fréquence Respiratoire
    "Temp": np.round(np.random.normal(37.2, 0.8, n_samples), 1),
    "SpO2": np.random.normal(97, 4, n_samples).astype(int),
}

# Limitation des valeurs aberrantes mathématiques pour rester réaliste
donnees["FC"] = np.clip(donnees["FC"], 40, 200)
donnees["Tension_Sys"] = np.clip(donnees["Tension_Sys"], 70, 220)
donnees["Tension_Dia"] = np.clip(donnees["Tension_Dia"], 40, 130)
donnees["FR"] = np.clip(donnees["FR"], 8, 40)
donnees["SpO2"] = np.clip(donnees["SpO2"], 70, 100)

df = pd.DataFrame(donnees)

# 2. Génération des Observations Physiques (0 = Non, 1 = Oui)
df["Obs_Paleur"] = np.random.choice([0, 1], p=[0.75, 0.25], size=n_samples)
df["Obs_Cyanose"] = np.random.choice([0, 1], p=[0.92, 0.08], size=n_samples)
df["Obs_Sueurs"] = np.random.choice([0, 1], p=[0.80, 0.20], size=n_samples)
df["Obs_Inconscient"] = np.random.choice([0, 1], p=[0.96, 0.04], size=n_samples)
df["Obs_Confusion"] = np.random.choice([0, 1], p=[0.88, 0.12], size=n_samples)
df["Obs_Frissons"] = np.random.choice([0, 1], p=[0.85, 0.15], size=n_samples)
df["Obs_Hemorragie"] = np.random.choice([0, 1], p=[0.95, 0.05], size=n_samples)
df["Obs_DouleurThorax"] = np.random.choice([0, 1], p=[0.85, 0.15], size=n_samples)
df["Obs_DetresseRespi"] = np.random.choice([0, 1], p=[0.90, 0.10], size=n_samples)
df["Obs_Eruption"] = np.random.choice([0, 1], p=[0.93, 0.07], size=n_samples)
df["Obs_TraumaPenetrant"] = np.random.choice([0, 1], p=[0.98, 0.02], size=n_samples) # Très rare

    # 3. Fonction pour ajouter de l'incertitude (Bruit médical)
def appliquer_bruit(condition, vrai_positif=0.79, faux_positif=0.12):
        hasard = np.random.rand(len(condition))
        return np.where(condition, hasard < vrai_positif, hasard < faux_positif).astype(int)

# 4. Création des diagnostics finaux (Vérité Terrain) avec nos règles
regle_cardio = (df["FC"] > 120) | (df["Tension_Sys"] > 160) | df["Obs_DouleurThorax"] | (df["Obs_Paleur"] & df["Obs_Sueurs"])
df["Verite_Cardio"] = appliquer_bruit(regle_cardio)

regle_respi = (df["SpO2"] < 92) | (df["FR"] > 25) | df["Obs_DetresseRespi"] | df["Obs_Cyanose"]
df["Verite_Respi"] = appliquer_bruit(regle_respi)

regle_infectieux = (df["Temp"] > 38.5) | ((df["Temp"] > 37.8) & df["Obs_Frissons"]) | df["Obs_Eruption"]
df["Verite_Infectieux"] = appliquer_bruit(regle_infectieux)

regle_neuro = df["Obs_Inconscient"] | df["Obs_Confusion"] | (df["Tension_Sys"] > 200)
df["Verite_Neuro"] = appliquer_bruit(regle_neuro)

# ==========================================
# 5. EXPORTATION ROBUSTE AVEC PATHLIB
# ==========================================
# On part du dossier où se trouve CE script (Code_ML)
BASE_DIR = Path(__file__).resolve().parent

# On cible le sous-dossier Données_syn
dossier_syn = BASE_DIR.parent / "Données_syn"

# Crée le dossier s'il n'existe pas
dossier_syn.mkdir(parents=True, exist_ok=True)

# On définit le chemin complet du fichier
chemin_complet = dossier_syn / "dataset_prise_constante.csv"

# Sauvegarde
df.to_csv(chemin_complet, index=False)
print(f"Fichier sauvegardé avec succès sous : {chemin_complet}")