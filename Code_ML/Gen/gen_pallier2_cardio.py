import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. CONFIGURATION DES CHEMINS
# ==========================================
print("Génération du Palier 2 - Pôle Cardiologie (6 Diagnostics Précis)...")

BASE_DIR = Path(__file__).resolve().parent
chemin_entree = BASE_DIR.parent / "Données_syn" / "Données_triée" / "Dossier_Palier2_Cardio" / "patients_cardio.csv"

dossier_sortie = BASE_DIR.parent / "Données_syn" / "Données_triée" / "Dossier_Palier2_Cardio"
dossier_sortie.mkdir(parents=True, exist_ok=True)
chemin_sortie = dossier_sortie / "dataset_cardio_palier2.csv"

# ==========================================
# 2. CHARGEMENT OU CRÉATION DES PATIENTS
# ==========================================
# Les patients ici sont ceux que le Palier 1 a orientés vers la cardiologie.
# S'ils n'existent pas encore (test standalone), on en génère des factices.
try:
    df_palier1 = pd.read_csv(chemin_entree)
    nombre_patients = len(df_palier1)
    ids_patients = df_palier1["ID_Patient"].values
    print(f"-> {nombre_patients} patients chargés depuis le Palier 1.")
except FileNotFoundError:
    print(f"-> Fichier {chemin_entree} introuvable.")
    print("Création de 1200 patients factices pour la démonstration...")
    nombre_patients = 1200
    ids_patients = [f"PAT_{i:05d}" for i in range(1, nombre_patients + 1)]
    df_palier1 = pd.DataFrame({"ID_Patient": ids_patients})

np.random.seed(42)

# ==========================================
# 3. DÉFINITION DES DIAGNOSTICS CARDIAQUES
# ==========================================
# Hypothèse clinique : parmi les patients orientés en cardiologie d'urgence,
# les pathologies les plus fréquentes sont les suivantes.
# Les probabilités reflètent les épidémiologies aux urgences cardiologiques.
diagnostics = [
    "SCA_NSTEMI",          # Infarctus sans sus-décalage ST (le plus fréquent)
    "SCA_STEMI",           # Infarctus avec sus-décalage ST (urgence absolue)
    "Insuffisance_Cardiaque_Aigue",  # OAP, décompensation
    "Trouble_Rythme_Grave", # FA, TV, fibrillation ventriculaire
    "Embolie_Pulmonaire_Grave", # Aussi orienté ici car douleur thoracique
    "Pericardite_Aigue",   # Inflammation du péricarde, souvent jeune
]

# Répartition réaliste aux urgences cardiologiques
# Source : données épidémiologiques des services d'urgences cardiologiques
probabilites = [0.30, 0.15, 0.25, 0.15, 0.08, 0.07]
y_cible = np.random.choice(diagnostics, size=nombre_patients, p=probabilites)

# ==========================================
# 4. GÉNÉRATION DES EXAMENS CARDIOLOGIQUES
# ==========================================
# Chaque patient reçoit un bilan standardisé aux urgences cardiologiques :
# ECG, biologie (troponine, BNP, D-dimères), et échocardiographie.
resultats = {
    "ID_Patient": ids_patients,
    "Age": np.zeros(nombre_patients),
    "Diabetique": np.zeros(nombre_patients, dtype=int),       # 0: Non, 1: Oui
    "Hypertendu": np.zeros(nombre_patients, dtype=int),       # 0: Non, 1: Oui
    "Fumeur": np.zeros(nombre_patients, dtype=int),           # 0: Non, 1: Oui
    # --- ECG ---
    "ECG_SusDecalage_ST": np.zeros(nombre_patients, dtype=int),  # 0: Non, 1: Oui (STEMI)
    "ECG_Trouble_Rythme": np.zeros(nombre_patients, dtype=int),  # 0: Normal, 1: Arythmie
    # --- Biologie ---
    "Troponine_ng_L": np.zeros(nombre_patients),   # Marqueur de nécrose cardiaque (N < 14)
    "BNP_pg_mL": np.zeros(nombre_patients),        # Marqueur d'insuffisance cardiaque (N < 100)
    "D_Dimeres_ng_mL": np.zeros(nombre_patients),  # Marqueur thrombotique (N < 500)
    "Creatinine_umol_L": np.zeros(nombre_patients),# Fonction rénale (N : 60-110)
    # --- Echo ---
    "FEVG_pct": np.zeros(nombre_patients),         # Fraction d'éjection VG (N > 55%)
    "Epanchement_Pericarde": np.zeros(nombre_patients, dtype=int), # 0: Non, 1: Oui
    "Diagnostic_Final_Cardio": y_cible
}

for i in range(nombre_patients):
    diag = y_cible[i]

    # --- Valeurs de base (Patient cardiologique moyen) ---
    age = np.random.normal(62, 15)
    diabetique = np.random.choice([0, 1], p=[0.75, 0.25])
    hypertendu = np.random.choice([0, 1], p=[0.55, 0.45])
    fumeur = np.random.choice([0, 1], p=[0.65, 0.35])
    ecg_st = 0
    ecg_rythme = 0
    troponine = np.random.normal(8, 3)       # Normale (< 14 ng/L)
    bnp = np.random.normal(60, 30)           # Normale (< 100 pg/mL)
    ddimeres = np.random.normal(300, 100)
    creatinine = np.random.normal(85, 15)
    fevg = np.random.normal(60, 5)           # Normale (> 55%)
    epanchement = 0

    # --- Altérations spécifiques par diagnostic (avec bruit statistique réaliste) ---

    if diag == "SCA_NSTEMI":
        # Infarctus sans sus-décalage : troponine élevée, ECG souvent peu modifié
        age = np.random.normal(68, 12)
        diabetique = np.random.choice([0, 1], p=[0.55, 0.45])  # Facteur de risque
        hypertendu = np.random.choice([0, 1], p=[0.40, 0.60])
        troponine = np.random.normal(250, 150)  # Élévation marquée
        bnp = np.random.normal(120, 60)
        fevg = np.random.normal(48, 8)          # Légère altération de la pompe

    elif diag == "SCA_STEMI":
        # Infarctus avec sus-décalage : urgence absolue, signes très marqués
        age = np.random.normal(65, 12)
        ecg_st = 1                              # Toujours présent par définition
        troponine = np.random.normal(1800, 600) # Explosion de la troponine
        bnp = np.random.normal(200, 80)
        fevg = np.random.normal(38, 10)         # Pompe cardiaque sévèrement altérée
        fumeur = np.random.choice([0, 1], p=[0.40, 0.60])

    elif diag == "Insuffisance_Cardiaque_Aigue":
        # OAP : BNP très élevé, FEVG effondrée
        age = np.random.normal(75, 10)
        hypertendu = np.random.choice([0, 1], p=[0.25, 0.75]) # HTA souvent en cause
        bnp = np.random.normal(1500, 500)       # Explosion du BNP
        troponine = np.random.normal(25, 15)    # Légère élévation possible
        fevg = np.random.normal(30, 8)          # FEVG effondrée (< 40%)
        creatinine = np.random.normal(110, 30)  # Rein souffre (cardiorenal syndrome)

    elif diag == "Trouble_Rythme_Grave":
        # Fibrillation auriculaire, tachycardie ventriculaire
        age = np.random.normal(72, 13)
        ecg_rythme = 1                          # Toujours présent par définition
        troponine = np.random.normal(20, 10)    # Peut s'élever légèrement par souffrance
        bnp = np.random.normal(180, 80)
        fevg = np.random.normal(45, 10)

    elif diag == "Embolie_Pulmonaire_Grave":
        # Embolie grave avec retentissement cardiaque droit
        age = np.random.normal(58, 15)
        ddimeres = np.random.normal(5000, 1500) # Toujours très élevés
        troponine = np.random.normal(80, 40)    # Souffrance du VD
        bnp = np.random.normal(300, 100)        # VD en surcharge
        fevg = np.random.normal(52, 7)          # VG préservé mais VD dilaté
        ecg_rythme = np.random.choice([0, 1], p=[0.6, 0.4]) # S1Q3T3 possible

    elif diag == "Pericardite_Aigue":
        # Inflammation du péricarde : douleur thoracique mais biomarqueurs peu altérés
        age = np.random.normal(35, 12)          # Souvent plus jeune
        troponine = np.random.normal(18, 8)     # Légèrement élevée (myopericardite possible)
        bnp = np.random.normal(50, 20)          # Souvent normal
        fevg = np.random.normal(60, 4)          # FEVG préservée
        epanchement = np.random.choice([0, 1], p=[0.3, 0.7])  # Épanchement souvent présent
        ecg_st = np.random.choice([0, 1], p=[0.4, 0.6])  # Sus-décalage diffus (pas localisé)

    # --- Sécurisation des limites physiologiques ---
    resultats["Age"][i] = max(18, min(99, age))
    resultats["Diabetique"][i] = diabetique
    resultats["Hypertendu"][i] = hypertendu
    resultats["Fumeur"][i] = fumeur
    resultats["ECG_SusDecalage_ST"][i] = ecg_st
    resultats["ECG_Trouble_Rythme"][i] = ecg_rythme
    resultats["Troponine_ng_L"][i] = max(0.5, troponine)
    resultats["BNP_pg_mL"][i] = max(5.0, bnp)
    resultats["D_Dimeres_ng_mL"][i] = max(50.0, ddimeres)
    resultats["Creatinine_umol_L"][i] = max(40.0, min(500.0, creatinine))
    resultats["FEVG_pct"][i] = max(10.0, min(75.0, fevg))
    resultats["Epanchement_Pericarde"][i] = epanchement

# ==========================================
# 5. FINALISATION ET SAUVEGARDE
# ==========================================
df_cardio = pd.DataFrame(resultats)

# Arrondis pour un rendu médical réaliste
df_cardio["Age"] = df_cardio["Age"].astype(int)
df_cardio["Troponine_ng_L"] = df_cardio["Troponine_ng_L"].round(1)
df_cardio["BNP_pg_mL"] = df_cardio["BNP_pg_mL"].round(0)
df_cardio["D_Dimeres_ng_mL"] = df_cardio["D_Dimeres_ng_mL"].round(0)
df_cardio["Creatinine_umol_L"] = df_cardio["Creatinine_umol_L"].round(0)
df_cardio["FEVG_pct"] = df_cardio["FEVG_pct"].round(1)

# Fusion avec les données du Palier 1 (constantes vitales déjà collectées)
df_complet = pd.merge(df_palier1, df_cardio, on="ID_Patient")

df_complet.to_csv(chemin_sortie, index=False)
print(f"-> Succès ! Fichier '{chemin_sortie.name}' généré avec {len(df_complet)} patients.")