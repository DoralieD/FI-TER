import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. CONFIGURATION DES CHEMINS
# ==========================================
print("Génération du Palier 2 - Pôle Infectiologie (6 Diagnostics Précis)...")

BASE_DIR = Path(__file__).resolve().parent
chemin_entree = BASE_DIR.parent / "Données_syn" / "Données_triée" / "Dossier_Palier2_Infectieux" / "patients_infectieux.csv"

dossier_sortie = BASE_DIR.parent / "Données_syn" / "Données_triée" / "Dossier_Palier2_Infectieux"
dossier_sortie.mkdir(parents=True, exist_ok=True)
chemin_sortie = dossier_sortie / "dataset_infectieux_palier2.csv"

# ==========================================
# 2. CHARGEMENT OU CRÉATION DES PATIENTS
# ==========================================
try:
    df_palier1 = pd.read_csv(chemin_entree)
    nombre_patients = len(df_palier1)
    ids_patients = df_palier1["ID_Patient"].values
    print(f"-> {nombre_patients} patients chargés depuis le Palier 1.")
except FileNotFoundError:
    print(f"-> Fichier {chemin_entree} introuvable.")
    print("Création de 900 patients factices pour la démonstration...")
    nombre_patients = 900
    ids_patients = [f"PAT_{i:05d}" for i in range(1, nombre_patients + 1)]
    df_palier1 = pd.DataFrame({"ID_Patient": ids_patients})

np.random.seed(42)

# ==========================================
# 3. DÉFINITION DES DIAGNOSTICS INFECTIEUX
# ==========================================
# Hypothèse clinique : parmi les patients fébriles orientés en infectiologie,
# on distingue 6 grandes catégories avec des marqueurs biologiques distincts.
# La sepsis grave est une urgence vitale et doit être identifiée rapidement.
diagnostics = [
    "Infection_Urinaire_Grave",   # Pyélonéphrite/Sepsis urinaire (très fréquent)
    "Pneumonie_Communautaire",    # Infection pulmonaire à domicile
    "Sepsis_Grave",               # Infection avec défaillance d'organe (SOFA ≥ 2)
    "Gastroenterite_Bacterienne", # Infection digestive (Salmonelle, Campylobacter)
    "Cellulite_Infectieuse",      # Infection cutanée/tissus mous
    "Meningite_Bacterienne",      # Urgence neurologique et infectieuse
]

# Répartition épidémiologique aux urgences infectieuses
probabilites = [0.30, 0.25, 0.15, 0.15, 0.10, 0.05]
y_cible = np.random.choice(diagnostics, size=nombre_patients, p=probabilites)

# ==========================================
# 4. GÉNÉRATION DES EXAMENS INFECTIOLOGIQUES
# ==========================================
# Bilan standard d'une infection aux urgences :
# NFS, CRP, PCT, hémocultures, ECBU (bandelette urinaire), lactates
resultats = {
    "ID_Patient": ids_patients,
    "Age": np.zeros(nombre_patients),
    "Immunodeprime": np.zeros(nombre_patients, dtype=int),  # 0: Non, 1: Oui (cancer, VIH, corticoïdes)
    # --- Signes vitaux de la réponse inflammatoire ---
    "Temperature_C": np.zeros(nombre_patients),
    "FC_Tachycardie": np.zeros(nombre_patients, dtype=int), # FC > 90 bpm : 0/1
    "FR_Polypnee": np.zeros(nombre_patients, dtype=int),    # FR > 20/min : 0/1
    # --- Biologie - Marqueurs d'infection ---
    "Leucocytes_G_L": np.zeros(nombre_patients),     # Globules blancs (N: 4-10)
    "CRP_mg_L": np.zeros(nombre_patients),            # Protéine C-Réactive (N < 5)
    "PCT_ng_mL": np.zeros(nombre_patients),           # Procalcitonine, marqueur bactérien (N < 0.5)
    "Lactates_mmol_L": np.zeros(nombre_patients),     # Marqueur de choc (N < 2, choc > 4)
    # --- Biologie - Retentissement sur les organes ---
    "Creatinine_umol_L": np.zeros(nombre_patients),  # Rein (N: 60-110)
    "Bilirubine_umol_L": np.zeros(nombre_patients),  # Foie (N < 20)
    # --- Examens orientés ---
    "ECBU_Positif": np.zeros(nombre_patients, dtype=int),    # Analyse d'urine positive : 0/1
    "Hemoc_Positive": np.zeros(nombre_patients, dtype=int),  # Hémoculture positive : 0/1
    "Diagnostic_Final_Infectieux": y_cible
}

for i in range(nombre_patients):
    diag = y_cible[i]

    # --- Valeurs de base (Patient infectieux modéré) ---
    age = np.random.normal(50, 20)
    immunodeprime = np.random.choice([0, 1], p=[0.88, 0.12])
    temp = np.random.normal(38.5, 0.5)
    tachycardie = np.random.choice([0, 1], p=[0.5, 0.5])
    polypnee = np.random.choice([0, 1], p=[0.7, 0.3])
    leuco = np.random.normal(13, 3)
    crp = np.random.normal(60, 30)
    pct = np.random.normal(1.5, 1.0)
    lactates = np.random.normal(1.5, 0.5)
    creatinine = np.random.normal(90, 20)
    bilirubine = np.random.normal(12, 4)
    ecbu = 0
    hemoc = 0

    # --- Altérations spécifiques par diagnostic ---

    if diag == "Infection_Urinaire_Grave":
        # Pyélonéphrite : fièvre, frissons, douleur lombaire - bactériurie massive
        age = np.random.normal(45, 20)
        temp = np.random.normal(39.2, 0.6)
        leuco = np.random.normal(16, 4)
        crp = np.random.normal(120, 40)
        pct = np.random.normal(3.0, 1.5)         # PCT modérément élevée
        ecbu = np.random.choice([0, 1], p=[0.05, 0.95])  # ECBU positif +++
        hemoc = np.random.choice([0, 1], p=[0.70, 0.30]) # Hémoculture positive dans 30% des pyélos

    elif diag == "Pneumonie_Communautaire":
        # Pneumopathie : syndrome alvéolaire, fièvre, toux productive
        age = np.random.normal(60, 15)
        temp = np.random.normal(39.0, 0.7)
        leuco = np.random.normal(17, 5)
        crp = np.random.normal(180, 60)
        pct = np.random.normal(5.0, 2.0)         # PCT nettement élevée (bactérien)
        polypnee = 1                              # Polypnée quasi-constante
        hemoc = np.random.choice([0, 1], p=[0.85, 0.15]) # Bactériémie possible
        lactates = np.random.normal(1.8, 0.6)

    elif diag == "Sepsis_Grave":
        # Sepsis = infection + défaillance d'organe (critères SOFA)
        # Urgence vitale absolue : mortalité élevée
        immunodeprime = np.random.choice([0, 1], p=[0.60, 0.40])  # Souvent terrain fragile
        temp = np.random.normal(39.5, 0.8)
        leuco = np.random.normal(20, 6)
        crp = np.random.normal(280, 80)
        pct = np.random.normal(25.0, 10.0)       # PCT très élevée = bactériémie probable
        lactates = np.random.normal(4.5, 1.5)    # Lactates > 4 = choc septique
        tachycardie = 1
        polypnee = 1
        creatinine = np.random.normal(160, 50)   # Rein en souffrance
        bilirubine = np.random.normal(35, 15)    # Foie atteint
        hemoc = np.random.choice([0, 1], p=[0.30, 0.70]) # Hémoculture souvent positive

    elif diag == "Gastroenterite_Bacterienne":
        # Salmonelle, Campylobacter, C.diff : fièvre + troubles digestifs + leucocytose
        age = np.random.normal(35, 20)
        temp = np.random.normal(38.7, 0.5)
        leuco = np.random.normal(14, 3)
        crp = np.random.normal(70, 25)
        pct = np.random.normal(1.0, 0.5)         # PCT peu élevée (infection localisée)
        lactates = np.random.normal(1.6, 0.5)
        ecbu = 0                                  # ECBU négatif

    elif diag == "Cellulite_Infectieuse":
        # Infection des tissus mous : fièvre + placard rouge et chaud + leucocytose
        age = np.random.normal(55, 18)
        temp = np.random.normal(38.8, 0.6)
        leuco = np.random.normal(15, 3)
        crp = np.random.normal(100, 35)
        pct = np.random.normal(0.8, 0.4)         # PCT modérée (infection locorégionale)
        hemoc = np.random.choice([0, 1], p=[0.85, 0.15])

    elif diag == "Meningite_Bacterienne":
        # Urgence absolue : fièvre + méningisme + purpura possible
        # Mortalité élevée sans antibiotiques immédiats
        age = np.random.normal(28, 15)
        temp = np.random.normal(40.0, 0.6)       # Très forte fièvre
        leuco = np.random.normal(22, 5)           # Hyperleucocytose marquée
        crp = np.random.normal(250, 70)
        pct = np.random.normal(15.0, 6.0)        # PCT très élevée
        tachycardie = 1
        polypnee = 1
        hemoc = np.random.choice([0, 1], p=[0.40, 0.60]) # Hémoculture positive souvent
        immunodeprime = np.random.choice([0, 1], p=[0.75, 0.25]) # Plus de risque chez ID

    # --- Sécurisation des limites physiologiques ---
    resultats["Age"][i] = max(1, min(99, age))
    resultats["Immunodeprime"][i] = immunodeprime
    resultats["Temperature_C"][i] = max(35.0, min(42.5, temp))
    resultats["FC_Tachycardie"][i] = tachycardie
    resultats["FR_Polypnee"][i] = polypnee
    resultats["Leucocytes_G_L"][i] = max(0.5, min(50.0, leuco))
    resultats["CRP_mg_L"][i] = max(0.1, min(600.0, crp))
    resultats["PCT_ng_mL"][i] = max(0.05, min(100.0, pct))
    resultats["Lactates_mmol_L"][i] = max(0.5, min(15.0, lactates))
    resultats["Creatinine_umol_L"][i] = max(40.0, min(600.0, creatinine))
    resultats["Bilirubine_umol_L"][i] = max(3.0, min(200.0, bilirubine))
    resultats["ECBU_Positif"][i] = ecbu
    resultats["Hemoc_Positive"][i] = hemoc

# ==========================================
# 5. FINALISATION ET SAUVEGARDE
# ==========================================
df_infectieux = pd.DataFrame(resultats)

# Arrondis pour un rendu médical réaliste
df_infectieux["Age"] = df_infectieux["Age"].astype(int)
df_infectieux["Temperature_C"] = df_infectieux["Temperature_C"].round(1)
df_infectieux["Leucocytes_G_L"] = df_infectieux["Leucocytes_G_L"].round(1)
df_infectieux["CRP_mg_L"] = df_infectieux["CRP_mg_L"].round(1)
df_infectieux["PCT_ng_mL"] = df_infectieux["PCT_ng_mL"].round(2)
df_infectieux["Lactates_mmol_L"] = df_infectieux["Lactates_mmol_L"].round(2)
df_infectieux["Creatinine_umol_L"] = df_infectieux["Creatinine_umol_L"].round(0)
df_infectieux["Bilirubine_umol_L"] = df_infectieux["Bilirubine_umol_L"].round(1)

# Fusion avec les données du Palier 1
df_complet = pd.merge(df_palier1, df_infectieux, on="ID_Patient")

df_complet.to_csv(chemin_sortie, index=False)
print(f"-> Succès ! Fichier '{chemin_sortie.name}' généré avec {len(df_complet)} patients.")