import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. CONFIGURATION DES CHEMINS
# ==========================================
print("Génération du Palier 2 - Pôle Neurologie...")

BASE_DIR = Path(__file__).resolve().parent
chemin_entree = BASE_DIR / "Données_syn" / "Données_triée" / "Dossier_Palier2_Neuro" / "patients_neuro.csv"

dossier_sortie = BASE_DIR / "Données_syn" / "Données_triée" / "Dossier_Palier2_Neuro"
dossier_sortie.mkdir(parents=True, exist_ok=True)
chemin_sortie = dossier_sortie / "dataset_neuro_palier2.csv"

# ==========================================
# 2. CHARGEMENT OU CRÉATION DES PATIENTS
# ==========================================
try:
    df_palier1 = pd.read_csv(chemin_entree)
    nombre_patients = len(df_palier1)
    ids_patients = df_palier1["ID_Patient"].values
    print(f"-> {nombre_patients} patients chargés depuis le couloir Neuro.")
except FileNotFoundError:
    print(f"-> Fichier {chemin_entree} introuvable.")
    print("Création de 800 patients factices pour la démonstration...")
    nombre_patients = 800
    ids_patients = np.arange(1, nombre_patients + 1)
    # On ajoute une colonne Température basique pour la méningite si elle n'existait pas au Palier 1
    df_palier1 = pd.DataFrame({"ID_Patient": ids_patients, "Temperature": np.random.normal(37.5, 0.5, nombre_patients)})

np.random.seed(42)

# ==========================================
# 3. DÉFINITION DES DIAGNOSTICS NEURO
# ==========================================
diagnostics = [
    "AVC_Ischemique",    # Caillot
    "AVC_Hemorragique",  # Saignement
    "Crise_Epilepsie",   # Convulsions
    "Meningite",         # Infection
    "Migraine_Severe"    # Bénin
]

probabilites = [0.30, 0.15, 0.20, 0.10, 0.25]
y_cible = np.random.choice(diagnostics, size=nombre_patients, p=probabilites)

# ==========================================
# 4. GÉNÉRATION DES EXAMENS NEUROLOGIQUES
# ==========================================
resultats = {
    "ID_Patient": ids_patients,
    "Age": np.zeros(nombre_patients),
    "Score_Glasgow": np.zeros(nombre_patients, dtype=int), # De 3 à 15
    "Deficit_Moteur": np.zeros(nombre_patients, dtype=int), # 0: Non, 1: Partiel, 2: Paralysie
    "Trouble_Parole": np.zeros(nombre_patients, dtype=int), # 0 ou 1
    "Raideur_Nuque": np.zeros(nombre_patients, dtype=int),  # 0 ou 1
    "Scanner_Cerebral": np.zeros(nombre_patients, dtype=int), # 0: Normal, 1: Ischémie, 2: Sang
    "Diagnostic_Final_Neuro": y_cible
}

for i in range(nombre_patients):
    diag = y_cible[i]
    
    # Valeurs de base (Patient normal ou migraineux)
    age = np.random.normal(45, 15)
    glasgow = np.random.choice([15, 14], p=[0.9, 0.1])
    moteur = 0
    parole = 0
    nuque = 0
    scanner = 0
    
    # Logique clinique selon la pathologie
    if diag == "AVC_Ischemique":
        age = np.random.normal(70, 10)
        glasgow = np.random.choice([15, 13, 11], p=[0.5, 0.3, 0.2])
        moteur = np.random.choice([0, 1, 2], p=[0.1, 0.4, 0.5]) # Très souvent une paralysie
        parole = np.random.choice([0, 1], p=[0.4, 0.6]) # Aphasie fréquente
        scanner = np.random.choice([0, 1], p=[0.3, 0.7]) # 1 = Ischémie (parfois invisible au tout début)
        
    elif diag == "AVC_Hemorragique":
        age = np.random.normal(65, 12)
        glasgow = np.random.choice([12, 8, 5], p=[0.3, 0.4, 0.3]) # Souvent plus grave/coma
        moteur = np.random.choice([1, 2], p=[0.3, 0.7])
        parole = np.random.choice([0, 1], p=[0.3, 0.7])
        scanner = 2 # 2 = Sang blanc au scanner (toujours visible)
        
    elif diag == "Crise_Epilepsie":
        age = np.random.normal(30, 15)
        # Post-crise, le patient est souvent confus (Glasgow bas qui remonte)
        glasgow = np.random.choice([14, 12, 9], p=[0.4, 0.4, 0.2])
        scanner = 0 # Scanner normal
        
    elif diag == "Meningite":
        age = np.random.normal(25, 15)
        glasgow = np.random.choice([15, 13], p=[0.7, 0.3])
        nuque = np.random.choice([0, 1], p=[0.1, 0.9]) # Raideur de la nuque presque toujours là
        scanner = 0 # Scanner souvent normal
        # On simule aussi une forte fièvre si la colonne existe
        if "Temperature" in df_palier1.columns:
            df_palier1.at[i, "Temperature"] = np.random.normal(39.5, 0.5)
            
    elif diag == "Migraine_Severe":
        age = np.random.normal(35, 10)
        glasgow = 15 # Parfaitement conscient, juste douloureux
        # Très rarement des symptômes qui miment un AVC (Aura migraineuse)
        moteur = np.random.choice([0, 1], p=[0.95, 0.05])
        parole = np.random.choice([0, 1], p=[0.95, 0.05])

    # Enregistrement avec sécurisation des bornes
    resultats["Age"][i] = max(18, min(99, age))
    resultats["Score_Glasgow"][i] = glasgow
    resultats["Deficit_Moteur"][i] = moteur
    resultats["Trouble_Parole"][i] = parole
    resultats["Raideur_Nuque"][i] = nuque
    resultats["Scanner_Cerebral"][i] = scanner

# Création du DataFrame et fusion
df_neuro = pd.DataFrame(resultats)
df_neuro["Age"] = df_neuro["Age"].astype(int)

df_complet = pd.merge(df_palier1, df_neuro, on="ID_Patient")
df_complet.to_csv(chemin_sortie, index=False)

print(f"-> Succès ! Fichier '{chemin_sortie.name}' généré avec {len(df_complet)} patients.")