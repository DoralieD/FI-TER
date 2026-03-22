import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. CONFIGURATION DES CHEMINS
# ==========================================
print("Génération du Palier 2 Renforcé (7 Diagnostics Précis)...")

BASE_DIR = Path(__file__).resolve().parent
chemin_entree = BASE_DIR / "Données_syn" / "Données_triée" / "Dossier_Palier2_Respi" / "patients_respi.csv"

dossier_sortie = BASE_DIR / "Données_syn" / "Données_triée" / "Dossier_Palier2_Respi"
dossier_sortie.mkdir(parents=True, exist_ok=True)
chemin_sortie = dossier_sortie / "dataset_respi_palier2_renforce.csv"

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
    print("Création de 1000 patients factices pour la démonstration...")
    nombre_patients = 1000
    ids_patients = np.arange(1, nombre_patients + 1)
    df_palier1 = pd.DataFrame({"ID_Patient": ids_patients})

np.random.seed(42)

# ==========================================
# 3. LES 7 DIAGNOSTICS ACTIONNABLES
# ==========================================
diagnostics = [
    "Pneumonie_Bacterienne", # Antibiotiques urgents
    "Infection_Virale_Severe", # Surveillance, Oxygène (ex: Grippe, COVID)
    "Crise_Asthme",          # Bronchodilatateurs (Sujets jeunes)
    "Exacerbation_BPCO",     # Bronchodilatateurs (Sujets âgés/fumeurs)
    "Embolie_Pulmonaire",    # Anticoagulants urgents
    "Pneumothorax",          # Pose d'un drain
    "Hyperventilation_Angoisse" # Réassurance
]

# Répartition réaliste aux urgences
probabilites = [0.20, 0.25, 0.15, 0.15, 0.10, 0.05, 0.10]
y_cible = np.random.choice(diagnostics, size=nombre_patients, p=probabilites)

# ==========================================
# 4. GÉNÉRATION DES DONNÉES CLINIQUES ET LABO
# ==========================================
resultats = {
    "ID_Patient": ids_patients,
    "Age": np.zeros(nombre_patients),
    "Fumeur": np.zeros(nombre_patients, dtype=int),
    "Temperature_C": np.zeros(nombre_patients),
    "Leucocytes_G_L": np.zeros(nombre_patients),  # Globules blancs (Normale: 4 à 10)
    "D_Dimeres_ng_mL": np.zeros(nombre_patients),
    "CRP_mg_L": np.zeros(nombre_patients),
    "Gaz_Sang_PaO2": np.zeros(nombre_patients),
    "Gaz_Sang_PaCO2": np.zeros(nombre_patients),
    "Radio_Thorax": np.zeros(nombre_patients, dtype=int), # 0:Normal, 1:Foyer, 2:Décollement, 3:Distension
    "Diagnostic_Final_Respi": y_cible
}

for i in range(nombre_patients):
    diag = y_cible[i]
    
    # --- Valeurs par défaut (Patient moyen stressé) ---
    age = np.random.normal(50, 15)
    fumeur = np.random.choice([0, 1], p=[0.7, 0.3])
    temp = np.random.normal(37.2, 0.4)
    leuco = np.random.normal(7, 2)
    ddimeres = np.random.normal(250, 100)
    crp = np.random.normal(3, 2)
    pao2 = np.random.normal(95, 5)
    paco2 = np.random.normal(40, 2)
    radio = 0
    
    # --- Altérations spécifiques selon la maladie (Avec du bruit statistique) ---
    if diag == "Pneumonie_Bacterienne":
        temp = np.random.normal(39.5, 0.6)        # Forte fièvre
        leuco = np.random.normal(18, 4)           # Explosion des globules blancs
        crp = np.random.normal(150, 50)           # Forte inflammation
        pao2 = np.random.normal(70, 10)
        radio = np.random.choice([1, 0], p=[0.9, 0.1]) # Foyer visible
        
    elif diag == "Infection_Virale_Severe":
        temp = np.random.normal(38.8, 0.5)        # Fièvre modérée à forte
        leuco = np.random.normal(5, 2)            # Globules blancs normaux ou bas (typique virus)
        crp = np.random.normal(40, 20)            # Inflammation modérée
        pao2 = np.random.normal(65, 12)
        radio = np.random.choice([0, 1], p=[0.6, 0.4]) # Souvent normale au début
        
    elif diag == "Crise_Asthme":
        age = np.random.normal(25, 10)            # Souvent jeune
        paco2 = np.random.normal(35, 5)           # Respire vite au début
        radio = np.random.choice([0, 3], p=[0.8, 0.2])
        
    elif diag == "Exacerbation_BPCO":
        age = np.random.normal(70, 8)             # Souvent âgé
        fumeur = np.random.choice([1, 0], p=[0.9, 0.1]) # Presque toujours fumeur ou ex-fumeur
        paco2 = np.random.normal(60, 10)          # Rétention de CO2 sévère
        radio = 3                                 # Distension thoracique visible
        
    elif diag == "Embolie_Pulmonaire":
        ddimeres = np.random.normal(4000, 1500)   # Marqueur de caillot très élevé
        pao2 = np.random.normal(60, 10)           # Manque d'oxygène
        paco2 = np.random.normal(32, 4)           # Hyperventilation réflexe
        
    elif diag == "Pneumothorax":
        age = np.random.normal(30, 15)            # Souvent homme jeune et grand
        pao2 = np.random.normal(75, 10)
        radio = 2                                 # Décollement de la plèvre visible
        
    elif diag == "Hyperventilation_Angoisse":
        paco2 = np.random.normal(28, 4)           # Fait chuter son CO2 en respirant trop vite
        pao2 = np.random.normal(99, 2)            # Oxygénation parfaite

    # --- Sécurisation des limites biologiques ---
    resultats["Age"][i] = max(15, min(100, age))
    resultats["Fumeur"][i] = fumeur
    resultats["Temperature_C"][i] = max(35.5, min(41.5, temp))
    resultats["Leucocytes_G_L"][i] = max(1.0, min(35.0, leuco))
    resultats["D_Dimeres_ng_mL"][i] = max(10.0, ddimeres)
    resultats["CRP_mg_L"][i] = max(0.1, crp)
    resultats["Gaz_Sang_PaO2"][i] = max(35.0, min(105.0, pao2))
    resultats["Gaz_Sang_PaCO2"][i] = max(15.0, min(90.0, paco2))
    resultats["Radio_Thorax"][i] = radio

# ==========================================
# 5. FINALISATION ET SAUVEGARDE
# ==========================================
df_renforce = pd.DataFrame(resultats)

# Arrondis pour le réalisme médical
df_renforce["Age"] = df_renforce["Age"].astype(int)
df_renforce["Temperature_C"] = df_renforce["Temperature_C"].round(1)
df_renforce["Leucocytes_G_L"] = df_renforce["Leucocytes_G_L"].round(1)
df_renforce["D_Dimeres_ng_mL"] = df_renforce["D_Dimeres_ng_mL"].round(0)
df_renforce["CRP_mg_L"] = df_renforce["CRP_mg_L"].round(1)
df_renforce["Gaz_Sang_PaO2"] = df_renforce["Gaz_Sang_PaO2"].round(1)
df_renforce["Gaz_Sang_PaCO2"] = df_renforce["Gaz_Sang_PaCO2"].round(1)

# Fusion avec le Palier 1
df_complet = pd.merge(df_palier1, df_renforce, on="ID_Patient")

df_complet.to_csv(chemin_sortie, index=False)

print(f"-> Succès ! Fichier '{chemin_sortie.name}' généré.")
