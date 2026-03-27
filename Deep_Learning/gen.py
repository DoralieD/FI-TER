import pandas as pd
import numpy as np
from pathlib import Path

# Fixer la graine pour reproductibilité
np.random.seed(42)
n_samples = 200000

print("\n" + "="*50)
print(f"🏥 GÉNÉRATION FICHIER HÔPITAL ({n_samples} patients)")
print("="*50)

# ==========================================
# 1. PALIER 1 : CONSTANTES ET TRIAGE GLOBAL
# ==========================================
print("\nÉtape 1 : Création des patients à l'accueil (Palier 1)...")

donnees = {
    "ID_Patient": [f"PAT_{i:05d}" for i in range(1, n_samples + 1)],
    "Age": np.random.normal(55, 18, n_samples).clip(15, 99).astype(int), # Âge global pour tout l'hôpital
    "FC": np.random.normal(85, 20, n_samples).astype(int).clip(40, 200),
    "Tension_Sys": np.random.normal(125, 25, n_samples).astype(int).clip(70, 220),
    "Tension_Dia": np.random.normal(80, 15, n_samples).astype(int).clip(40, 130),
    "FR": np.random.normal(16, 5, n_samples).astype(int).clip(8, 40),
    "Temp": np.round(np.random.normal(37.2, 0.8, n_samples), 1),
    "SpO2": np.random.normal(97, 4, n_samples).astype(int).clip(70, 100),
}

df_hopital = pd.DataFrame(donnees)

# Observations (0 = Non, 1 = Oui)
df_hopital["Obs_Paleur"] = np.random.choice([0, 1], p=[0.75, 0.25], size=n_samples)
df_hopital["Obs_Cyanose"] = np.random.choice([0, 1], p=[0.92, 0.08], size=n_samples)
df_hopital["Obs_Sueurs"] = np.random.choice([0, 1], p=[0.80, 0.20], size=n_samples)
df_hopital["Obs_Inconscient"] = np.random.choice([0, 1], p=[0.96, 0.04], size=n_samples)
df_hopital["Obs_Confusion"] = np.random.choice([0, 1], p=[0.88, 0.12], size=n_samples)
df_hopital["Obs_Frissons"] = np.random.choice([0, 1], p=[0.85, 0.15], size=n_samples)
df_hopital["Obs_Hemorragie"] = np.random.choice([0, 1], p=[0.95, 0.05], size=n_samples)
df_hopital["Obs_DouleurThorax"] = np.random.choice([0, 1], p=[0.85, 0.15], size=n_samples)
df_hopital["Obs_DetresseRespi"] = np.random.choice([0, 1], p=[0.90, 0.10], size=n_samples)
df_hopital["Obs_Eruption"] = np.random.choice([0, 1], p=[0.93, 0.07], size=n_samples)
df_hopital["Obs_TraumaPenetrant"] = np.random.choice([0, 1], p=[0.98, 0.02], size=n_samples)

def appliquer_bruit(condition, vrai_positif=0.95, faux_positif=0.1):
    hasard = np.random.rand(len(condition))
    return np.where(condition, hasard < vrai_positif, hasard < faux_positif).astype(int)

# Triage
df_hopital["Verite_Cardio"] = appliquer_bruit((df_hopital["FC"] > 120) | (df_hopital["Tension_Sys"] > 160) | df_hopital["Obs_DouleurThorax"] | (df_hopital["Obs_Paleur"] & df_hopital["Obs_Sueurs"]))
df_hopital["Verite_Respi"] = appliquer_bruit((df_hopital["SpO2"] < 92) | (df_hopital["FR"] > 25) | df_hopital["Obs_DetresseRespi"] | df_hopital["Obs_Cyanose"])
df_hopital["Verite_Infectieux"] = appliquer_bruit((df_hopital["Temp"] > 38.5) | ((df_hopital["Temp"] > 37.8) & df_hopital["Obs_Frissons"]) | df_hopital["Obs_Eruption"])
df_hopital["Verite_Neuro"] = appliquer_bruit(df_hopital["Obs_Inconscient"] | df_hopital["Obs_Confusion"] | (df_hopital["Tension_Sys"] > 200))

# On indexe le Super Fichier par l'ID Patient pour faciliter les fusions magiques
df_hopital.set_index("ID_Patient", inplace=True)


# ==========================================
# 2. PALIER 2 : CARDIOLOGIE
# ==========================================
ids_cardio = df_hopital[df_hopital["Verite_Cardio"] == 1].index
n_cardio = len(ids_cardio)
print(f"Étape 2 : Ajout des examens Cardiologiques ({n_cardio} patients)...")

if n_cardio > 0:
    diagnostics_cardio = ["SCA_NSTEMI", "SCA_STEMI", "Insuffisance_Cardiaque_Aigue", "Trouble_Rythme_Grave", "Embolie_Pulmonaire_Grave", "Pericardite_Aigue"]
    y_cardio = np.random.choice(diagnostics_cardio, size=n_cardio, p=[0.30, 0.15, 0.25, 0.15, 0.08, 0.07])
    
    res_cardio = {
        "ID_Patient": ids_cardio,
        "Diabetique": np.random.choice([0, 1], size=n_cardio, p=[0.75, 0.25]),
        "Hypertendu": np.random.choice([0, 1], size=n_cardio, p=[0.55, 0.45]),
        "Fumeur": np.random.choice([0, 1], size=n_cardio, p=[0.65, 0.35]),
        "ECG_SusDecalage_ST": np.zeros(n_cardio, dtype=int),
        "ECG_Trouble_Rythme": np.zeros(n_cardio, dtype=int),
        "Troponine_ng_L": np.random.normal(8, 3, n_cardio).clip(0.5, None),
        "BNP_pg_mL": np.random.normal(60, 30, n_cardio).clip(5, None),
        "D_Dimeres_ng_mL": np.random.normal(300, 100, n_cardio).clip(50, None),
        "Creatinine_umol_L": np.random.normal(85, 15, n_cardio).clip(40, 500),
        "FEVG_pct": np.random.normal(60, 5, n_cardio).clip(10, 75),
        "Epanchement_Pericarde": np.zeros(n_cardio, dtype=int),
        "Diagnostic_Final_Cardio": y_cardio
    }
    
    # Modifications cliniques spécifiques
    for i in range(n_cardio):
        diag = y_cardio[i]
        if diag == "SCA_STEMI":
            res_cardio["ECG_SusDecalage_ST"][i] = 1
            res_cardio["Troponine_ng_L"][i] = max(100, np.random.normal(1800, 600))
        elif diag == "SCA_NSTEMI":
            res_cardio["Troponine_ng_L"][i] = max(20, np.random.normal(250, 150))
        elif diag == "Insuffisance_Cardiaque_Aigue":
            res_cardio["BNP_pg_mL"][i] = max(300, np.random.normal(1500, 500))
            res_cardio["FEVG_pct"][i] = max(10, np.random.normal(30, 8))
        elif diag == "Trouble_Rythme_Grave":
            res_cardio["ECG_Trouble_Rythme"][i] = 1
        elif diag == "Embolie_Pulmonaire_Grave":
            res_cardio["D_Dimeres_ng_mL"][i] = max(500, np.random.normal(5000, 1500))

    df_cardio = pd.DataFrame(res_cardio).set_index("ID_Patient")
    # FUSION MAGIQUE : On remplit les trous du Super Fichier sans écraser le reste
    df_hopital = df_hopital.combine_first(df_cardio)

# ==========================================
# 3. PALIER 2 : INFECTIOLOGIE
# ==========================================
ids_inf = df_hopital[df_hopital["Verite_Infectieux"] == 1].index
n_inf = len(ids_inf)
print(f"Étape 3 : Ajout des examens Infectiologiques ({n_inf} patients)...")

if n_inf > 0:
    diagnostics_inf = ["Infection_Urinaire_Grave", "Pneumonie_Communautaire", "Sepsis_Grave", "Gastroenterite_Bacterienne", "Cellulite_Infectieuse", "Meningite_Bacterienne"]
    y_inf = np.random.choice(diagnostics_inf, size=n_inf, p=[0.30, 0.25, 0.15, 0.15, 0.10, 0.05])
    
    res_inf = {
        "ID_Patient": ids_inf,
        "Immunodeprime": np.random.choice([0, 1], size=n_inf, p=[0.88, 0.12]),
        "Leucocytes_G_L": np.random.normal(13, 3, n_inf).clip(0.5, 50),
        "CRP_mg_L": np.random.normal(60, 30, n_inf).clip(0.1, 600),
        "PCT_ng_mL": np.random.normal(1.5, 1.0, n_inf).clip(0.05, 100),
        "Lactates_mmol_L": np.random.normal(1.5, 0.5, n_inf).clip(0.5, 15),
        "Bilirubine_umol_L": np.random.normal(12, 4, n_inf).clip(3, 200),
        "ECBU_Positif": np.zeros(n_inf, dtype=int),
        "Hemoc_Positive": np.zeros(n_inf, dtype=int),
        "Diagnostic_Final_Infectieux": y_inf
    }
    
    for i in range(n_inf):
        diag = y_inf[i]
        if diag == "Infection_Urinaire_Grave":
            res_inf["ECBU_Positif"][i] = np.random.choice([0, 1], p=[0.05, 0.95])
        elif diag == "Sepsis_Grave":
            res_inf["Lactates_mmol_L"][i] = max(2.5, np.random.normal(4.5, 1.5))
            res_inf["PCT_ng_mL"][i] = max(5.0, np.random.normal(25.0, 10.0))

    df_inf = pd.DataFrame(res_inf).set_index("ID_Patient")
    df_hopital = df_hopital.combine_first(df_inf)

# ==========================================
# 4. PALIER 2 : NEUROLOGIE
# ==========================================
ids_neuro = df_hopital[df_hopital["Verite_Neuro"] == 1].index
n_neuro = len(ids_neuro)
print(f"Étape 4 : Ajout des examens Neurologiques ({n_neuro} patients)...")

if n_neuro > 0:
    diagnostics_neuro = ["AVC_Ischemique", "AVC_Hemorragique", "Crise_Epilepsie", "Meningite", "Migraine_Severe"]
    y_neuro = np.random.choice(diagnostics_neuro, size=n_neuro, p=[0.30, 0.15, 0.20, 0.10, 0.25])
    
    res_neuro = {
        "ID_Patient": ids_neuro,
        "Score_Glasgow": np.random.choice([15, 14], size=n_neuro, p=[0.9, 0.1]),
        "Deficit_Moteur": np.zeros(n_neuro, dtype=int),
        "Trouble_Parole": np.zeros(n_neuro, dtype=int),
        "Raideur_Nuque": np.zeros(n_neuro, dtype=int),
        "Scanner_Cerebral": np.zeros(n_neuro, dtype=int),
        "Diagnostic_Final_Neuro": y_neuro
    }
    
    for i in range(n_neuro):
        diag = y_neuro[i]
        if diag == "AVC_Ischemique":
            res_neuro["Scanner_Cerebral"][i] = np.random.choice([0, 1], p=[0.3, 0.7])
            res_neuro["Deficit_Moteur"][i] = np.random.choice([0, 1, 2], p=[0.1, 0.4, 0.5])
        elif diag == "AVC_Hemorragique":
            res_neuro["Scanner_Cerebral"][i] = 2
            res_neuro["Score_Glasgow"][i] = np.random.choice([12, 8, 5], p=[0.3, 0.4, 0.3])
        elif diag == "Meningite":
            res_neuro["Raideur_Nuque"][i] = np.random.choice([0, 1], p=[0.1, 0.9])

    df_neuro = pd.DataFrame(res_neuro).set_index("ID_Patient")
    df_hopital = df_hopital.combine_first(df_neuro)

# ==========================================
# 5. PALIER 2 : RESPIRATOIRE
# ==========================================
ids_respi = df_hopital[df_hopital["Verite_Respi"] == 1].index
n_respi = len(ids_respi)
print(f"Étape 5 : Ajout des examens Respiratoires ({n_respi} patients)...")

if n_respi > 0:
    diagnostics_respi = ["Pneumonie_Bacterienne", "Infection_Virale_Severe", "Crise_Asthme", "Exacerbation_BPCO", "Embolie_Pulmonaire", "Pneumothorax", "Hyperventilation_Angoisse"]
    y_respi = np.random.choice(diagnostics_respi, size=n_respi, p=[0.20, 0.25, 0.15, 0.15, 0.10, 0.05, 0.10])
    
    res_respi = {
        "ID_Patient": ids_respi,
        "Gaz_Sang_PaO2": np.random.normal(95, 5, n_respi).clip(35, 105),
        "Gaz_Sang_PaCO2": np.random.normal(40, 2, n_respi).clip(15, 90),
        "Radio_Thorax": np.zeros(n_respi, dtype=int),
        "Diagnostic_Final_Respi": y_respi
    }
    
    for i in range(n_respi):
        diag = y_respi[i]
        if diag == "Pneumothorax":
            res_respi["Radio_Thorax"][i] = 2
        elif diag == "Exacerbation_BPCO":
            res_respi["Radio_Thorax"][i] = 3
            res_respi["Gaz_Sang_PaCO2"][i] = max(45, np.random.normal(60, 10))

    df_respi = pd.DataFrame(res_respi).set_index("ID_Patient")
    df_hopital = df_hopital.combine_first(df_respi)


# ==========================================
# 6. SAUVEGARDE DU SUPER FICHIER
# ==========================================
print("\nÉtape 6 : Finalisation et Exportation...")

# On remet l'ID Patient en tant que colonne normale
df_hopital.reset_index(inplace=True)

# Définition du chemin de sauvegarde
BASE_DIR = Path(__file__).resolve().parent
dossier_syn = BASE_DIR / "Données_syn"
dossier_syn.mkdir(parents=True, exist_ok=True)

chemin_super_fichier = dossier_syn / "super_fichier_hopital.csv"
df_hopital.to_csv(chemin_super_fichier, index=False)

print("\n" + "="*50)
print(f"Fichier sauvegardé : {chemin_super_fichier.name}")
print(f"-> Nombre total de patients : {len(df_hopital)}")
print(f"-> Nombre de colonnes (Constantes + Examens de toutes les spécialités) : {len(df_hopital.columns)}")
print("="*50)