import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import truncnorm

# Fix seed for reproducibility
np.random.seed(42)
n_samples = 500000

print("\n" + "="*50)
print(f"🏥 GENERATING HOSPITAL DATASET ({n_samples} patients)")
print("="*50)

# Fonction pour générer une loi normale tronquée (évite les piques du .clip)
def get_truncnorm(mean, std, clip_min, clip_max, size):
    clip_max = np.inf if clip_max is None else clip_max
    a, b = (clip_min - mean) / std, (clip_max - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

# ==========================================
# 1. TIER 1: VITALS & GLOBAL TRIAGE
# ==========================================
print("\nStep 1: Creating reception patients (Tier 1)...")

data = {
    "Patient_ID": [f"PAT_{i:07d}" for i in range(1, n_samples + 1)],
    "Age": get_truncnorm(55, 18, 15, 99, n_samples).astype(int),
    "HR": get_truncnorm(85, 20, 40, 200, n_samples).astype(int), 
    "Sys_BP": get_truncnorm(125, 25, 70, 220, n_samples).astype(int), 
    "Dia_BP": get_truncnorm(80, 15, 40, 130, n_samples).astype(int), 
    "RR": get_truncnorm(16, 5, 8, 40, n_samples).astype(int), 
    "Temp": np.round(get_truncnorm(37.2, 0.8, 34.0, 42.0, n_samples), 1),
    "SpO2": get_truncnorm(97, 4, 70, 100, n_samples).astype(int),
}

df_hospital = pd.DataFrame(data)

# Observations (0 = No, 1 = Yes)
df_hospital["Obs_Pallor"] = np.random.choice([0, 1], p=[0.75, 0.25], size=n_samples)
df_hospital["Obs_Cyanosis"] = np.random.choice([0, 1], p=[0.92, 0.08], size=n_samples)
df_hospital["Obs_Sweating"] = np.random.choice([0, 1], p=[0.80, 0.20], size=n_samples)
df_hospital["Obs_Unconscious"] = np.random.choice([0, 1], p=[0.96, 0.04], size=n_samples)
df_hospital["Obs_Confusion"] = np.random.choice([0, 1], p=[0.88, 0.12], size=n_samples)
df_hospital["Obs_Chills"] = np.random.choice([0, 1], p=[0.85, 0.15], size=n_samples)
df_hospital["Obs_Hemorrhage"] = np.random.choice([0, 1], p=[0.95, 0.05], size=n_samples)
df_hospital["Obs_ChestPain"] = np.random.choice([0, 1], p=[0.85, 0.15], size=n_samples)
df_hospital["Obs_RespiratoryDistress"] = np.random.choice([0, 1], p=[0.90, 0.10], size=n_samples)
df_hospital["Obs_Rash"] = np.random.choice([0, 1], p=[0.93, 0.07], size=n_samples)
df_hospital["Obs_PenetratingTrauma"] = np.random.choice([0, 1], p=[0.98, 0.02], size=n_samples)

def apply_noise(condition, true_positive=0.95, false_positive=0.1):
    chance = np.random.rand(len(condition))
    return np.where(condition, chance < true_positive, chance < false_positive).astype(int)

# Triage (Ground Truth)
df_hospital["Truth_Cardio"] = apply_noise((df_hospital["HR"] > 120) | (df_hospital["Sys_BP"] > 160) | df_hospital["Obs_ChestPain"] | (df_hospital["Obs_Pallor"] & df_hospital["Obs_Sweating"]))
df_hospital["Truth_Respi"] = apply_noise((df_hospital["SpO2"] < 92) | (df_hospital["RR"] > 25) | df_hospital["Obs_RespiratoryDistress"] | df_hospital["Obs_Cyanosis"])
df_hospital["Truth_Infectious"] = apply_noise((df_hospital["Temp"] > 38.5) | ((df_hospital["Temp"] > 37.8) & df_hospital["Obs_Chills"]) | df_hospital["Obs_Rash"])
df_hospital["Truth_Neuro"] = apply_noise(df_hospital["Obs_Unconscious"] | df_hospital["Obs_Confusion"] | (df_hospital["Sys_BP"] > 200))

# Index the Super File by Patient ID for easy merging later
df_hospital.set_index("Patient_ID", inplace=True)

# ==========================================
# 2. TIER 2: CARDIOLOGY
# ==========================================
ids_cardio = df_hospital[df_hospital["Truth_Cardio"] == 1].index
n_cardio = len(ids_cardio)
print(f"Step 2: Adding Cardiology exams ({n_cardio} patients)...")

if n_cardio > 0:
    diagnostics_cardio = ["NSTEMI", "STEMI", "Acute_Heart_Failure", "Severe_Arrhythmia", "Severe_Pulmonary_Embolism", "Acute_Pericarditis"]
    y_cardio = np.random.choice(diagnostics_cardio, size=n_cardio, p=[0.30, 0.15, 0.25, 0.15, 0.08, 0.07])
    
    res_cardio = {
        "Patient_ID": ids_cardio,
        "Diabetic": np.random.choice([0, 1], size=n_cardio, p=[0.75, 0.25]),
        "Hypertensive": np.random.choice([0, 1], size=n_cardio, p=[0.55, 0.45]),
        "Smoker": np.random.choice([0, 1], size=n_cardio, p=[0.65, 0.35]),
        "ECG_ST_Elevation": np.zeros(n_cardio, dtype=int),
        "ECG_Arrhythmia": np.zeros(n_cardio, dtype=int),
        "Troponin_ng_L": get_truncnorm(8, 3, 0.5, None, n_cardio),
        "BNP_pg_mL": get_truncnorm(60, 30, 5, None, n_cardio),
        "D_Dimers_ng_mL": get_truncnorm(300, 100, 50, None, n_cardio),
        "Creatinine_umol_L": get_truncnorm(85, 15, 40, 500, n_cardio),
        "LVEF_pct": get_truncnorm(60, 5, 10, 75, n_cardio), 
        "Pericardial_Effusion": np.zeros(n_cardio, dtype=int),
        "Final_Diagnosis_Cardio": y_cardio
    }
    
    for i in range(n_cardio):
        diag = y_cardio[i]
        if diag == "STEMI":
            res_cardio["ECG_ST_Elevation"][i] = 1
            res_cardio["Troponin_ng_L"][i] = max(100, get_truncnorm(1800, 600, 100, None, 1)[0])
        elif diag == "NSTEMI":
            res_cardio["Troponin_ng_L"][i] = max(20, get_truncnorm(250, 150, 20, None, 1)[0])
        elif diag == "Acute_Heart_Failure":
            res_cardio["BNP_pg_mL"][i] = max(300, get_truncnorm(1500, 500, 300, None, 1)[0])
            res_cardio["LVEF_pct"][i] = max(10, get_truncnorm(30, 8, 10, 75, 1)[0])
        elif diag == "Severe_Arrhythmia":
            res_cardio["ECG_Arrhythmia"][i] = 1
        elif diag == "Severe_Pulmonary_Embolism":
            res_cardio["D_Dimers_ng_mL"][i] = max(500, get_truncnorm(5000, 1500, 500, None, 1)[0])

    df_cardio = pd.DataFrame(res_cardio).set_index("Patient_ID")
    df_hospital = df_hospital.combine_first(df_cardio)

# ==========================================
# 3. TIER 2: INFECTIOUS DISEASES
# ==========================================
ids_inf = df_hospital[df_hospital["Truth_Infectious"] == 1].index
n_inf = len(ids_inf)
print(f"Step 3: Adding Infectious Disease exams ({n_inf} patients)...")

if n_inf > 0:
    diagnostics_inf = ["Severe_UTI", "Community_Acquired_Pneumonia", "Severe_Sepsis", "Bacterial_Gastroenteritis", "Infectious_Cellulitis", "Bacterial_Meningitis"]
    y_inf = np.random.choice(diagnostics_inf, size=n_inf, p=[0.30, 0.25, 0.15, 0.15, 0.10, 0.05])
    
    res_inf = {
        "Patient_ID": ids_inf,
        "Immunocompromised": np.random.choice([0, 1], size=n_inf, p=[0.88, 0.12]),
        "Leukocytes_G_L": get_truncnorm(13, 3, 0.5, 50, n_inf),
        "CRP_mg_L": get_truncnorm(60, 30, 0.1, 600, n_inf),
        "PCT_ng_mL": get_truncnorm(1.5, 1.0, 0.05, 100, n_inf),
        "Lactates_mmol_L": get_truncnorm(1.5, 0.5, 0.5, 15, n_inf),
        "Bilirubin_umol_L": get_truncnorm(12, 4, 3, 200, n_inf),
        "Urine_Culture_Positive": np.zeros(n_inf, dtype=int),
        "Blood_Culture_Positive": np.zeros(n_inf, dtype=int),
        "Final_Diagnosis_Infectious": y_inf
    }
    
    for i in range(n_inf):
        diag = y_inf[i]
        if diag == "Severe_UTI":
            res_inf["Urine_Culture_Positive"][i] = np.random.choice([0, 1], p=[0.05, 0.95])
        elif diag == "Severe_Sepsis":
            res_inf["Lactates_mmol_L"][i] = max(2.5, get_truncnorm(4.5, 1.5, 2.5, 15, 1)[0])
            res_inf["PCT_ng_mL"][i] = max(5.0, get_truncnorm(25.0, 10.0, 5.0, 100, 1)[0])

    df_inf = pd.DataFrame(res_inf).set_index("Patient_ID")
    df_hospital = df_hospital.combine_first(df_inf)

# ==========================================
# 4. TIER 2: NEUROLOGY
# ==========================================
ids_neuro = df_hospital[df_hospital["Truth_Neuro"] == 1].index
n_neuro = len(ids_neuro)
print(f"Step 4: Adding Neurology exams ({n_neuro} patients)...")

if n_neuro > 0:
    diagnostics_neuro = ["Ischemic_Stroke", "Hemorrhagic_Stroke", "Epileptic_Seizure", "Meningitis", "Severe_Migraine"]
    y_neuro = np.random.choice(diagnostics_neuro, size=n_neuro, p=[0.30, 0.15, 0.20, 0.10, 0.25])
    
    res_neuro = {
        "Patient_ID": ids_neuro,
        "Glasgow_Score": np.random.choice([15, 14], size=n_neuro, p=[0.9, 0.1]),
        "Motor_Deficit": np.zeros(n_neuro, dtype=int),
        "Speech_Disorder": np.zeros(n_neuro, dtype=int),
        "Neck_Stiffness": np.zeros(n_neuro, dtype=int),
        "Brain_CT": np.zeros(n_neuro, dtype=int),
        "Final_Diagnosis_Neuro": y_neuro
    }
    
    for i in range(n_neuro):
        diag = y_neuro[i]
        if diag == "Ischemic_Stroke":
            res_neuro["Brain_CT"][i] = np.random.choice([0, 1], p=[0.3, 0.7])
            res_neuro["Motor_Deficit"][i] = np.random.choice([0, 1, 2], p=[0.1, 0.4, 0.5])
        elif diag == "Hemorrhagic_Stroke":
            res_neuro["Brain_CT"][i] = 2
            res_neuro["Glasgow_Score"][i] = np.random.choice([12, 8, 5], p=[0.3, 0.4, 0.3])
        elif diag == "Meningitis":
            res_neuro["Neck_Stiffness"][i] = np.random.choice([0, 1], p=[0.1, 0.9])

    df_neuro = pd.DataFrame(res_neuro).set_index("Patient_ID")
    df_hospital = df_hospital.combine_first(df_neuro)

# ==========================================
# 5. TIER 2: RESPIRATORY
# ==========================================
ids_respi = df_hospital[df_hospital["Truth_Respi"] == 1].index
n_respi = len(ids_respi)
print(f"Step 5: Adding Respiratory exams ({n_respi} patients)...")

if n_respi > 0:
    diagnostics_respi = ["Bacterial_Pneumonia", "Severe_Viral_Infection", "Asthma_Attack", "COPD_Exacerbation", "Pulmonary_Embolism", "Pneumothorax", "Anxiety_Hyperventilation"]
    y_respi = np.random.choice(diagnostics_respi, size=n_respi, p=[0.20, 0.25, 0.15, 0.15, 0.10, 0.05, 0.10])
    
    res_respi = {
        "Patient_ID": ids_respi,
        "Blood_Gas_PaO2": get_truncnorm(95, 5, 35, 105, n_respi),
        "Blood_Gas_PaCO2": get_truncnorm(40, 2, 15, 90, n_respi),
        "Chest_XRay": np.zeros(n_respi, dtype=int),
        "Final_Diagnosis_Respi": y_respi
    }
    
    for i in range(n_respi):
        diag = y_respi[i]
        if diag == "Pneumothorax":
            res_respi["Chest_XRay"][i] = 2
        elif diag == "COPD_Exacerbation":
            res_respi["Chest_XRay"][i] = 3
            res_respi["Blood_Gas_PaCO2"][i] = max(45, get_truncnorm(60, 10, 45, 90, 1)[0])

    df_respi = pd.DataFrame(res_respi).set_index("Patient_ID")
    df_hospital = df_hospital.combine_first(df_respi)

# ==========================================
# 6. SAVING THE SUPER FILE
# ==========================================
print("\nStep 6: Finalization and Export...")
df_hospital.reset_index(inplace=True)

BASE_DIR = Path(__file__).resolve().parent
synthetic_dir = BASE_DIR / "Synthetic_Data"
synthetic_dir.mkdir(parents=True, exist_ok=True)

super_file_path = synthetic_dir / "super_hospital_file.csv"
df_hospital.to_csv(super_file_path, index=False)

print("\n" + "="*50)
print(f"File saved: {super_file_path.name}")
print(f"-> Total patients: {len(df_hospital)}")
print("="*50)