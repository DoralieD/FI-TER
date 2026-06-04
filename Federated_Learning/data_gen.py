import pandas as pd
import numpy as np
import os
from pathlib import Path
import torch
from diffusers import AutoPipelineForText2Image
from tqdm import tqdm
import warnings

# On masque les petits avertissements de la console
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
np.random.seed(42)
n_samples = 100  # À augmenter (ex: 1000) quand tu seras prêt pour la grande génération

print(f"Génération de {n_samples} patients (Données + Images conditionnelles) en cours...")

# ==========================================
# 1. INITIALISATION DU MODÈLE MEDICAL
# ==========================================
print("Chargement du modèle spécialisé en radiographie...")

# Modèle public spécialisé dans les radios du thorax
model_id = "danyalmalik/stable-diffusion-chest-xray" 

pipe = AutoPipelineForText2Image.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=False
)

# Optimisations VRAM pour ta RTX 5060 (8 Go)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()

# ==========================================
# 2. GÉNÉRATION DES DONNÉES TABULAIRES
# ==========================================
donnees = {
    "ID_Patient": [f"PAT_{i:05d}" for i in range(1, n_samples + 1)],
    "FC": np.random.normal(85, 20, n_samples).astype(int),
    "Tension_Sys": np.random.normal(125, 25, n_samples).astype(int),
    "Tension_Dia": np.random.normal(80, 15, n_samples).astype(int),
    "FR": np.random.normal(16, 5, n_samples).astype(int),
    "Temp": np.round(np.random.normal(37.2, 0.8, n_samples), 1),
    "SpO2": np.random.normal(97, 4, n_samples).astype(int),
}

donnees["FC"] = np.clip(donnees["FC"], 40, 200)
donnees["Tension_Sys"] = np.clip(donnees["Tension_Sys"], 70, 220)
donnees["Tension_Dia"] = np.clip(donnees["Tension_Dia"], 40, 130)
donnees["FR"] = np.clip(donnees["FR"], 8, 40)
donnees["SpO2"] = np.clip(donnees["SpO2"], 70, 100)

df = pd.DataFrame(donnees)

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
df["Obs_TraumaPenetrant"] = np.random.choice([0, 1], p=[0.98, 0.02], size=n_samples)

def appliquer_bruit(condition, vrai_positif=0.79, faux_positif=0.12):
        hasard = np.random.rand(len(condition))
        return np.where(condition, hasard < vrai_positif, hasard < faux_positif).astype(int)

regle_cardio = (df["FC"] > 120) | (df["Tension_Sys"] > 160) | df["Obs_DouleurThorax"] | (df["Obs_Paleur"] & df["Obs_Sueurs"])
df["Verite_Cardio"] = appliquer_bruit(regle_cardio)

regle_respi = (df["SpO2"] < 92) | (df["FR"] > 25) | df["Obs_DetresseRespi"] | df["Obs_Cyanose"]
df["Verite_Respi"] = appliquer_bruit(regle_respi)

regle_infectieux = (df["Temp"] > 38.5) | ((df["Temp"] > 37.8) & df["Obs_Frissons"]) | df["Obs_Eruption"]
df["Verite_Infectieux"] = appliquer_bruit(regle_infectieux)

regle_neuro = df["Obs_Inconscient"] | df["Obs_Confusion"] | (df["Tension_Sys"] > 200)
df["Verite_Neuro"] = appliquer_bruit(regle_neuro)

# ==========================================
# 3. GÉNÉRATION DES IMAGES (AVEC PRESCRIPTION)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
dossier_syn = BASE_DIR / "Données_syn"
dossier_images = dossier_syn / "images_patients"

dossier_syn.mkdir(parents=True, exist_ok=True)
dossier_images.mkdir(parents=True, exist_ok=True)

chemins_images = []

print("Début du diagnostic et de la génération des images...")
for index, row in tqdm(df.iterrows(), total=n_samples):
    
    # --- LA PRESCRIPTION INTELLIGENTE ---
    # Par défaut, un patient a 5% de chances de faire une radio (routine/erreur)
    probabilite_radio = 0.05 
    
    if row["Verite_Respi"] == 1 or row["Verite_Infectieux"] == 1:
        probabilite_radio = 0.90  # Presque systématique
    elif row["Verite_Cardio"] == 1:
        probabilite_radio = 0.50  # Une fois sur deux
    
    # On lance les dés virtuels pour voir si on fait la radio
    if np.random.rand() < probabilite_radio:
        
        # Le patient DOIT passer une radio, on la génère.
        conditions = []
        if row["Verite_Cardio"] == 1:
            conditions.append("cardiomegaly")
        if row["Verite_Respi"] == 1:
            conditions.append("severe pneumonia, fluid in lungs")
        if row["Verite_Infectieux"] == 1:
            conditions.append("lung infection")
            
        if not conditions:
            prompt_medical = "Healthy, clear lungs"
        else:
            prompt_medical = "showing signs of " + " and ".join(conditions)

        # Prompt simplifié pour le modèle Fine-Tuné
        prompt_final = f"chest x-ray, {prompt_medical}"
        prompt_negatif = "text, cropped, weird anatomy"

        # Génération (30 étapes car le modèle est rapide)
        image = pipe(
            prompt=prompt_final, 
            negative_prompt=prompt_negatif,
            num_inference_steps=30, 
            guidance_scale=7.0
        ).images[0]
        
        nom_fichier = f"{row['ID_Patient']}_Radio.png"
        chemin_image_complet = dossier_images / nom_fichier
        image.save(chemin_image_complet)
        
        chemins_images.append(f"images_patients/{nom_fichier}")
        
    else:
        # Le patient n'a pas besoin de radio !
        chemins_images.append("Aucune")

df["Chemin_Image"] = chemins_images

# ==========================================
# 4. EXPORTATION FINALE
# ==========================================
chemin_csv_complet = dossier_syn / "dataset_prise_constante.csv"
df.to_csv(chemin_csv_complet, index=False)

print(f"\nTerminé ! {n_samples} patients générés.")
# Petite stat pour voir combien de radios ont été évitées :
radios_generees = len([c for c in chemins_images if c != "Aucune"])
print(f"Bilan : {radios_generees} radiographies générées sur {n_samples} patients ({(radios_generees/n_samples)*100:.1f}%).")
print(f"CSV sauvegardé sous : {chemin_csv_complet}")