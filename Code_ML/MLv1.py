import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. GÉNÉRATION DES DONNÉES SYNTHÉTIQUES (Multi-Observations)
# ==========================================
np.random.seed(4232)
n_samples = 10000

# Capteurs IoMT (Données continues)
donnees = {
    "ID_Patient": [f"PAT_{i:04d}" for i in range(1, n_samples + 1)],
    "Frequence_Cardiaque": np.random.randint(55, 160, n_samples),
    "Saturation_O2": np.random.randint(75, 100, n_samples),
    "Temperature": np.random.uniform(35.5, 41.0, n_samples),
}
df = pd.DataFrame(donnees)

# Observations physiques (Checkboxes médicales : 0 = Non, 1 = Oui)
# L'utilisation de probabilités (p) permet de simuler la rareté de certains symptômes
df["Obs_Paleur"] = np.random.choice([0, 1], p=[0.7, 0.3], size=n_samples)
df["Obs_Cyanose"] = np.random.choice([0, 1], p=[0.9, 0.1], size=n_samples) # Plus rare
df["Obs_Sueurs"] = np.random.choice([0, 1], p=[0.8, 0.2], size=n_samples)
df["Obs_Inconscient"] = np.random.choice([0, 1], p=[0.95, 0.05], size=n_samples) # Très rare
df["Obs_Confusion"] = np.random.choice([0, 1], p=[0.85, 0.15], size=n_samples)
df["Obs_Frissons"] = np.random.choice([0, 1], p=[0.8, 0.2], size=n_samples)

# ==========================================
# 2. CRÉATION DE LA VÉRITÉ TERRAIN (Avec incertitude médicale / Bruit)
# ==========================================
# Fonction pour simuler la réalité : la règle de base n'est vraie qu'à X%
def appliquer_bruit_medical(condition_logique, vrai_positif=0.85, faux_positif=0.10):
    """
    vrai_positif (0.85) : Si le patient a les symptômes, il a 85% de chances d'avoir vraiment la maladie.
    faux_positif (0.10) : Même sans les symptômes typiques, il a 10% de chances d'avoir la maladie (cas atypique).
    """
    hasard = np.random.rand(len(condition_logique))
    # On applique les pourcentages avec np.where
    resultat = np.where(condition_logique, hasard < vrai_positif, hasard < faux_positif)
    return resultat.astype(int)

# --- Définition des règles théoriques ---
regle_cardio = (df["Frequence_Cardiaque"] > 110) & (df["Obs_Paleur"] | df["Obs_Sueurs"] | df["Obs_Inconscient"])
regle_respi = (df["Saturation_O2"] < 90) & (df["Obs_Cyanose"] | df["Obs_Confusion"] | df["Obs_Inconscient"])
regle_infectieux = (df["Temperature"] > 39.0) | ((df["Temperature"] > 38.0) & df["Obs_Frissons"] & df["Obs_Sueurs"])
regle_neuro = df["Obs_Inconscient"] | df["Obs_Confusion"]

# --- Application du bruit pour créer la vérité terrain ---
# On ajoute volontairement de l'erreur humaine/biologique !
df["Verite_Cardio"] = appliquer_bruit_medical(regle_cardio, vrai_positif=0.85, faux_positif=0.05)
df["Verite_Respi"] = appliquer_bruit_medical(regle_respi, vrai_positif=0.90, faux_positif=0.08)
df["Verite_Infectieux"] = appliquer_bruit_medical(regle_infectieux, vrai_positif=0.80, faux_positif=0.15)
df["Verite_Neuro"] = appliquer_bruit_medical(regle_neuro, vrai_positif=0.95, faux_positif=0.02)

# Sauvegarde
df.to_csv("patients_triage_realistes.csv", index=False)
print("Fichier CSV généré avec de l'incertitude médicale !")


# ==========================================
# 3. PRÉPARATION MACHINE LEARNING
# ==========================================
# On sélectionne toutes les colonnes sauf les IDs et la vérité terrain
colonnes_features = [col for col in df.columns if col not in ["ID_Patient"] and not col.startswith("Verite_")]
X = df[colonnes_features]

# Séparation des données
X_train, X_test, df_train, df_test = train_test_split(X, df, test_size=0.2, random_state=4232)


# ==========================================
# 4. ARCHITECTURE DU GRAPHE DE TRIAGE
# ==========================================
class NoeudTriageMulti:
    def __init__(self, nom_maladie):
        self.nom = nom_maladie
        # Plus besoin de 'Pipeline' complexe car tout est déjà en chiffres (0 ou 1) !
        self.modele = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=4232)

    def entrainer(self, X_train, y_train):
        self.modele.fit(X_train, y_train)

    def predire_binaire(self, X_data, seuil=0.5):
        return (self.modele.predict_proba(X_data)[:, 1] >= seuil).astype(int)

class GraphePalier1:
    def __init__(self):
        self.noeuds = {
            "Cardiovasculaire": NoeudTriageMulti("Cardiovasculaire"),
            "Respiratoire": NoeudTriageMulti("Respiratoire"),
            "Infectieux": NoeudTriageMulti("Infectieux"),
            "Neurologique": NoeudTriageMulti("Neurologique")
        }
    
    def entrainer_graphe(self, X_train, df_train):
        print("Entraînement des nœuds sur les combinaisons de symptômes...")
        for nom in self.noeuds.keys():
            colonne_cible = f"Verite_{nom[:6] if nom == 'Neurologique' else nom.split(' ')[0]}" # Petite astuce de formatage
            
            # Gestion exacte du nom des colonnes cibles
            if nom == "Cardiovasculaire": cible = "Verite_Cardio"
            elif nom == "Respiratoire": cible = "Verite_Respi"
            elif nom == "Infectieux": cible = "Verite_Infectieux"
            elif nom == "Neurologique": cible = "Verite_Neuro"

            self.noeuds[nom].entrainer(X_train, df_train[cible])

    def evaluer(self, X_test, df_test):
        print("\n=== PERFORMANCES SUR LES PATIENTS DE TEST ===")
        for nom, noeud in self.noeuds.items():
            if nom == "Cardiovasculaire": cible = "Verite_Cardio"
            elif nom == "Respiratoire": cible = "Verite_Respi"
            elif nom == "Infectieux": cible = "Verite_Infectieux"
            elif nom == "Neurologique": cible = "Verite_Neuro"

            predictions = noeud.predire_binaire(X_test)
            precision = accuracy_score(df_test[cible], predictions)
            print(f"Nœud {nom.ljust(16)} : Précision {precision * 100:.2f}%")


# Exécution
mon_graphe = GraphePalier1()
mon_graphe.entrainer_graphe(X_train, df_train)
mon_graphe.evaluer(X_test, df_test)

# ==========================================
# 5. TEST SUR UN CAS COMPLEXE MULTI-SYMPTÔMES
# ==========================================
print("\n--- ANALYSE D'UN CAS GRAVE (Multi-Symptômes) ---")
# On simule un patient qui fait un arrêt cardiaque et respiratoire (cumul des symptômes)
patient_critique = pd.DataFrame([{
    "Frequence_Cardiaque": 140,
    "Saturation_O2": 82,
    "Temperature": 36.5,
    "Obs_Paleur": 1,        # OUI
    "Obs_Cyanose": 1,       # OUI
    "Obs_Sueurs": 0,        # NON
    "Obs_Inconscient": 1,   # OUI (Point critique qui va déclencher plusieurs branches)
    "Obs_Confusion": 0,     # NON
    "Obs_Frissons": 0       # NON
}])

print("Symptômes du patient : Tachycardie, Hypoxie, Pâleur, Cyanose, INCONSCIENT.")
for nom, noeud in mon_graphe.noeuds.items():
    proba = noeud.modele.predict_proba(patient_critique)[0][1]
    alerte = "⚠️ ALERTE" if proba > 0.4 else "OK"
    print(f"Risque {nom.ljust(16)} : {proba*100:05.1f}% -> {alerte}")