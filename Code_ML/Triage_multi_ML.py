import pandas as pd
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import importlib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CONFIGURATION DES CHEMINS ET CHARGEMENT
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
chemin_donnees = BASE_DIR / "Données_syn" / "dataset_prise_constante.csv"
dossier_tri = BASE_DIR / "Données_syn" / "Données_triée"
dossier_graphes = BASE_DIR / "Dossier_graphiques" / "Triage1"

dossier_tri.mkdir(parents=True, exist_ok=True)
dossier_graphes.mkdir(parents=True, exist_ok=True)

print("Chargement de la base de données...")
try:
    df = pd.read_csv(chemin_donnees)
except FileNotFoundError:
    print(f"Erreur : Le fichier {chemin_donnees} est introuvable.")
    exit()

colonnes_cibles = ["Verite_Cardio", "Verite_Respi", "Verite_Infectieux", "Verite_Neuro"]
y = df[colonnes_cibles]
X = df.drop(columns=["ID_Patient"] + colonnes_cibles, errors="ignore")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Mise à l'échelle des données (StandardScaler)...")
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# ==========================================
# 2. CHARGEMENT DYNAMIQUE DES MODÈLES (DEPUIS JSON)
# ==========================================
chemin_config = BASE_DIR / "models_ml.json"
modeles_a_tester = {}

def instancier_modele(config_dict):
    module = importlib.import_module(config_dict["module"])
    ModelClass = getattr(module, config_dict["class"])
    params = config_dict.get("params", {}).copy()
    
    if "estimators_config" in config_dict:
        estimators = []
        for nom, conf in config_dict["estimators_config"].items():
            estimators.append((nom, instancier_modele(conf)))
        params["estimators"] = estimators
        
    if "final_estimator_config" in config_dict:
        params["final_estimator"] = instancier_modele(config_dict["final_estimator_config"])
        
    return ModelClass(**params)

try:
    with open(chemin_config, "r") as f:
        config = json.load(f)
        
    for nom_algo, config_algo in config.items():
        modeles_a_tester[nom_algo] = instancier_modele(config_algo)
        
    print(f"{len(modeles_a_tester)} modèles chargés avec succès depuis la configuration.")
except FileNotFoundError:
    print(f"Erreur : Le fichier de configuration '{chemin_config}' est introuvable.")
    exit()

# ==========================================
# 3. L'ARCHITECTURE DU GRAPHE (L'IA)
# ==========================================
class NoeudSpecialiste:
    def __init__(self, nom, modele_base):
        self.nom = nom
        self.modele = clone(modele_base)

    def entrainer(self, X_train, y_train):
        self.modele.fit(X_train, y_train)

class TriagePalier1:
    def __init__(self, nom_algo, modele_base):
        self.nom_algo = nom_algo
        self.noeuds = {
            "Cardio": NoeudSpecialiste("Cardio", modele_base),
            "Respi": NoeudSpecialiste("Respi", modele_base),
            "Infectieux": NoeudSpecialiste("Infectieux", modele_base),
            "Neuro": NoeudSpecialiste("Neuro", modele_base)
        }
        self.temps_entrainement = {}
        self.temps_prediction_echantillon = {}
        self.roc_auc = {}
        self.fpr = {}
        self.tpr = {}

    def entrainer_systeme(self, X_train, y_train):
        print(f"\n[{self.nom_algo}] Entraînement en cours...")
        for nom, noeud in self.noeuds.items():
            start = time.time()
            noeud.entrainer(X_train, y_train[f"Verite_{nom}"])
            end = time.time()
            self.temps_entrainement[nom] = end - start
        print(f"[{self.nom_algo}] Entraînement terminé.")

    def tester_systeme(self, X_test, y_test):
        for nom, noeud in self.noeuds.items():
            y_vrai = y_test[f"Verite_{nom}"]
            
            start = time.time()
            predictions = noeud.modele.predict(X_test)
            end = time.time()
            
            self.temps_prediction_echantillon[nom] = (end - start) / len(X_test)
            
            if hasattr(noeud.modele, "predict_proba"):
                probabilites = noeud.modele.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_vrai, probabilites)
                self.fpr[nom] = fpr
                self.tpr[nom] = tpr
                self.roc_auc[nom] = auc(fpr, tpr)
            else:
                self.roc_auc[nom] = 0.0 

# ==========================================
# 4. EXÉCUTION DE TOUS LES MODÈLES
# ==========================================
resultats_globaux = {}

for nom_algo, modele_base in modeles_a_tester.items():
    hopital_ia = TriagePalier1(nom_algo, modele_base)
    hopital_ia.entrainer_systeme(X_train, y_train)
    hopital_ia.tester_systeme(X_test, y_test)
    resultats_globaux[nom_algo] = hopital_ia

# ==========================================
# 5. SYSTÈME DE VOTE ET GÉNÉRATION DES CSV UNIQUES
# ==========================================
print("\n" + "="*50)
print("5. VOTE MAJORITAIRE DES MODÈLES ET GÉNÉRATION DES CSV")
print("="*50)

specialites = ["Cardio", "Respi", "Infectieux", "Neuro"]
nombre_modeles = len(modeles_a_tester)
# Pour déclencher l'alerte, on demande au moins la moitié des votes (ex: 3 sur 6)
seuil_majorite = nombre_modeles / 2.0 

for nom in specialites:
    # Tableau rempli de 0 pour accumuler les votes
    votes_totaux = np.zeros(len(X_test))
    
    # On récolte les votes (0 ou 1) de chaque algorithme
    for nom_algo, hopital_ia in resultats_globaux.items():
        noeud = hopital_ia.noeuds[nom]
        predictions_binaires = noeud.modele.predict(X_test)
        votes_totaux += predictions_binaires
        
    # Le patient est retenu si la somme de ses votes atteint le seuil
    masque_positif = votes_totaux >= seuil_majorite
    
    # Extraction des vraies données depuis 'df'
    patients_a_risque = df.loc[X_test.index[masque_positif]].copy()
    
    # Ajout du score de consensus pour le médecin
    patients_a_risque.insert(1, f"Accord_Modeles_Sur_{nombre_modeles}", votes_totaux[masque_positif].astype(int))
    
    # Sauvegarde dans un seul dossier par spécialité
    chemin_palier = dossier_tri / f"Dossier_Palier2_{nom}"
    chemin_palier.mkdir(parents=True, exist_ok=True)
    
    chemin_final = chemin_palier / f"patients_{nom.lower()}.csv"
    patients_a_risque.to_csv(chemin_final, index=False)
    
    print(f"-> {len(patients_a_risque)} patients orientés en {nom} (sauvegardé dans {chemin_palier.name})")

# ==========================================
# 6. GÉNÉRATION DES GRAPHIQUES DYNAMIQUES ET ROC
# ==========================================
print("\n" + "="*50)
print("6. GÉNÉRATION DES GRAPHIQUES DE COMPARAISON")
print("="*50)

x = np.arange(len(specialites))
largeur_totale = 0.8
largeur_barre = largeur_totale / nombre_modeles

# --- Graphique 1 : Comparaison des scores AUC ---
fig, ax = plt.subplots(figsize=(12, 6))
for i, (nom_algo, hopital_ia) in enumerate(resultats_globaux.items()):
    auc_scores = [hopital_ia.roc_auc.get(sp, 0) for sp in specialites]
    position_x = x - (largeur_totale / 2) + (i * largeur_barre) + (largeur_barre / 2)
    ax.bar(position_x, auc_scores, largeur_barre, label=nom_algo)

ax.set_ylabel("Score AUC (proche de 1 = parfait)")
ax.set_title("Comparaison des performances (AUC) par Algorithme")
ax.set_xticks(x)
ax.set_xticklabels(specialites)
ax.set_ylim(0.0, 1.05)
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
chemin_auc = dossier_graphes / "comparaison_auc_dynamique.png"
plt.savefig(chemin_auc)
plt.close()
print(f"-> Graphique généré : {chemin_auc.name}")

# --- Graphique 2 : Courbes ROC Individuelles par Algorithme ---
couleurs = {'Cardio': '#E63946', 'Respi': '#457B9D', 'Infectieux': '#2A9D8F', 'Neuro': '#8338EC'}

for nom_algo, hopital_ia in resultats_globaux.items():
    plt.figure(figsize=(10, 8))
    
    for nom in specialites:
        if nom in hopital_ia.fpr and len(hopital_ia.fpr[nom]) > 0:
            plt.plot(hopital_ia.fpr[nom], hopital_ia.tpr[nom], color=couleurs.get(nom, 'black'), lw=2,
                     label=f"{nom} (AUC = {hopital_ia.roc_auc[nom]:.3f})")

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taux de Faux Positifs (FPR)")
    plt.ylabel("Taux de Vrais Positifs (TPR)")
    plt.title(f"Courbes ROC - Algorithme : {nom_algo}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    chemin_roc = dossier_graphes / f"courbes_roc_{nom_algo}.png"
    plt.savefig(chemin_roc)
    plt.close()
    print(f"-> Graphique généré : {chemin_roc.name}")

# --- Graphique 3 : Comparaison des temps d'entraînement ---
fig, ax = plt.subplots(figsize=(12, 6))
for i, (nom_algo, hopital_ia) in enumerate(resultats_globaux.items()):
    temps = [hopital_ia.temps_entrainement.get(sp, 0) for sp in specialites]
    position_x = x - (largeur_totale / 2) + (i * largeur_barre) + (largeur_barre / 2)
    ax.bar(position_x, temps, largeur_barre, label=nom_algo)

ax.set_ylabel("Temps d'entraînement (secondes)")
ax.set_title("Comparaison des Temps d'Entraînement")
ax.set_xticks(x)
ax.set_xticklabels(specialites)
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
chemin_temps = dossier_graphes / "comparaison_temps_dynamique.png"
plt.savefig(chemin_temps)
plt.close()
print(f"-> Graphique généré : {chemin_temps.name}")

print("\nOpération de triage par vote majoritaire terminée.")

# --- Graphique 4 : Comparaison des temps de prédiction (par échantillon) ---
fig, ax = plt.subplots(figsize=(12, 6))
for i, (nom_algo, hopital_ia) in enumerate(resultats_globaux.items()):
    # On récupère les temps de prédiction pour chaque spécialité
    temps_pred = [hopital_ia.temps_prediction_echantillon.get(sp, 0) for sp in specialites]
    position_x = x - (largeur_totale / 2) + (i * largeur_barre) + (largeur_barre / 2)
    ax.bar(position_x, temps_pred, largeur_barre, label=nom_algo)

ax.set_ylabel("Temps de prédiction par patient (secondes)")
ax.set_title("Comparaison des Temps de Prédiction (Vitesse de diagnostic)")
ax.set_xticks(x)
ax.set_xticklabels(specialites)
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
chemin_temps_pred = dossier_graphes / "comparaison_temps_prediction_dynamique.png"
plt.savefig(chemin_temps_pred)
plt.close()
print(f"-> Graphique généré : {chemin_temps_pred.name}")