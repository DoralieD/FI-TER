import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==========================================
# 1. CONFIGURATION DES CHEMINS ET CHARGEMENT
# ==========================================
# Définition du dossier de base (Code_ML)
BASE_DIR = Path(__file__).resolve().parent

# Chemins de lecture et d'écriture relatifs au dossier Code_ML
chemin_donnees = BASE_DIR / "Données_syn" / "dataset_urgences.csv"
dossier_tri = BASE_DIR / "Données_syn" / "Données_triée"
dossier_graphes = BASE_DIR / "Dossier_graphiques"

# Création automatique des dossiers de sortie s'ils n'existent pas
dossier_tri.mkdir(parents=True, exist_ok=True)
dossier_graphes.mkdir(parents=True, exist_ok=True)

try:
    df = pd.read_csv(chemin_donnees)
except FileNotFoundError:
    print(f"Erreur : Le fichier {chemin_donnees} est introuvable.")
    exit()

colonnes_cibles = ["Verite_Cardio", "Verite_Respi", "Verite_Infectieux", "Verite_Neuro"]

y = df[colonnes_cibles]
X = df.drop(columns=["ID_Patient"] + colonnes_cibles)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. L'ARCHITECTURE DU GRAPHE (L'IA)
# ==========================================
class NoeudSpecialiste:
    def __init__(self, nom):
        self.nom = nom
        self.modele = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)

    def entrainer(self, X_train, y_train):
        self.modele.fit(X_train, y_train)

    def evaluer(self, X_test, y_test):
        predictions = self.modele.predict(X_test)
        precision = accuracy_score(y_test, predictions)
        print(f"Précision du nœud [{self.nom.ljust(10)}] : {precision * 100:.2f}%")

    def evaluer_risque_pourcentage(self, patient):
        return self.modele.predict_proba(patient)[0][1]

class TriagePalier1:
    def __init__(self):
        self.noeuds = {
            "Cardio": NoeudSpecialiste("Cardio"),
            "Respi": NoeudSpecialiste("Respi"),
            "Infectieux": NoeudSpecialiste("Infectieux"),
            "Neuro": NoeudSpecialiste("Neuro")
        }
        self.temps_entrainement = {}
        self.temps_prediction = {}
        self.fpr = {}
        self.tpr = {}
        self.roc_auc = {}

    def entrainer_systeme(self, X_train, y_train):
        print("Entraînement des 4 intelligences artificielles en cours...")
        for nom, noeud in self.noeuds.items():
            start_time = time.time()
            noeud.entrainer(X_train, y_train[f"Verite_{nom}"])
            end_time = time.time()
            self.temps_entrainement[nom] = end_time - start_time
        print("Entraînement terminé.\n")

    def tester_systeme(self, X_test, y_test):
        print("=== RÉSULTATS DE L'EXAMEN ===")
        for nom, noeud in self.noeuds.items():
            noeud.evaluer(X_test, y_test[f"Verite_{nom}"])
            
            start_time = time.time()
            probabilites = noeud.modele.predict_proba(X_test)[:, 1]
            end_time = time.time()
            self.temps_prediction[nom] = end_time - start_time
            
            vraies_valeurs = y_test[f"Verite_{nom}"]
            fpr, tpr, _ = roc_curve(vraies_valeurs, probabilites)
            self.fpr[nom] = fpr
            self.tpr[nom] = tpr
            self.roc_auc[nom] = auc(fpr, tpr)
            
        print("==========================================================\n")

    def trier_nouveau_patient(self, donnees_patient):
        pass

# ==========================================
# 3. EXÉCUTION DU PROGRAMME
# ==========================================

hopital_ia = TriagePalier1()
hopital_ia.entrainer_systeme(X_train, y_train)
hopital_ia.tester_systeme(X_test, y_test)

# ==========================================
# 4. RÉPARTITION DES PATIENTS DANS LES FICHIERS CSV
# ==========================================

print("\n" + "="*50)
print("4. GÉNÉRATION DES DOSSIERS ET FICHIERS CSV MULTIPLES")
print("="*50)

seuil_alerte = 0.40
ids_test = df.loc[X_test.index, "ID_Patient"]

for nom, noeud in hopital_ia.noeuds.items():
    risques = noeud.modele.predict_proba(X_test)[:, 1]
    masque_risque = risques >= seuil_alerte
    
    patients_a_risque = X_test[masque_risque].copy()
    patients_a_risque.insert(0, "ID_Patient", ids_test[masque_risque])
    patients_a_risque[f"Risque_{nom}_%"] = (risques[masque_risque] * 100).round(2)
    
    # Création du sous-dossier avec pathlib
    chemin_palier = dossier_tri / f"Dossier_Palier2_{nom}"
    chemin_palier.mkdir(exist_ok=True)
    
    # Exportation
    chemin_final_fichier = chemin_palier / f"patients_{nom.lower()}.csv"
    patients_a_risque.to_csv(chemin_final_fichier, index=False)
    
    print(f"{len(patients_a_risque)} patients classés dans : {chemin_final_fichier.relative_to(BASE_DIR.parent)}")

# ==========================================
# 5. VISUALISATION AVEC MATPLOTLIB
# ==========================================
print("\n" + "="*50)
print("5. GÉNÉRATION DES GRAPHIQUES CLASSIQUES")
print("="*50)

specialites = []
nombre_patients = []

for nom, noeud in hopital_ia.noeuds.items():
    risques = noeud.modele.predict_proba(X_test)[:, 1]
    specialites.append(nom)
    nombre_patients.append((risques >= 0.40).sum())

plt.figure(figsize=(10, 6))
plt.bar(specialites, nombre_patients, color=['#E63946', '#457B9D', '#2A9D8F', '#8338EC'])
plt.title(f"Répartition des patients à risque par spécialité (Total Test: {len(X_test)})")
plt.ylabel("Nombre de patients")
plt.xlabel("Pôles d'urgence")
plt.grid(axis='y', linestyle='--', alpha=0.7)

chemin_graphe = dossier_graphes / "repartition_patients.png"
plt.savefig(chemin_graphe)
plt.close()
print(f"Graphique généré : {chemin_graphe.name}")


# ==========================================
# 6. ÉVALUATION DES PERFORMANCES (ROC, AUC, TEMPS)
# ==========================================
print("\n" + "="*50)
print("6. GÉNÉRATION DES GRAPHIQUES DE PERFORMANCE")
print("="*50)

# --- Graphique 3 : Courbes ROC et AUC ---
plt.figure(figsize=(10, 8))
couleurs = {'Cardio': '#E63946', 'Respi': '#457B9D', 'Infectieux': '#2A9D8F', 'Neuro': '#8338EC'}

for nom in hopital_ia.noeuds.keys():
    plt.plot(hopital_ia.fpr[nom], hopital_ia.tpr[nom], color=couleurs.get(nom, 'black'), lw=2,
             label=f"{nom} (AUC = {hopital_ia.roc_auc[nom]:.3f})")

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Taux de Faux Positifs (FPR)")
plt.ylabel("Taux de Vrais Positifs (TPR)")
plt.title("Courbes ROC par spécialité de Triage")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

chemin_roc = dossier_graphes / "courbes_roc.png"
plt.savefig(chemin_roc)
plt.close()
print(f"Graphique généré : {chemin_roc.name}")

# --- Graphique 4 : Temps d'entraînement et de prédiction ---
noms = list(hopital_ia.noeuds.keys())
t_entrainement = [hopital_ia.temps_entrainement[nom] for nom in noms]
t_prediction = [hopital_ia.temps_prediction[nom] for nom in noms]

x = np.arange(len(noms))
largeur = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(x - largeur/2, t_entrainement, largeur, label="Entraînement sur 80%", color='#F4A261')
bar2 = ax.bar(x + largeur/2, t_prediction, largeur, label="Prédiction sur 20%", color='#264653')

ax.set_ylabel("Temps d'exécution (en secondes)")
ax.set_title("Vitesse de l'IA : Temps d'Entraînement vs Prédiction")
ax.set_xticks(x)
ax.set_xticklabels(noms)
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

chemin_temps = dossier_graphes / "temps_execution.png"
plt.savefig(chemin_temps)
plt.close()
print(f"Graphique généré : {chemin_temps.name}")