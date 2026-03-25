import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import json
import importlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, roc_curve, auc

# ==========================================
# 1. CONFIGURATION ET CHARGEMENT
# ==========================================
print("Lancement de l'IA Multiclasse - Diagnostic Respiratoire (Palier 2)...")

BASE_DIR = Path(__file__).resolve().parent
chemin_donnees = BASE_DIR.parent / "Données_syn" / "Données_triée" / "Dossier_Palier2_Respi" / "dataset_respi_palier2_renforce.csv"
dossier_graphes = BASE_DIR.parent / "Dossier_graphiques" / "Palier2_Respi"
dossier_graphes.mkdir(parents=True, exist_ok=True)
try:
    df = pd.read_csv(chemin_donnees)
    print(f"-> Base de données chargée : {len(df)} patients.")
except FileNotFoundError:
    print(f"Erreur : Le fichier {chemin_donnees} est introuvable.")
    exit()

colonnes_a_retirer = ["ID_Patient", "Diagnostic_Final_Respi"]
colonnes_a_retirer += [col for col in df.columns if "Verite_" in col or "Accord_" in col or "Risque_" in col]

y_brut = df["Diagnostic_Final_Respi"]
X = df.drop(columns=colonnes_a_retirer, errors="ignore")

# Encodage pour le multiclasse
encodeur = LabelEncoder()
y = encodeur.fit_transform(y_brut)
noms_classes = encodeur.classes_
n_classes = len(noms_classes)

# Binarisation de Y pour le calcul de la courbe ROC (One-vs-Rest)
y_binarise = label_binarize(y, classes=np.arange(n_classes))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_bin, y_test_bin = train_test_split(y_binarise, test_size=0.2, random_state=42)

print("Mise à l'échelle des données...")
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ==========================================
# 2. CHARGEMENT DYNAMIQUE DES MODÈLES (DEPUIS JSON)
# ==========================================
chemin_config = BASE_DIR.parent / "models_ml.json"
modeles_a_tester = {}

def instancier_modele(config_dict):
    module = importlib.import_module(config_dict["module"])
    ModelClass = getattr(module, config_dict["class"])
    params = config_dict.get("params", {}).copy()
    if config_dict["class"] == "XGBClassifier":
        params["eval_metric"] = "mlogloss" 
    return ModelClass(**params)

try:
    with open(chemin_config, "r") as f:
        config = json.load(f)
    for nom_algo, config_algo in config.items():
        modeles_a_tester[nom_algo] = instancier_modele(config_algo)
    print(f"-> {len(modeles_a_tester)} modèles chargés.")
except FileNotFoundError:
    print(f"Erreur : Fichier de configuration '{chemin_config}' introuvable.")
    exit()

# ==========================================
# 3. ENTRAÎNEMENT ET ÉVALUATION MULTICLASSE
# ==========================================
print("\n" + "="*50)
print("3. COMPÉTITION DES IA SUR LE DIAGNOSTIC FINAL")
print("="*50)

resultats_accuracy = {}
resultats_temps_entrainement = {}
resultats_temps_prediction = {}
resultats_roc = {} # Stockera les fpr, tpr et auc pour chaque modèle et chaque classe

for nom_algo, modele in modeles_a_tester.items():
    print(f"\n--- Algorithme : {nom_algo} ---")
    
    # Entraînement
    start = time.time()
    modele.fit(X_train, y_train)
    temps_entrainement = time.time() - start
    resultats_temps_entrainement[nom_algo] = temps_entrainement
    
    # Prédiction
    start_pred = time.time()
    predictions = modele.predict(X_test)
    temps_pred_total = time.time() - start_pred
    resultats_temps_prediction[nom_algo] = temps_pred_total / len(X_test) # Temps par échantillon
    
    acc = accuracy_score(y_test, predictions)
    resultats_accuracy[nom_algo] = acc
    
    # Calcul des probabilités pour les courbes ROC (si supporté par le modèle)
    if hasattr(modele, "predict_proba"):
        probas = modele.predict_proba(X_test)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # Calcul ROC One-vs-Rest pour chaque maladie
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probas[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        resultats_roc[nom_algo] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

    # --- Matrice de Confusion ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=noms_classes, cmap="Blues", ax=ax, xticks_rotation=45
    )
    plt.title(f"Matrice de Confusion - {nom_algo}")
    plt.tight_layout()
    plt.savefig(dossier_graphes / f"matrice_confusion_{nom_algo}.png")
    plt.close()

# ==========================================
# 4. GÉNÉRATION DES GRAPHIQUES COMPARATIFS
# ==========================================
print("\n" + "="*50)
print("4. GÉNÉRATION DES GRAPHIQUES GLOBAUX")
print("="*50)

noms_modeles = list(resultats_accuracy.keys())
x = np.arange(len(noms_modeles))

# --- Bilan Accuracy ---
plt.figure(figsize=(10, 6))
bars = plt.bar(noms_modeles, [resultats_accuracy[nom] for nom in noms_modeles], color='#2A9D8F')
plt.title("Précision de diagnostic par algorithme (Palier 2 Respiratoire)")
plt.ylabel("Précision (Accuracy)")
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval*100:.1f}%", ha='center', va='bottom', fontweight='bold')
plt.savefig(dossier_graphes / "bilan_accuracy_palier2.png")
plt.close()

# --- Temps d'Entraînement ---
plt.figure(figsize=(10, 6))
plt.bar(noms_modeles, [resultats_temps_entrainement[nom] for nom in noms_modeles], color='#E63946')
plt.title("Comparaison des Temps d'Entraînement (Palier 2)")
plt.ylabel("Temps d'entraînement (secondes)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(dossier_graphes / "bilan_temps_entrainement_palier2.png")
plt.close()

# --- Temps de Prédiction ---
plt.figure(figsize=(10, 6))
plt.bar(noms_modeles, [resultats_temps_prediction[nom] for nom in noms_modeles], color='#457B9D')
plt.title("Comparaison des Temps de Prédiction (Par Patient)")
plt.ylabel("Temps (secondes)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(dossier_graphes / "bilan_temps_prediction_palier2.png")
plt.close()

# --- Courbes ROC Multiclasses (Une image par modèle) ---
couleurs = ['#E63946', '#457B9D', '#2A9D8F', '#8338EC', '#F4A261']

for nom_algo, metrics in resultats_roc.items():
    plt.figure(figsize=(10, 8))
    
    for i, color in zip(range(n_classes), couleurs):
        plt.plot(metrics['fpr'][i], metrics['tpr'][i], color=color, lw=2,
                 label=f"{noms_classes[i]} (AUC = {metrics['auc'][i]:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title(f"Courbes ROC (One-vs-Rest) - {nom_algo}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(dossier_graphes / f"courbes_roc_{nom_algo}.png")
    plt.close()

print(f"-> Graphiques générés dans : {dossier_graphes}")
print("\nOpération Palier 2 terminée avec succès.")