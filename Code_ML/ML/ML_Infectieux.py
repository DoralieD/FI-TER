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
print("Lancement de l'IA Multiclasse - Diagnostic Infectieux (Palier 2)...")

BASE_DIR = Path(__file__).resolve().parent
chemin_donnees = BASE_DIR.parent / "Données_syn" / "Données_triée" / "Dossier_Palier2_Infectieux" / "dataset_infectieux_palier2.csv"

# Dossier de sortie des graphiques spécifique à l'infectiologie
dossier_graphes = BASE_DIR.parent / "Dossier_graphiques" / "Palier2_Infectieux"
dossier_graphes.mkdir(parents=True, exist_ok=True)

try:
    df = pd.read_csv(chemin_donnees)
    print(f"-> Base de données chargée : {len(df)} patients.")
except FileNotFoundError:
    print(f"Erreur : Le fichier {chemin_donnees} est introuvable. Veuillez exécuter le générateur Infectieux d'abord.")
    exit()

# ==========================================
# 2. PRÉPARATION DES DONNÉES
# ==========================================
# Suppression des colonnes non pertinentes pour la prédiction
colonnes_a_retirer = ["ID_Patient", "Diagnostic_Final_Infectieux"]
colonnes_a_retirer += [col for col in df.columns if "Verite_" in col or "Accord_" in col or "Risque_" in col]

y_brut = df["Diagnostic_Final_Infectieux"]
X = df.drop(columns=colonnes_a_retirer, errors="ignore")

# Gestion des valeurs manquantes : remplacement par la médiane
# Hypothèse : un examen non réalisé (NaN) correspond à un résultat "normal"
# → la médiane est une meilleure approximation que la moyenne (robuste aux valeurs extrêmes)
if X.isnull().any().any():
    X = X.fillna(X.median(numeric_only=True))

# Encodage de la cible (6 pathologies infectieuses → 0 à 5)
encodeur = LabelEncoder()
y = encodeur.fit_transform(y_brut)
noms_classes = encodeur.classes_
n_classes = len(noms_classes)
print(f"-> Diagnostics infectieux détectés : {list(noms_classes)}")

# Binarisation pour les courbes ROC (One-vs-Rest)
y_binarise = label_binarize(y, classes=np.arange(n_classes))

# Séparation train / test (80% / 20%) avec stratification
# La stratification est importante ici car Meningite_Bacterienne est sous-représentée (5%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
y_train_bin, y_test_bin = train_test_split(
    y_binarise, test_size=0.2, random_state=42, stratify=y
)

# Normalisation (apprise sur train, appliquée sur test — évite le data leakage)
print("Mise à l'échelle des données...")
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ==========================================
# 3. CHARGEMENT DYNAMIQUE DES MODÈLES (DEPUIS JSON)
# ==========================================
chemin_config = BASE_DIR.parent / "models_ml.json"
modeles_a_tester = {}

def instancier_modele(config_dict):
    """Instancie dynamiquement un modèle depuis sa configuration JSON."""
    module = importlib.import_module(config_dict["module"])
    ModelClass = getattr(module, config_dict["class"])
    params = config_dict.get("params", {}).copy()
    # XGBoost nécessite mlogloss en mode multiclasse
    if config_dict["class"] == "XGBClassifier":
        params["eval_metric"] = "mlogloss"
    return ModelClass(**params)

try:
    with open(chemin_config, "r") as f:
        config = json.load(f)
    for nom_algo, config_algo in config.items():
        modeles_a_tester[nom_algo] = instancier_modele(config_algo)
    print(f"-> {len(modeles_a_tester)} modèles chargés depuis {chemin_config.name}.")
except FileNotFoundError:
    print(f"Erreur : Fichier de configuration '{chemin_config}' introuvable.")
    exit()

# ==========================================
# 3. ENTRAÎNEMENT ET ÉVALUATION MULTICLASSE
# ==========================================
print("\n" + "="*50)
print("3. COMPÉTITION DES IA SUR LE DIAGNOSTIC INFECTIEUX")
print("="*50)

resultats_accuracy = {}
resultats_temps_entrainement = {}
resultats_temps_prediction = {}
resultats_roc = {}

for nom_algo, modele in modeles_a_tester.items():
    print(f"\n--- Algorithme : {nom_algo} ---")

    # --- Entraînement ---
    start = time.time()
    modele.fit(X_train, y_train)
    temps_entrainement = time.time() - start
    resultats_temps_entrainement[nom_algo] = temps_entrainement

    # --- Prédiction ---
    start_pred = time.time()
    predictions = modele.predict(X_test)
    temps_pred_total = time.time() - start_pred
    resultats_temps_prediction[nom_algo] = temps_pred_total / len(X_test)

    # --- Métriques ---
    acc = accuracy_score(y_test, predictions)
    resultats_accuracy[nom_algo] = acc
    print(f"Précision (Accuracy) : {acc * 100:.2f}%")

    # --- Courbes ROC One-vs-Rest ---
    if hasattr(modele, "predict_proba"):
        probas = modele.predict_proba(X_test)
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probas[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        resultats_roc[nom_algo] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

    # --- Matrice de Confusion ---
    # Couleur verte (teal) associée à l'infectiologie dans la convention graphique du projet
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions,
        display_labels=noms_classes,
        cmap="Greens",
        ax=ax,
        xticks_rotation=45
    )
    plt.title(f"Matrice de Confusion Infectieux - {nom_algo}")
    plt.tight_layout()
    plt.savefig(dossier_graphes / f"matrice_confusion_{nom_algo}.png")
    plt.close()

# ==========================================
# 4. GÉNÉRATION DES GRAPHIQUES DE SYNTHÈSE
# ==========================================
print("\n" + "="*50)
print("4. GÉNÉRATION DES GRAPHIQUES GLOBAUX")
print("="*50)

noms_modeles = list(resultats_accuracy.keys())

# --- Graphique 1 : Bilan Accuracy ---
plt.figure(figsize=(10, 6))
bars = plt.bar(noms_modeles, [resultats_accuracy[nom] for nom in noms_modeles], color="#2A9D8F")
plt.title("Précision de diagnostic par algorithme (Palier 2 Infectieux)")
plt.ylabel("Précision (Accuracy)")
plt.ylim(0, 1.10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02,
             f"{yval*100:.1f}%", ha="center", va="bottom", fontweight="bold")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(dossier_graphes / "bilan_accuracy_infectieux.png")
plt.close()

# --- Graphique 2 : Temps d'entraînement ---
plt.figure(figsize=(10, 6))
plt.bar(noms_modeles, [resultats_temps_entrainement[nom] for nom in noms_modeles], color="#E63946")
plt.title("Temps d'Entraînement par Algorithme (Palier 2 Infectieux)")
plt.ylabel("Secondes")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(dossier_graphes / "bilan_temps_entrainement_infectieux.png")
plt.close()

# --- Graphique 3 : Vitesse de prédiction ---
plt.figure(figsize=(10, 6))
plt.bar(noms_modeles, [resultats_temps_prediction[nom] for nom in noms_modeles], color="#457B9D")
plt.title("Vitesse de Prédiction par Patient (Palier 2 Infectieux)")
plt.ylabel("Secondes par patient")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(dossier_graphes / "bilan_temps_prediction_infectieux.png")
plt.close()

# --- Graphique 4 : Courbes ROC Multiclasses ---
# Note clinique : la détection du Sepsis_Grave est la plus critique.
# Un AUC élevé sur cette classe est prioritaire sur les autres.
couleurs_roc = ["#E63946", "#457B9D", "#2A9D8F", "#8338EC", "#F4A261", "#FB8500"]

for nom_algo, metrics in resultats_roc.items():
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(n_classes), couleurs_roc):
        plt.plot(
            metrics["fpr"][i], metrics["tpr"][i],
            color=color, lw=2,
            label=f"{noms_classes[i]} (AUC = {metrics['auc'][i]:.3f})"
        )
    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Aléatoire (AUC = 0.5)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taux de Faux Positifs (FPR)")
    plt.ylabel("Taux de Vrais Positifs (TPR)")
    plt.title(f"Courbes ROC Infectieux (One-vs-Rest) - {nom_algo}")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dossier_graphes / f"courbes_roc_{nom_algo}.png")
    plt.close()

print(f"-> Tous les graphiques ont été sauvegardés dans : {dossier_graphes.name}")
print("\nOpération Palier 2 Infectieux terminée avec succès.")