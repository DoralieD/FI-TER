import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier  # <-- remplace RandomForestClassifier

# ==========================================
# 1. CHARGEMENT DU FICHIER DE DONNÉES
# ==========================================
print("Chargement de la base de données (10 000 patients)...")
try:
    df = pd.read_csv("dataset_urgences_10000.csv")
except FileNotFoundError:
    print("Erreur : Le fichier 'dataset_urgences_10000.csv' est introuvable.")
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
        # XGBClassifier remplace RandomForestClassifier
        # use_label_encoder=False et eval_metric='logloss' évitent des warnings
        self.modele = XGBClassifier(
            n_estimators=200,        # plus d'arbres qu'un RF classique
            max_depth=6,             # profondeur plus faible, XGB compense par le boosting
            learning_rate=0.1,       # taux d'apprentissage (eta)
            subsample=0.8,           # échantillonnage des lignes par arbre
            colsample_bytree=0.8,    # échantillonnage des colonnes par arbre
            eval_metric="logloss",
            random_state=42
        )

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
            "Cardio":     NoeudSpecialiste("Cardio"),
            "Respi":      NoeudSpecialiste("Respi"),
            "Infectieux": NoeudSpecialiste("Infectieux"),
            "Neuro":      NoeudSpecialiste("Neuro")
        }

    def entrainer_systeme(self, X_train, y_train):
        print("Entraînement des 4 XGBoost en cours...")
        for nom, noeud in self.noeuds.items():
            noeud.entrainer(X_train, y_train[f"Verite_{nom}"])
        print("Entraînement terminé.\n")

    def tester_systeme(self, X_test, y_test):
        print("=== RÉSULTATS DE L'EXAMEN (sur 2000 patients inconnus) ===")
        for nom, noeud in self.noeuds.items():
            noeud.evaluer(X_test, y_test[f"Verite_{nom}"])
        print("==========================================================\n")

    def trier_nouveau_patient(self, donnees_patient):
        print("--- ARRIVÉE D'UN NOUVEAU PATIENT AUX URGENCES ---")

        if donnees_patient["Obs_TraumaPenetrant"].iloc[0] == 1:
            print("ALERTE ROUGE : Objet étranger / Trauma pénétrant détecté !")
            print("=> DÉCISION DU TRIAGE : Bypass IA. Orientation immédiate en salle de déchocage / Bloc Chirurgical.")
            return ["Chirurgie_Urgence"]

        print("Cas médical complexe. Lancement de l'analyse IA du graphe...")
        chemins_ouverts = []
        seuil_alerte = 0.40

        features_ia = donnees_patient.drop(columns=["ID_Patient"], errors="ignore")

        for nom, noeud in self.noeuds.items():
            risque = noeud.evaluer_risque_pourcentage(features_ia)
            print(f"Risque {nom.ljust(10)} : {risque*100:05.1f}%")
            if risque >= seuil_alerte:
                chemins_ouverts.append(nom)

        print("\n=> DÉCISION DU TRIAGE (Fin du Palier 1) :")
        if chemins_ouverts:
            print(f"Ouverture des branches vers le Palier 2 : {chemins_ouverts}")
        else:
            print("Aucun risque critique identifié. Orientation vers médecine générale.")

        return chemins_ouverts

# ==========================================
# 3. EXÉCUTION DU PROGRAMME
# ==========================================
hopital_ia = TriagePalier1()
hopital_ia.entrainer_systeme(X_train, y_train)
hopital_ia.tester_systeme(X_test, y_test)

# ==========================================
# 4. RÉPARTITION DES 2000 PATIENTS DANS LES FICHIERS CSV
# ==========================================
print("\n" + "="*50)
print("4. GÉNÉRATION DES 4 FICHIERS CSV POUR LES 2000 PATIENTS")
print("="*50)

seuil_alerte = 0.40
ids_tous = df["ID_Patient"]          # tous les 10 000 IDs
X_tous = df.drop(columns=["ID_Patient"] + colonnes_cibles)  # toutes les features

for nom, noeud in hopital_ia.noeuds.items():
    risques = noeud.modele.predict_proba(X_tous)[:, 1]  # sur les 10 000

    masque_risque = risques >= seuil_alerte
    patients_a_risque = X_tous[masque_risque].copy()

    patients_a_risque.insert(0, "ID_Patient", ids_tous[masque_risque])
    patients_a_risque[f"Risque_{nom}_%"] = (risques[masque_risque] * 100).round(2)

    nom_dossier = f"Dossier_Palier2_{nom}"
    os.makedirs(nom_dossier, exist_ok=True)

    chemin_fichier = os.path.join(nom_dossier, f"patients_{nom.lower()}.csv")
    patients_a_risque.to_csv(chemin_fichier, index=False)

    print(f"-> {len(patients_a_risque)} patients classés dans : {chemin_fichier}")

print("\nOpération terminée : Les patients ont été répartis dans les 4 fichiers CSV.")