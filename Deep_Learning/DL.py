import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import time
import multiprocessing
import copy
import shap
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                             roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore') # Pour cacher les petits avertissements de SHAP

# ==========================================
# 1. PRÉPARATION ET CONFIGURATION
# ==========================================
if __name__ == '__main__':
    nombre_coeurs = multiprocessing.cpu_count()
    torch.set_num_threads(nombre_coeurs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*60)
    print(f"SCRIPT : DL AVANCE + XAI + DASHBOARD COMPLET")
    print(f"Materiel : {device} ({nombre_coeurs} coeurs)")
    print("="*60)

    BASE_DIR = Path(__file__).resolve().parent
    chemin_donnees = BASE_DIR / "Données_syn" / "super_fichier_hopital.csv"
    dossier_graphes = BASE_DIR / "Dossier_graphiques" / "Dashboard_Ultime"
    dossier_graphes.mkdir(parents=True, exist_ok=True)

    print("-> Chargement et Nettoyage du fichier...")
    try:
        df = pd.read_csv(chemin_donnees)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {chemin_donnees} est introuvable.")
        exit()
    
    df['Diagnostic_Global'] = df['Diagnostic_Final_Cardio'] \
        .combine_first(df['Diagnostic_Final_Infectieux']) \
        .combine_first(df['Diagnostic_Final_Neuro']) \
        .combine_first(df['Diagnostic_Final_Respi'])

    df = df.dropna(subset=['Diagnostic_Global'])
    df = df.fillna(0)

    colonnes_a_retirer = ["ID_Patient", "Diagnostic_Global"]
    colonnes_a_retirer += [col for col in df.columns if "Diagnostic_Final" in col or "Verite_" in col]

    X_brut = df.drop(columns=colonnes_a_retirer, errors="ignore")
    y_brut = df["Diagnostic_Global"]

    encodeur = LabelEncoder()
    y_encode = encodeur.fit_transform(y_brut)
    noms_classes = encodeur.classes_
    n_classes = len(noms_classes)
    n_features = X_brut.shape[1]

    # Binarisation pour Courbes ROC
    y_binarise = label_binarize(y_encode, classes=np.arange(n_classes))

    X_train, X_test, y_train, y_test = train_test_split(X_brut, y_encode, test_size=0.2, random_state=42)
    y_train_bin, y_test_bin = train_test_split(y_binarise, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    poids_classes = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    poids_tensor = torch.FloatTensor(poids_classes).to(device)

    class DatasetHopital(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx): return self.X[idx], self.y[idx]

    batch_size = 1024
    train_loader = DataLoader(DatasetHopital(X_train_scaled, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DatasetHopital(X_test_scaled, y_test), batch_size=batch_size, shuffle=False)

    # ==========================================
    # 2. LES ARCHITECTURES
    # ==========================================
    class HopitalNet(nn.Module):
        def __init__(self, input_size, num_classes):
            super(HopitalNet, self).__init__()
            self.fc1 = nn.Linear(input_size, 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.3)
            self.fc2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(0.2)
            self.fc3 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
            x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
            return self.fc3(x)

    class TabularTransformer(nn.Module):
        def __init__(self, n_features, num_classes, d_model=64, n_heads=4, n_layers=2):
            super(TabularTransformer, self).__init__()
            self.tokenizer_weights = nn.Parameter(torch.randn(n_features, d_model))
            self.tokenizer_biases = nn.Parameter(torch.randn(n_features, d_model))
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, dropout=0.2, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            batch_size = x.size(0)
            x_emb = x.unsqueeze(2) * self.tokenizer_weights.unsqueeze(0) + self.tokenizer_biases.unsqueeze(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x_emb = torch.cat((cls_tokens, x_emb), dim=1)
            out = self.transformer(x_emb)
            return self.head(out[:, 0, :])

    # ==========================================
    # 3. ENTRAÎNEMENT INTELLIGENT ET ÉVALUATION
    # ==========================================
    def entrainer_intelligent(modele, nom_modele, epoches=30, patience=4):
        print(f"\nEntrainement de {nom_modele} (Early Stopping max {epoches} epoques)...")
        start_time = time.time()
        modele.to(device)
        critere = nn.CrossEntropyLoss(weight=poids_tensor)
        optimiseur = optim.AdamW(modele.parameters(), lr=0.002, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiseur, mode='min', factor=0.5, patience=2)
        
        meilleure_perte_val = float('inf')
        meilleurs_poids = None
        compteur_patience = 0
        
        historique_train = []
        historique_val = []

        for epoch in range(epoches):
            modele.train()
            perte_train = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimiseur.zero_grad()
                perte = critere(modele(batch_X), batch_y)
                perte.backward()
                optimiseur.step()
                perte_train += perte.item()
            
            perte_train_moyenne = perte_train / len(train_loader)
            historique_train.append(perte_train_moyenne)
            
            modele.eval()
            perte_val = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    perte_val += critere(modele(batch_X), batch_y).item()
            
            perte_val_moyenne = perte_val / len(val_loader)
            historique_val.append(perte_val_moyenne)
            
            scheduler.step(perte_val_moyenne)

            if perte_val_moyenne < meilleure_perte_val:
                meilleure_perte_val = perte_val_moyenne
                meilleurs_poids = copy.deepcopy(modele.state_dict())
                compteur_patience = 0
            else:
                compteur_patience += 1
                if compteur_patience >= patience:
                    print(f"-> STOP: Stagnation atteinte a l'epoque {epoch+1}.")
                    break
        
        if meilleurs_poids is not None:
            modele.load_state_dict(meilleurs_poids)
            
        temps_total = time.time() - start_time
        print(f"-> Termine en {temps_total:.1f} sec.")
        return modele, temps_total, historique_train, historique_val

    def evaluer_modele(modele):
        modele.eval()
        toutes_probas, toutes_preds = [], []
        start_time = time.time()
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device)
                sorties = modele(batch_X)
                probas = torch.softmax(sorties, dim=1)
                _, preds = torch.max(sorties, 1)
                toutes_probas.extend(probas.cpu().numpy())
                toutes_preds.extend(preds.cpu().numpy())
        temps_pred = time.time() - start_time
        return np.array(toutes_preds), np.array(toutes_probas), temps_pred

    # Exécution des deux Champions
    modele_nn, temps_entrainement_nn, train_loss_nn, val_loss_nn = entrainer_intelligent(HopitalNet(n_features, n_classes), "Neural Network")
    preds_nn, probas_nn, temps_pred_nn = evaluer_modele(modele_nn)

    modele_tf, temps_entrainement_tf, train_loss_tf, val_loss_tf = entrainer_intelligent(TabularTransformer(n_features, n_classes), "Transformer")
    preds_tf, probas_tf, temps_pred_tf = evaluer_modele(modele_tf)

    # ==========================================
    # 4. GENERATION DU DA
    # ==========================================
    print("\nGenerating the Visual Dashboard...")

    # Graph 1 : Performances
    acc_nn = accuracy_score(y_test, preds_nn)
    prec_nn, rec_nn, f1_nn, _ = precision_recall_fscore_support(y_test, preds_nn, average='macro', zero_division=0)
    acc_tf = accuracy_score(y_test, preds_tf)
    prec_tf, rec_tf, f1_tf, _ = precision_recall_fscore_support(y_test, preds_tf, average='macro', zero_division=0)

    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, [acc_nn, prec_nn, rec_nn, f1_nn], width, label='Neural Network', color='#457B9D')
    bars2 = ax.bar(x + width/2, [acc_tf, prec_tf, rec_tf, f1_tf], width, label='Transformer', color='#E63946')
    
    ax.bar_label(bars1, padding=3, fmt='%.3f')
    ax.bar_label(bars2, padding=3, fmt='%.3f')
    
    ax.set_ylabel('Scores')
    ax.set_title('Global Performance Overview', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(dossier_graphes / "01_Performances_Globales.png", bbox_inches='tight')
    plt.close()

    # Graph 2 : Computation Time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bar_time1 = ax1.bar(["Neural Net", "Transformer"], [temps_entrainement_nn, temps_entrainement_tf], color=['#457B9D', '#E63946'])
    ax1.bar_label(bar_time1, padding=3, fmt='%.1f')
    ax1.set_title("Total Training Time", pad=15)
    ax1.set_ylabel("Seconds")
    
    bar_time2 = ax2.bar(["Neural Net", "Transformer"], [temps_pred_nn, temps_pred_tf], color=['#457B9D', '#E63946'])
    ax2.bar_label(bar_time2, padding=3, fmt='%.3f')
    ax2.set_title("Inference Time", pad=15)
    plt.tight_layout()
    plt.savefig(dossier_graphes / "02_Temps_de_Calcul.png", bbox_inches='tight')
    plt.close()

    # Graph 3 : Confusion Matrix NN
    fig, ax = plt.subplots(figsize=(18, 16))
    disp_nn = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, preds_nn), display_labels=noms_classes)
    disp_nn.plot(cmap="Blues", ax=ax, xticks_rotation=90, colorbar=False)
    plt.title("Confusion Matrix - Neural Network", fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(dossier_graphes / "03_Matrice_Confusion_NN.png", bbox_inches='tight')
    plt.close()

    # Graph 4 : Confusion Matrix TF
    fig, ax = plt.subplots(figsize=(18, 16))
    disp_tf = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, preds_tf), display_labels=noms_classes)
    disp_tf.plot(cmap="Reds", ax=ax, xticks_rotation=90, colorbar=False)
    plt.title("Confusion Matrix - Transformer", fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(dossier_graphes / "04_Matrice_Confusion_Transformer.png", bbox_inches='tight')
    plt.close()

    # Graph 5 : ROC Curves
    fpr_nn, tpr_nn, _ = roc_curve(y_test_bin.ravel(), probas_nn.ravel())
    fpr_tf, tpr_tf, _ = roc_curve(y_test_bin.ravel(), probas_tf.ravel())
    plt.figure(figsize=(9, 7))
    plt.plot(fpr_nn, tpr_nn, color='#457B9D', lw=2, label=f'Neural Net (AUC = {auc(fpr_nn, tpr_nn):.3f})')
    plt.plot(fpr_tf, tpr_tf, color='#E63946', lw=2, label=f'Transformer (AUC = {auc(fpr_tf, tpr_tf):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Comparative ROC Curve', pad=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(dossier_graphes / "05_Courbes_ROC.png", bbox_inches='tight')
    plt.close()

    # Graph 6 : SHAP
    print("\nGenerating SHAP explanations (This may take a moment)...")
    modele_nn.eval()
    echantillon_fond = torch.FloatTensor(X_train_scaled[:100]).to(device)
    echantillon_test = torch.FloatTensor(X_test_scaled[:100]).to(device)
    explainer = shap.DeepExplainer(modele_nn, echantillon_fond)
    shap_values = explainer.shap_values(echantillon_test)
    
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=X_brut.columns, show=False)
    plt.title("Feature Importance (SHAP Summary)", pad=20, fontsize=16)
    plt.tight_layout()
    plt.savefig(dossier_graphes / "06_Explicabilite_SHAP.png", bbox_inches='tight')
    plt.close()

    # Graph 7 : Learning Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(train_loss_nn, label='Training Loss', color='#457B9D', linestyle='-')
    ax1.plot(val_loss_nn, label='Validation Loss', color='#E63946', linestyle='--')
    ax1.set_title("Learning Curve - Neural Network", pad=15)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss (CrossEntropy)")
    ax1.legend()

    ax2.plot(train_loss_tf, label='Training Loss', color='#457B9D', linestyle='-')
    ax2.plot(val_loss_tf, label='Validation Loss', color='#E63946', linestyle='--')
    ax2.set_title("Learning Curve - Transformer", pad=15)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss (CrossEntropy)")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(dossier_graphes / "07_Courbes_Apprentissage.png", bbox_inches='tight')
    plt.close()

# ==========================================
    # 5. LES NOUVEAUX GRAPHIQUES (ARCHITECTURE, PR CURVE, CORRELATION)
    # ==========================================
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import seaborn as sns
    import matplotlib.patches as patches

   # Graphique 8 : Schéma de l'Architecture (Dessin conceptuel en Matplotlib)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Boîtes pour le MLP (Gauche)
    ax.add_patch(patches.FancyBboxPatch((0.1, 0.7), 0.3, 0.15, facecolor='#457B9D', edgecolor='black', boxstyle='round,pad=0.02'))
    ax.text(0.25, 0.775, 'Multimodal Inputs\n(Vitals + Labs)', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.add_patch(patches.FancyBboxPatch((0.1, 0.45), 0.3, 0.15, facecolor='#A8DADC', edgecolor='black', boxstyle='round,pad=0.02'))
    ax.text(0.25, 0.525, 'Concatenation\n(Early Fusion)', color='black', ha='center', va='center', fontsize=12)

    ax.add_patch(patches.FancyBboxPatch((0.1, 0.2), 0.3, 0.15, facecolor='#1D3557', edgecolor='black', boxstyle='round,pad=0.02'))
    ax.text(0.25, 0.275, 'MLP Layers\n(256 -> 128 -> 24)', color='white', ha='center', va='center', fontsize=12, fontweight='bold')

    # Flèches MLP
    ax.annotate('', xy=(0.25, 0.6), xytext=(0.25, 0.7), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate('', xy=(0.25, 0.35), xytext=(0.25, 0.45), arrowprops=dict(arrowstyle="->", lw=2))

    # Boîtes pour le Transformer (Droite)
    ax.add_patch(patches.FancyBboxPatch((0.6, 0.7), 0.3, 0.15, facecolor='#E63946', edgecolor='black', boxstyle='round,pad=0.02'))
    ax.text(0.75, 0.775, 'Multimodal Inputs\n(Vitals + Labs)', color='white', ha='center', va='center', fontsize=12, fontweight='bold')

    ax.add_patch(patches.FancyBboxPatch((0.6, 0.45), 0.3, 0.15, facecolor='#F1FAEE', edgecolor='black', boxstyle='round,pad=0.02'))
    ax.text(0.75, 0.525, 'Feature Tokenizer\n+ [CLS] Token', color='black', ha='center', va='center', fontsize=12)

    ax.add_patch(patches.FancyBboxPatch((0.6, 0.2), 0.3, 0.15, facecolor='#9B2226', edgecolor='black', boxstyle='round,pad=0.02'))
    ax.text(0.75, 0.275, 'Multi-Head Attention\n(Contextual Fusion)', color='white', ha='center', va='center', fontsize=12, fontweight='bold')

    # Flèches Transformer
    ax.annotate('', xy=(0.75, 0.6), xytext=(0.75, 0.7), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate('', xy=(0.75, 0.35), xytext=(0.75, 0.45), arrowprops=dict(arrowstyle="->", lw=2))

    # Titres globaux
    ax.text(0.25, 0.9, 'Baseline MLP (Early Fusion)', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(0.75, 0.9, 'Tabular Transformer (Contextual Fusion)', ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(dossier_graphes / "08_Schema_Architecture.png", bbox_inches='tight')
    plt.close()

    # Graphique 9 : Courbes Precision-Recall (Idéal pour la médecine)
    precision_nn, recall_nn, _ = precision_recall_curve(y_test_bin.ravel(), probas_nn.ravel())
    ap_nn = average_precision_score(y_test_bin, probas_nn, average="micro")
    
    precision_tf, recall_tf, _ = precision_recall_curve(y_test_bin.ravel(), probas_tf.ravel())
    ap_tf = average_precision_score(y_test_bin, probas_tf, average="micro")

    plt.figure(figsize=(9, 7))
    plt.plot(recall_nn, precision_nn, color='#457B9D', lw=2, label=f'Neural Net (AP = {ap_nn:.3f})')
    plt.plot(recall_tf, precision_tf, color='#E63946', lw=2, label=f'Transformer (AP = {ap_tf:.3f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title('Precision-Recall Curve (Micro-Averaged)', pad=15)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(dossier_graphes / "09_Courbe_Precision_Recall.png", bbox_inches='tight')
    plt.close()

    # Graphique 10 : Matrice de Corrélation (Validation des données synthétiques)
    plt.figure(figsize=(12, 10))
    # On prend un échantillon des colonnes les plus importantes pour que ça reste lisible
    colonnes_corr = [col for col in X_brut.columns if len(X_brut[col].unique()) > 2][:15] 
    matrice_corr = X_brut[colonnes_corr].corr()
    
    sns.heatmap(matrice_corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Physiological Correlation Matrix (Synthetic Data Validation)", pad=20, fontsize=16)
    plt.tight_layout()
    plt.savefig(dossier_graphes / "10_Matrice_Correlation.png", bbox_inches='tight')
    plt.close()
    


