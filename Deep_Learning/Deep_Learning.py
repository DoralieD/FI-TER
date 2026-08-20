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
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                             roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, average_precision_score)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PREPARATION AND CONFIGURATION
# ==========================================
if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*70)
    print(f"🚀 STARTING PIPELINE: ADVANCED DL + XGBOOST + XAI")
    print(f"💻 Hardware Config : {device} ({num_cores} cores detected)")
    print("="*70)

    BASE_DIR = Path(__file__).resolve().parent
    data_path = BASE_DIR / "Synthetic_Data" / "super_hospital_file.csv"
    graph_folder = BASE_DIR / "Graph_Folder" / "Dashboard_Ultimate"
    graph_folder.mkdir(parents=True, exist_ok=True)

    print("\nLoading and Preprocessing Data")
    try:
        df = pd.read_csv(data_path)
        print(f"  -> File successfully loaded. Initial shape: {df.shape}")
    except FileNotFoundError:
        print(f"  [!] Error: The file {data_path} was not found.")
        exit()
    
    # Merge targets
    df['Global_Diagnosis'] = df['Final_Diagnosis_Cardio'] \
        .combine_first(df['Final_Diagnosis_Infectious']) \
        .combine_first(df['Final_Diagnosis_Neuro']) \
        .combine_first(df['Final_Diagnosis_Respi'])

    df = df.dropna(subset=['Global_Diagnosis'])

    cols_to_remove = ["Patient_ID", "Global_Diagnosis"]
    cols_to_remove += [col for col in df.columns if "Final_Diagnosis" in col or "Truth_" in col]

    X_raw = df.drop(columns=cols_to_remove, errors="ignore")
    y_raw = df["Global_Diagnosis"]

    # CORRECTION: Median Imputation to avoid NaN bias (Reviewer 1)
    print("  -> Imputing missing laboratory values with Median...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    class_names = encoder.classes_
    n_classes = len(class_names)
    n_features = X_imputed.shape[1]

    y_binarized = label_binarize(y_encoded, classes=np.arange(n_classes))

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.2, random_state=42)
    y_train_bin, y_test_bin = train_test_split(y_binarized, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_tensor = torch.FloatTensor(class_weights).to(device)

    class HospitalDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx): return self.X[idx], self.y[idx]

    batch_size = 1024
    train_loader = DataLoader(HospitalDataset(X_train_scaled, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(HospitalDataset(X_test_scaled, y_test), batch_size=batch_size, shuffle=False)
    print(f"  -> Preprocessing complete. Target classes: {n_classes} | Features: {n_features}")

    # ==========================================
    # 2. ARCHITECTURES
    # ==========================================
    class HospitalNet(nn.Module):
        def __init__(self, input_size, num_classes):
            super(HospitalNet, self).__init__()
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
    # 3. TRAINING & EVALUATION FUNCTIONS
    # ==========================================
    def smart_train(model, model_name, epochs=30, patience=4):
        print(f"\nTraining {model_name} (Max Epochs: {epochs})...")
        model.to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        
        history_train, history_val = [], []
        start_time = time.time()

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history_train.append(avg_train_loss)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    val_loss += criterion(model(batch_X), batch_y).item()
            
            avg_val_loss = val_loss / len(val_loader)
            history_val.append(avg_val_loss)
            scheduler.step(avg_val_loss)

            # Progress Logging
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  -> Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  [!] Early stopping triggered at epoch {epoch+1} due to stagnation.")
                    break
        
        if best_weights is not None:
            model.load_state_dict(best_weights)
            
        total_time = time.time() - start_time
        print(f"  -> {model_name} Training Completed in {total_time:.2f} seconds.")
        return model, total_time, history_train, history_val

    def evaluate_model(model):
        model.eval()
        all_probs, all_preds = [], []
        start_time = time.time()
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        return np.array(all_preds), np.array(all_probs), time.time() - start_time

    # ==========================================
    # 4. EXECUTING MODELS
    # ==========================================
    print("\n[STEP 2/5] Training XGBoost Baseline")
    start_time_xgb = time.time()
    sample_weights = class_weights[y_train]
    xgb_model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    train_time_xgb = time.time() - start_time_xgb
    print(f"  -> XGBoost Training Completed in {train_time_xgb:.2f} seconds.")
    
    print("\n[STEP 3/5] Training Deep Learning Models")
    start_time_inf_xgb = time.time()
    preds_xgb = xgb_model.predict(X_test_scaled)
    probs_xgb = xgb_model.predict_proba(X_test_scaled)
    pred_time_xgb = time.time() - start_time_inf_xgb

    model_nn, train_time_nn, train_loss_nn, val_loss_nn = smart_train(HospitalNet(n_features, n_classes), "Neural Network")
    preds_nn, probs_nn, pred_time_nn = evaluate_model(model_nn)

    model_tf, train_time_tf, train_loss_tf, val_loss_tf = smart_train(TabularTransformer(n_features, n_classes), "Transformer")
    preds_tf, probs_tf, pred_time_tf = evaluate_model(model_tf)

    # ==========================================
    # 5. DASHBOARD GENERATION
    # ==========================================
    print("\n[STEP 4/5] Generating the Visual Dashboard")

    def get_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        return [acc, prec, rec, f1]

    metrics_nn = get_metrics(y_test, preds_nn)
    metrics_tf = get_metrics(y_test, preds_tf)
    metrics_xgb = get_metrics(y_test, preds_xgb)

    # Graph 1 : Performances
    print("  -> Creating Global Performance chart...")
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, metrics_nn, width, label='Neural Network', color='#457B9D')
    bars2 = ax.bar(x, metrics_tf, width, label='Transformer', color='#E63946')
    bars3 = ax.bar(x + width, metrics_xgb, width, label='XGBoost', color='#2A9D8F')
    ax.bar_label(bars1, padding=3, fmt='%.3f')
    ax.bar_label(bars2, padding=3, fmt='%.3f')
    ax.bar_label(bars3, padding=3, fmt='%.3f')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(graph_folder / "01_Global_Performances.png", bbox_inches='tight')
    plt.close()

    # Graph 2 : Computation Time
    print("  -> Creating Computation Time charts...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bar_time1 = ax1.bar(["Neural Net", "Transformer", "XGBoost"], [train_time_nn, train_time_tf, train_time_xgb], color=['#457B9D', '#E63946', '#2A9D8F'])
    ax1.bar_label(bar_time1, padding=3, fmt='%.1f')
    ax1.set_ylabel("Seconds")
    bar_time2 = ax2.bar(["Neural Net", "Transformer", "XGBoost"], [pred_time_nn, pred_time_tf, pred_time_xgb], color=['#457B9D', '#E63946', '#2A9D8F'])
    ax2.bar_label(bar_time2, padding=3, fmt='%.3f')
    plt.tight_layout()
    plt.savefig(graph_folder / "02_Computation_Time.png", bbox_inches='tight')
    plt.close()

    # Graph 3, 4, 5 : Confusion Matrices
    print("  -> Creating Confusion Matrices...")
    
    def plot_cm(preds, filename, cmap):
        # 1. On augmente considérablement la taille de la figure (ex: 12x12) 
        # pour donner de la place aux 24x24 cellules sans chevauchement.
        fig, ax = plt.subplots(figsize=(12, 12))
        
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, preds), display_labels=class_names)
        
        # 2. On augmente légèrement la taille de la police interne pour une lisibilité parfaite
        disp.plot(cmap=cmap, ax=ax, xticks_rotation=90, colorbar=False, text_kw={'fontsize': 8})
        
        # 3. Le titre a été supprimé car LaTeX s'en chargera.
        
        # 4. Ajustement des labels des axes (noms des maladies)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        # Nettoyage des labels X et Y par défaut si tu préfères que LaTeX gère tout
        plt.xlabel('Predicted label', fontsize=12)
        plt.ylabel('True label', fontsize=12)
        
        plt.tight_layout()
        
        # 5. dpi=400 permet à LaTeX de compresser l'image sans perte de qualité
        plt.savefig(graph_folder / filename, bbox_inches='tight', dpi=400)
        plt.close()

    # Appels de la fonction modifiée (sans l'argument 'title')
    plot_cm(preds_nn, "03_Confusion_Matrix_NN.png", "Blues")
    plot_cm(preds_tf, "04_Confusion_Matrix_Transformer.png", "Reds")
    plot_cm(preds_xgb, "05_Confusion_Matrix_XGBoost.png", "Greens")

    # Graph 6 : ROC Curves
    print("  -> Creating ROC Curves...")
    fpr_nn, tpr_nn, _ = roc_curve(y_test_bin.ravel(), probs_nn.ravel())
    fpr_tf, tpr_tf, _ = roc_curve(y_test_bin.ravel(), probs_tf.ravel())
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test_bin.ravel(), probs_xgb.ravel())
    plt.figure(figsize=(9, 7))
    plt.plot(fpr_nn, tpr_nn, color='#457B9D', lw=2, label=f'MLP (AUC = {auc(fpr_nn, tpr_nn):.3f})')
    plt.plot(fpr_tf, tpr_tf, color='#E63946', lw=2, label=f'Transformer (AUC = {auc(fpr_tf, tpr_tf):.3f})')
    plt.plot(fpr_xgb, tpr_xgb, color='#2A9D8F', lw=2, label=f'XGBoost (AUC = {auc(fpr_xgb, tpr_xgb):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(graph_folder / "06_ROC_Curves.png", bbox_inches='tight')
    plt.close()

    # Graph 7 : SHAP - Dual explainability
    print("\n[STEP 5/5] Generating SHAP Explainability Graphs (NN + XGBoost)")
    n_shap_samples = 300

    def aggregate_multiclass_shap(sv, n_classes_expected):
        if isinstance(sv, list):
            arr = np.abs(np.stack(sv, axis=0))
            return arr.mean(axis=0)

        sv = np.asarray(sv)
        if sv.ndim == 2:
            return np.abs(sv)
        if sv.ndim == 3:
            if sv.shape[-1] == n_classes_expected:
                return np.abs(sv).mean(axis=-1)
            elif sv.shape[0] == n_classes_expected:
                return np.abs(sv).mean(axis=0)
            else:
                return np.abs(sv).mean(axis=-1)
        raise ValueError(f"Unexpected SHAP values shape: {sv.shape}")

    def top_features(shap_agg, feature_names, k=8):
        importance = np.ravel(np.asarray(shap_agg).mean(axis=0))
        order = np.argsort(importance)[::-1][:k]
        return [feature_names[int(i)] for i in order]

    # --- 7a. Neural Network (DeepExplainer) ---
    print("  -> SHAP for Neural Network (DeepExplainer)...")
    model_nn.eval()
    background_sample = torch.FloatTensor(X_train_scaled[:100]).to(device)
    test_sample_nn = torch.FloatTensor(X_test_scaled[:n_shap_samples]).to(device)
    explainer_nn = shap.DeepExplainer(model_nn, background_sample)
    shap_values_nn = explainer_nn.shap_values(test_sample_nn)
    shap_values_nn_agg = aggregate_multiclass_shap(shap_values_nn, n_classes)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_values_nn_agg, X_test_scaled[:n_shap_samples],
        feature_names=X_raw.columns, show=False, plot_type="bar", max_display=12
    )
    plt.xlabel("Mean |SHAP| (Feature Impact)")
    plt.tight_layout()
    plt.savefig(graph_folder / "07a_SHAP_NeuralNetwork.png", bbox_inches='tight')
    plt.close()

    # --- 7b. XGBoost (TreeExplainer) ---
    print("  -> SHAP for XGBoost (TreeExplainer)...")
    explainer_xgb = shap.TreeExplainer(xgb_model)
    shap_values_xgb = explainer_xgb.shap_values(X_test_scaled[:n_shap_samples])
    shap_values_xgb_agg = aggregate_multiclass_shap(shap_values_xgb, n_classes)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_values_xgb_agg, X_test_scaled[:n_shap_samples],
        feature_names=X_raw.columns, show=False, plot_type="bar", max_display=12
    )
    plt.xlabel("Mean |SHAP| (Feature Impact)")
    plt.tight_layout()
    plt.savefig(graph_folder / "07b_SHAP_XGBoost.png", bbox_inches='tight')
    plt.close()

    top_nn = top_features(shap_values_nn_agg, list(X_raw.columns))
    top_xgb = top_features(shap_values_xgb_agg, list(X_raw.columns))
    overlap = set(top_nn) & set(top_xgb)
    print(f"  -> Top-8 NN features   : {top_nn}")
    print(f"  -> Top-8 XGBoost feats : {top_xgb}")
    print(f"  -> Overlap (top-8)     : {len(overlap)}/8 -> {sorted(overlap)}")


    # Graph 8 : Learning Curves
    print("  -> Creating remaining graphs (Learning curves, PR, Correlation)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_loss_nn, label='Training Loss', color='#457B9D')
    ax1.plot(val_loss_nn, label='Validation Loss', color='#E63946', linestyle='--')
    ax1.legend()
    ax2.plot(train_loss_tf, label='Training Loss', color='#457B9D')
    ax2.plot(val_loss_tf, label='Validation Loss', color='#E63946', linestyle='--')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(graph_folder / "08_Learning_Curves.png", bbox_inches='tight')
    plt.close()

    # Graph 9 : Precision-Recall Curves
    precision_nn, recall_nn, _ = precision_recall_curve(y_test_bin.ravel(), probs_nn.ravel())
    ap_nn = average_precision_score(y_test_bin, probs_nn, average="micro")
    precision_tf, recall_tf, _ = precision_recall_curve(y_test_bin.ravel(), probs_tf.ravel())
    ap_tf = average_precision_score(y_test_bin, probs_tf, average="micro")
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test_bin.ravel(), probs_xgb.ravel())
    ap_xgb = average_precision_score(y_test_bin, probs_xgb, average="micro")
    
    plt.figure(figsize=(9, 7))
    plt.plot(recall_nn, precision_nn, color='#457B9D', lw=2, label=f'MLP (AP = {ap_nn:.3f})')
    plt.plot(recall_tf, precision_tf, color='#E63946', lw=2, label=f'Transformer (AP = {ap_tf:.3f})')
    plt.plot(recall_xgb, precision_xgb, color='#2A9D8F', lw=2, label=f'XGBoost (AP = {ap_xgb:.3f})')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(graph_folder / "09_Precision_Recall_Curve.png", bbox_inches='tight')
    plt.close()

    # Graph 10 : Correlation Matrix
    plt.figure(figsize=(12, 10))
    corr_columns = ["Age", "HR", "Sys_BP", "RR", "Temp", "SpO2", "Troponin_ng_L", "BNP_pg_mL", "D_Dimers_ng_mL", "Lactates_mmol_L", "Leukocytes_G_L", "CRP_mg_L"]
    short_labels = ["Age", "HR", "Sys BP", "RR", "Temp", "SpO2", "Troponin", "BNP", "D-Dimers", "Lactates", "Leukocytes", "CRP"]
    corr_matrix = X_raw[corr_columns].corr()
    corr_matrix.columns = short_labels
    corr_matrix.index = short_labels
    sns.heatmap(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.75})
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig(graph_folder / "10_Correlation_Matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "="*70)
    print("🎉 PIPELINE EXECUTION COMPLETELY FINISHED!")
    print(f"All graphs have been saved to: {graph_folder}")
    print("="*70)