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
warnings.filterwarnings('ignore') # To hide minor SHAP warnings

# ==========================================
# 1. PREPARATION AND CONFIGURATION
# ==========================================
if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    
    print("\n" + "="*60)
    print(f"SCRIPT : ADVANCED DL + XAI + COMPLETE DASHBOARD")
    print(f"Hardware : {device} ({num_cores} cores)")
    print("="*60)

    BASE_DIR = Path(__file__).resolve().parent
    data_path = BASE_DIR / "Synthetic_Data" / "super_hospital_file.csv"
    graph_folder = BASE_DIR / "Graph_Folder" / "Dashboard_Ultimate"
    graph_folder.mkdir(parents=True, exist_ok=True)

    print("-> Loading and Cleaning file...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file {data_path} was not found.")
        exit()
    
    # Merge targets into a single Global Diagnosis
    df['Global_Diagnosis'] = df['Final_Diagnosis_Cardio'] \
        .combine_first(df['Final_Diagnosis_Infectious']) \
        .combine_first(df['Final_Diagnosis_Neuro']) \
        .combine_first(df['Final_Diagnosis_Respi'])

    df = df.dropna(subset=['Global_Diagnosis'])
    df = df.fillna(0)

    cols_to_remove = ["Patient_ID", "Global_Diagnosis"]
    cols_to_remove += [col for col in df.columns if "Final_Diagnosis" in col or "Truth_" in col]

    X_raw = df.drop(columns=cols_to_remove, errors="ignore")
    y_raw = df["Global_Diagnosis"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    class_names = encoder.classes_
    n_classes = len(class_names)
    n_features = X_raw.shape[1]

    # Binarization for ROC Curves
    y_binarized = label_binarize(y_encoded, classes=np.arange(n_classes))

    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_encoded, test_size=0.2, random_state=42)
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
    # 3. SMART TRAINING & EVALUATION
    # ==========================================
    def smart_train(model, model_name, epochs=30, patience=4):
        print(f"\nTraining {model_name} (Early Stopping max {epochs} epochs)...")
        start_time = time.time()
        model.to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        
        history_train = []
        history_val = []

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

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"-> STOP: Stagnation reached at epoch {epoch+1}.")
                    break
        
        if best_weights is not None:
            model.load_state_dict(best_weights)
            
        total_time = time.time() - start_time
        print(f"-> Finished in {total_time:.1f} sec.")
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
        pred_time = time.time() - start_time
        return np.array(all_preds), np.array(all_probs), pred_time

    # Execute both Champions
    model_nn, train_time_nn, train_loss_nn, val_loss_nn = smart_train(HospitalNet(n_features, n_classes), "Neural Network")
    preds_nn, probs_nn, pred_time_nn = evaluate_model(model_nn)

    model_tf, train_time_tf, train_loss_tf, val_loss_tf = smart_train(TabularTransformer(n_features, n_classes), "Transformer")
    preds_tf, probs_tf, pred_time_tf = evaluate_model(model_tf)

    # ==========================================
    # 4. DASHBOARD GENERATION
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
    plt.savefig(graph_folder / "01_Global_Performances.png", bbox_inches='tight')
    plt.close()

    # Graph 2 : Computation Time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bar_time1 = ax1.bar(["Neural Net", "Transformer"], [train_time_nn, train_time_tf], color=['#457B9D', '#E63946'])
    ax1.bar_label(bar_time1, padding=3, fmt='%.1f')
    ax1.set_title("Total Training Time", pad=15)
    ax1.set_ylabel("Seconds")
    
    bar_time2 = ax2.bar(["Neural Net", "Transformer"], [pred_time_nn, pred_time_tf], color=['#457B9D', '#E63946'])
    ax2.bar_label(bar_time2, padding=3, fmt='%.3f')
    ax2.set_title("Inference Time", pad=15)
    plt.tight_layout()
    plt.savefig(graph_folder / "02_Computation_Time.png", bbox_inches='tight')
    plt.close()

    # Graph 3 : Confusion Matrix NN
    fig, ax = plt.subplots(figsize=(18, 16))
    disp_nn = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, preds_nn), display_labels=class_names)
    disp_nn.plot(cmap="Blues", ax=ax, xticks_rotation=90, colorbar=False)
    plt.title("Confusion Matrix - Neural Network", fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(graph_folder / "03_Confusion_Matrix_NN.png", bbox_inches='tight')
    plt.close()

    # Graph 4 : Confusion Matrix TF
    fig, ax = plt.subplots(figsize=(18, 16))
    disp_tf = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, preds_tf), display_labels=class_names)
    disp_tf.plot(cmap="Reds", ax=ax, xticks_rotation=90, colorbar=False)
    plt.title("Confusion Matrix - Transformer", fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(graph_folder / "04_Confusion_Matrix_Transformer.png", bbox_inches='tight')
    plt.close()

    # Graph 5 : ROC Curves
    fpr_nn, tpr_nn, _ = roc_curve(y_test_bin.ravel(), probs_nn.ravel())
    fpr_tf, tpr_tf, _ = roc_curve(y_test_bin.ravel(), probs_tf.ravel())
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
    plt.savefig(graph_folder / "05_ROC_Curves.png", bbox_inches='tight')
    plt.close()

    # Graph 6 : SHAP
    print("\nGenerating SHAP explanations (This may take a moment)...")
    model_nn.eval()
    background_sample = torch.FloatTensor(X_train_scaled[:100]).to(device)
    test_sample = torch.FloatTensor(X_test_scaled[:100]).to(device)
    explainer = shap.DeepExplainer(model_nn, background_sample)
    shap_values = explainer.shap_values(test_sample)
    
    plt.figure(figsize=(14, 10))
    # SHAP will automatically use the English feature names from the dataframe
    shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=X_raw.columns, show=False)
    plt.title("Feature Importance (SHAP Summary)", pad=20, fontsize=16)
    plt.tight_layout()
    plt.savefig(graph_folder / "06_SHAP_Explicability.png", bbox_inches='tight')
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
    plt.savefig(graph_folder / "07_Learning_Curves.png", bbox_inches='tight')
    plt.close()

    # ==========================================
    # 5. ADVANCED GRAPHS (ARCHITECTURE, PR CURVE, CORRELATION)
    # ==========================================
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import seaborn as sns
    import matplotlib.patches as patches

   # Graph 8 : Architecture Schema (Matplotlib conceptual drawing)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # MLP Boxes (Left)
    ax.add_patch(patches.FancyBboxPatch((0.1, 0.7), 0.3, 0.15, facecolor='#457B9D', edgecolor='black', boxstyle='round,pad=0.02'))
    ax.text(0.25, 0.775, 'Multimodal Inputs\n(Vitals + Labs)', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.add_patch(patches.FancyBboxPatch((0.1, 0.45), 0.3, 0.15, facecolor='#A8DADC', edgecolor='black', boxstyle='round,pad=0.02'))
    ax.text(0.25, 0.525, 'Concatenation\n(Early Fusion)', color='black', ha='center', va='center', fontsize=12)

    ax.add_patch(patches.FancyBboxPatch((0.1, 0.2), 0.3, 0.15, facecolor='#1D3557', edgecolor='black', boxstyle='round,pad=0.02'))
    ax.text(0.25, 0.275, 'MLP Layers\n(256 -> 128 -> 24)', color='white', ha='center', va='center', fontsize=12, fontweight='bold')

    # MLP Arrows
    ax.annotate('', xy=(0.25, 0.6), xytext=(0.25, 0.7), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate('', xy=(0.25, 0.35), xytext=(0.25, 0.45), arrowprops=dict(arrowstyle="->", lw=2))

    # Transformer Boxes (Right)
    ax.add_patch(patches.FancyBboxPatch((0.6, 0.7), 0.3, 0.15, facecolor='#E63946', edgecolor='black', boxstyle='round,pad=0.02'))
    ax.text(0.75, 0.775, 'Multimodal Inputs\n(Vitals + Labs)', color='white', ha='center', va='center', fontsize=12, fontweight='bold')

    ax.add_patch(patches.FancyBboxPatch((0.6, 0.45), 0.3, 0.15, facecolor='#F1FAEE', edgecolor='black', boxstyle='round,pad=0.02'))
    ax.text(0.75, 0.525, 'Feature Tokenizer\n+ [CLS] Token', color='black', ha='center', va='center', fontsize=12)

    ax.add_patch(patches.FancyBboxPatch((0.6, 0.2), 0.3, 0.15, facecolor='#9B2226', edgecolor='black', boxstyle='round,pad=0.02'))
    ax.text(0.75, 0.275, 'Multi-Head Attention\n(Contextual Fusion)', color='white', ha='center', va='center', fontsize=12, fontweight='bold')

    # Transformer Arrows
    ax.annotate('', xy=(0.75, 0.6), xytext=(0.75, 0.7), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate('', xy=(0.75, 0.35), xytext=(0.75, 0.45), arrowprops=dict(arrowstyle="->", lw=2))

    # Global Titles
    ax.text(0.25, 0.9, 'Baseline MLP (Early Fusion)', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(0.75, 0.9, 'Tabular Transformer (Contextual Fusion)', ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(graph_folder / "08_Architecture_Schema.png", bbox_inches='tight')
    plt.close()

    # Graph 9 : Precision-Recall Curves
    precision_nn, recall_nn, _ = precision_recall_curve(y_test_bin.ravel(), probs_nn.ravel())
    ap_nn = average_precision_score(y_test_bin, probs_nn, average="micro")
    
    precision_tf, recall_tf, _ = precision_recall_curve(y_test_bin.ravel(), probs_tf.ravel())
    ap_tf = average_precision_score(y_test_bin, probs_tf, average="micro")

    plt.figure(figsize=(9, 7))
    plt.plot(recall_nn, precision_nn, color='#457B9D', lw=2, label=f'Neural Net (AP = {ap_nn:.3f})')
    plt.plot(recall_tf, precision_tf, color='#E63946', lw=2, label=f'Transformer (AP = {ap_tf:.3f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title('Precision-Recall Curve (Micro-Averaged)', pad=15)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(graph_folder / "09_Precision_Recall_Curve.png", bbox_inches='tight')
    plt.close()

    # Graph 10 : Correlation Matrix (Synthetic Data Validation)

    plt.figure(figsize=(12, 10))

    # Representative and readable subset of variables
    corr_columns = [
        "Age", "HR", "Sys_BP", "RR", "Temp", "SpO2",
        "Troponin_ng_L", "BNP_pg_mL", "D_Dimers_ng_mL",
        "Lactates_mmol_L", "Leukocytes_G_L", "CRP_mg_L"
    ]

    # Short labels for readability in the paper
    short_labels = [
        "Age", "HR", "Sys BP", "RR", "Temp", "SpO2",
        "Troponin", "BNP", "D-Dimers", "Lactates", "Leukocytes", "CRP"
    ]

    corr_matrix = X_raw[corr_columns].corr()
    corr_matrix.columns = short_labels
    corr_matrix.index = short_labels

    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75}
    )

    plt.title("Physiological Correlation Matrix", fontsize=16, pad=18)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()

    plt.savefig(graph_folder / "10_Correlation_Matrix.png", dpi=300, bbox_inches="tight")
    plt.close()