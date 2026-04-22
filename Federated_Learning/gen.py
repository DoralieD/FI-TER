from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# CONFIGURATION SIMPLE
# ==========================================================
# Le but est aligné avec ML_Triage.py : prédire précocement
# quatre risques/pathologies à partir des constantes vitales
# et observations initiales.
N_CLIENTS = 4
TEST_SIZE = 0.20
RANDOM_SEED = 42
MAX_ROWS = None  # ex: 3000 pour un test rapide

TARGET_COLUMNS = [
    "Verite_Cardio",
    "Verite_Respi",
    "Verite_Infectieux",
    "Verite_Neuro",
]

FEATURE_COLUMNS = [
    "FC", "Tension_Sys", "Tension_Dia", "FR", "Temp", "SpO2",
    "Obs_Paleur", "Obs_Cyanose", "Obs_Sueurs", "Obs_Inconscient",
    "Obs_Confusion", "Obs_Frissons", "Obs_Hemorragie", "Obs_DouleurThorax",
    "Obs_DetresseRespi", "Obs_Eruption", "Obs_TraumaPenetrant",
]

# Matrice de répartition non-IID :
# chaque ligne = un profil dominant, chaque colonne = un client.
# Les clients gardent un peu de tout, mais chaque client reçoit davantage
# d'un profil pathologique particulier.
NON_IID_ALLOCATION = {
    "Cardio":      [0.65, 0.15, 0.10, 0.10],
    "Respi":       [0.10, 0.65, 0.15, 0.10],
    "Infectieux":  [0.10, 0.10, 0.65, 0.15],
    "Neuro":       [0.15, 0.10, 0.10, 0.65],
    "Healthy":     [0.25, 0.25, 0.25, 0.25],
}

CLIENT_DESCRIPTIONS = {
    1: "Hôpital à dominante cardio",
    2: "Hôpital à dominante respiratoire",
    3: "Hôpital à dominante infectieuse",
    4: "Hôpital à dominante neurologique",
}


def folder_root() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return folder_root().parent


def find_dataset() -> Path:
    matches = sorted(project_root().rglob("dataset_prise_constante.csv"))
    if not matches:
        raise FileNotFoundError("Impossible de trouver dataset_prise_constante.csv dans FI-TER.")
    return matches[0]


def load_clean_dataframe() -> pd.DataFrame:
    path = find_dataset()
    df = pd.read_csv(path)
    expected = ["ID_Patient"] + FEATURE_COLUMNS + TARGET_COLUMNS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset: {missing}")

    df = df[expected].copy()
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        med = df[col].median()
        if pd.isna(med):
            med = 0.0
        df[col] = df[col].fillna(med)

    for col in TARGET_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        df[col] = df[col].clip(0, 1)

    if MAX_ROWS is not None and len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def standardize(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    stats = {}
    for col in FEATURE_COLUMNS:
        mean = float(train_scaled[col].mean())
        std = float(train_scaled[col].std(ddof=0))
        if std == 0 or np.isnan(std):
            std = 1.0
        train_scaled[col] = (train_scaled[col] - mean) / std
        test_scaled[col] = (test_scaled[col] - mean) / std
        stats[col] = {"mean": mean, "std": std}
    return train_scaled, test_scaled, stats


def assign_profile(row: pd.Series) -> str:
    if int(row["Verite_Cardio"]) == 1:
        return "Cardio"
    if int(row["Verite_Respi"]) == 1:
        return "Respi"
    if int(row["Verite_Infectieux"]) == 1:
        return "Infectieux"
    if int(row["Verite_Neuro"]) == 1:
        return "Neuro"
    return "Healthy"


def proportional_counts(n: int, proportions: List[float]) -> List[int]:
    raw = np.asarray(proportions, dtype=float)
    raw = raw / raw.sum()
    counts = np.floor(raw * n).astype(int)
    remainder = n - counts.sum()
    if remainder > 0:
        order = np.argsort(-(raw * n - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return counts.tolist()


def split_clients_non_iid(train_df: pd.DataFrame) -> List[pd.DataFrame]:
    if N_CLIENTS != 4:
        raise ValueError("Cette version non-IID est définie pour 4 clients/hôpitaux virtuels.")

    df = train_df.copy()
    df["profile"] = df.apply(assign_profile, axis=1)

    client_parts: Dict[int, List[pd.DataFrame]] = {i: [] for i in range(1, N_CLIENTS + 1)}

    for profile_name, proportions in NON_IID_ALLOCATION.items():
        subset = df[df["profile"] == profile_name].sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        counts = proportional_counts(len(subset), proportions)
        start = 0
        for client_idx, count in enumerate(counts, start=1):
            end = start + count
            if count > 0:
                piece = subset.iloc[start:end].copy()
                client_parts[client_idx].append(piece)
            start = end

    clients: List[pd.DataFrame] = []
    for idx in range(1, N_CLIENTS + 1):
        client_df = pd.concat(client_parts[idx], axis=0, ignore_index=True)
        client_df = client_df.sample(frac=1.0, random_state=RANDOM_SEED + idx).reset_index(drop=True)
        client_df.insert(1, "site_id", f"Hopital_{idx:02d}")
        client_df.insert(2, "site_profile", CLIENT_DESCRIPTIONS[idx])
        client_df = client_df.drop(columns=["profile"], errors="ignore")
        clients.append(client_df)
    return clients


def client_target_distribution(client_df: pd.DataFrame) -> dict:
    return {
        target: {
            "positifs": int(client_df[target].sum()),
            "taux_positif": float(client_df[target].mean()),
        }
        for target in TARGET_COLUMNS
    }


def main() -> None:
    dataset_path = find_dataset()
    df = load_clean_dataframe()

    # On stratifie sur le nombre total de pathologies positives pour garder un test raisonnablement équilibré.
    stratify_label = df[TARGET_COLUMNS].sum(axis=1).clip(upper=4)
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify_label,
    )

    train_df, test_df, scaling_stats = standardize(train_df, test_df)
    clients = split_clients_non_iid(train_df)

    out_dir = folder_root() / "generated_federated_data" / "early_disease_prediction_non_iid"
    clients_dir = out_dir / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)

    columns_to_save = ["ID_Patient", "site_id", "site_profile"] + FEATURE_COLUMNS + TARGET_COLUMNS
    for idx, client_df in enumerate(clients, start=1):
        client_df[columns_to_save].to_csv(clients_dir / f"client_{idx:02d}.csv", index=False)

    test_df[["ID_Patient"] + FEATURE_COLUMNS + TARGET_COLUMNS].to_csv(out_dir / "server_test.csv", index=False)

    metadata = {
        "objective": "Prédiction précoce multi-maladies (Cardio, Respi, Infectieux, Neuro) à partir des constantes vitales et observations initiales.",
        "source_dataset": str(dataset_path),
        "n_clients": N_CLIENTS,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "client_sizes": {f"client_{i:02d}": int(len(c)) for i, c in enumerate(clients, start=1)},
        "client_profiles": {f"client_{i:02d}": CLIENT_DESCRIPTIONS[i] for i in range(1, N_CLIENTS + 1)},
        "client_target_distributions": {f"client_{i:02d}": client_target_distribution(c) for i, c in enumerate(clients, start=1)},
        "test_size": TEST_SIZE,
        "seed": RANDOM_SEED,
        "max_rows": MAX_ROWS,
        "scaling_stats": scaling_stats,
        "non_iid_allocation": NON_IID_ALLOCATION,
        "federated_learning_definition": (
            "Chaque client représente un hôpital virtuel avec une distribution différente des pathologies. "
            "Le serveur garde un jeu de test global, mais pendant l'entraînement, "
            "chaque client entraîne localement ses modèles puis envoie seulement ses poids au serveur."
        ),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("=== Génération terminée (non-IID) ===")
    print(f"Objectif : {metadata['objective']}")
    print(f"Dataset : {dataset_path}")
    print(f"Clients utilisés : {N_CLIENTS}")
    for i, client_df in enumerate(clients, start=1):
        print(f"  - client_{i:02d} | {CLIENT_DESCRIPTIONS[i]} | {len(client_df)} lignes")
        for target in TARGET_COLUMNS:
            rate = client_df[target].mean()
            print(f"      {target}: taux positif={rate:.3f} ({int(client_df[target].sum())} positifs)")
    print(f"Jeu de test serveur : {len(test_df)} lignes")
    print(f"Dossier généré : {out_dir}")


if __name__ == "__main__":
    main()
