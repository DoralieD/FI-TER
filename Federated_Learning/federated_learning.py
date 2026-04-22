from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ==========================================================
# CONFIGURATION SIMPLE
# ==========================================================
# Version équilibrée : on cherche un meilleur compromis entre
# précision, recall et F1, au lieu de pousser seulement la précision.
ROUNDS = 8
LOCAL_EPOCHS = 5
ALPHA = 1e-4
RANDOM_SEED = 42
CALIBRATION_SIZE = 0.50
LEARNING_RATE = "adaptive"
ETA0 = 0.01
USE_AVERAGING = True
THRESHOLD_GRID = np.linspace(0.20, 0.85, 27)
MIN_PRECISION_FLOOR = 0.60
MIN_POSITIVE_PREDICTIONS = 20


def folder_root() -> Path:
    return Path(__file__).resolve().parent


class BinaryFederatedSGD:
    def __init__(self, n_features: int, alpha: float, seed: int):
        self.n_features = n_features
        self.alpha = alpha
        self.seed = seed
        self.classes_ = np.array([0, 1], dtype=int)
        self.model = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            random_state=seed,
            learning_rate=LEARNING_RATE,
            eta0=ETA0,
            average=USE_AVERAGING,
        )
        x_dummy = np.zeros((2, n_features), dtype=float)
        y_dummy = self.classes_.copy()
        self.model.partial_fit(x_dummy, y_dummy, classes=self.classes_)

    def clone(self) -> "BinaryFederatedSGD":
        clone = BinaryFederatedSGD(self.n_features, self.alpha, self.seed)
        clone.set_state(self.get_state())
        return clone

    def fit_local(self, x: np.ndarray, y: np.ndarray, local_epochs: int) -> None:
        if len(x) == 0:
            return
        rng = np.random.RandomState(self.seed)
        for _ in range(local_epochs):
            perm = rng.permutation(len(x))
            x_epoch = x[perm]
            y_epoch = y[perm]
            self.model.partial_fit(x_epoch, y_epoch, classes=self.classes_)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "coef_": self.model.coef_.copy(),
            "intercept_": self.model.intercept_.copy(),
        }

    def set_state(self, state: Dict[str, np.ndarray]) -> None:
        self.model.coef_ = state["coef_"].copy()
        self.model.intercept_ = state["intercept_"].copy()
        self.model.classes_ = self.classes_.copy()


def fedavg(states: List[Dict[str, np.ndarray]], weights: List[int]) -> Dict[str, np.ndarray]:
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()
    aggregated = {}
    for key in states[0].keys():
        stacked = np.stack([state[key] for state in states], axis=0)
        aggregated[key] = np.tensordot(weights_arr, stacked, axes=(0, 0))
    return aggregated


def load_data() -> Tuple[List[Tuple[str, np.ndarray, Dict[str, np.ndarray], pd.DataFrame]], np.ndarray, Dict[str, np.ndarray], pd.DataFrame, dict]:
    base_dir = folder_root() / "generated_federated_data" / "early_disease_prediction_non_iid"
    metadata = json.loads((base_dir / "metadata.json").read_text(encoding="utf-8"))
    features = metadata["feature_columns"]
    targets = metadata["target_columns"]

    clients = []
    for client_csv in sorted((base_dir / "clients").glob("client_*.csv")):
        df = pd.read_csv(client_csv)
        x = df[features].to_numpy(dtype=float)
        y_dict = {target: df[target].to_numpy(dtype=int) for target in targets}
        clients.append((client_csv.stem, x, y_dict, df))

    test_df = pd.read_csv(base_dir / "server_test.csv")
    x_test = test_df[features].to_numpy(dtype=float)
    y_test = {target: test_df[target].to_numpy(dtype=int) for target in targets}
    return clients, x_test, y_test, test_df, metadata


def split_server_sets(
    x_test: np.ndarray,
    y_test_dict: Dict[str, np.ndarray],
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    stratify_label = sum(y_test_dict[target] for target in y_test_dict.keys())
    stratify_label = np.clip(stratify_label, 0, 4)
    indices = np.arange(len(x_test))
    tune_idx, eval_idx = train_test_split(
        indices,
        test_size=1 - CALIBRATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify_label,
    )
    x_tune = x_test[tune_idx]
    x_eval = x_test[eval_idx]
    y_tune = {target: y_test_dict[target][tune_idx] for target in y_test_dict}
    y_eval = {target: y_test_dict[target][eval_idx] for target in y_test_dict}
    tune_df = test_df.iloc[tune_idx].reset_index(drop=True)
    eval_df = test_df.iloc[eval_idx].reset_index(drop=True)
    return x_tune, x_eval, y_tune, y_eval, tune_df, eval_df


def disease_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def choose_balanced_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, dict]:
    candidates = []
    for threshold in THRESHOLD_GRID:
        preds = (y_proba >= threshold).astype(int)
        positive_predictions = int(preds.sum())
        metrics = disease_metrics(y_true, preds)
        balanced_score = 0.40 * metrics["f1"] + 0.30 * metrics["recall"] + 0.30 * metrics["precision"]
        candidates.append({
            "threshold": float(threshold),
            "positive_predictions": positive_predictions,
            "balanced_score": float(balanced_score),
            **metrics,
        })

    viable = [
        row for row in candidates
        if row["positive_predictions"] >= MIN_POSITIVE_PREDICTIONS and row["precision"] >= MIN_PRECISION_FLOOR
    ]
    if not viable:
        viable = [row for row in candidates if row["positive_predictions"] >= MIN_POSITIVE_PREDICTIONS]
    if not viable:
        viable = candidates

    best = max(
        viable,
        key=lambda row: (
            row["balanced_score"],
            row["f1"],
            row["recall"],
            row["precision"],
            -row["threshold"],
        ),
    )
    return float(best["threshold"]), best


def save_outputs(result: dict, prediction_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(result["history"])
    history_df.to_csv(out_dir / "history.csv", index=False)
    prediction_df.to_csv(out_dir / "server_predictions.csv", index=False)
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["round"], history_df["mean_precision"], marker="o", label="Précision moyenne")
    plt.plot(history_df["round"], history_df["mean_recall"], marker="^", label="Recall moyen")
    plt.plot(history_df["round"], history_df["mean_f1"], marker="s", label="F1 moyen")
    plt.xlabel("Round fédéré")
    plt.ylabel("Score")
    plt.title("Prédiction précoce de maladies - FL équilibré")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "performance.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n=== Sorties ===")
    print(out_dir / "history.csv")
    print(out_dir / "server_predictions.csv")
    print(out_dir / "result.json")
    print(out_dir / "performance.png")


def main() -> None:
    clients, x_test, y_test_dict, test_df, metadata = load_data()
    x_tune, x_eval, y_tune_dict, y_eval_dict, tune_df, eval_df = split_server_sets(x_test, y_test_dict, test_df)
    targets = metadata["target_columns"]
    n_features = len(metadata["feature_columns"])

    global_models = {target: BinaryFederatedSGD(n_features, ALPHA, RANDOM_SEED) for target in targets}
    best_thresholds = {target: 0.5 for target in targets}
    best_threshold_reports = {target: {} for target in targets}
    history = []

    for round_idx in range(1, ROUNDS + 1):
        for target in targets:
            local_states = []
            local_weights = []
            for client_name, x_local, y_local_dict, _ in clients:
                local_model = global_models[target].clone()
                local_model.fit_local(x_local, y_local_dict[target], LOCAL_EPOCHS)
                local_states.append(local_model.get_state())
                local_weights.append(max(len(x_local), 1))
                print(f"Round {round_idx:02d} | {target} | {client_name} | samples={len(x_local)}")
            global_models[target].set_state(fedavg(local_states, local_weights))

        row = {"round": round_idx}
        precision_values = []
        f1_values = []
        recall_values = []

        for target in targets:
            tune_proba = global_models[target].predict_proba(x_tune)
            threshold, threshold_report = choose_balanced_threshold(y_tune_dict[target], tune_proba)
            best_thresholds[target] = threshold
            best_threshold_reports[target] = threshold_report

            eval_proba = global_models[target].predict_proba(x_eval)
            preds = (eval_proba >= threshold).astype(int)
            metrics = disease_metrics(y_eval_dict[target], preds)
            row[f"{target}_threshold"] = threshold
            row[f"{target}_accuracy"] = metrics["accuracy"]
            row[f"{target}_precision"] = metrics["precision"]
            row[f"{target}_recall"] = metrics["recall"]
            row[f"{target}_f1"] = metrics["f1"]
            precision_values.append(metrics["precision"])
            recall_values.append(metrics["recall"])
            f1_values.append(metrics["f1"])

        row["mean_precision"] = float(np.mean(precision_values))
        row["mean_recall"] = float(np.mean(recall_values))
        row["mean_f1"] = float(np.mean(f1_values))
        row["n_clients"] = metadata["n_clients"]
        history.append(row)
        print(
            f"[Round {round_idx:02d}] mean_precision={row['mean_precision']:.4f} "
            f"mean_f1={row['mean_f1']:.4f} mean_recall={row['mean_recall']:.4f}"
        )

    prediction_df = pd.DataFrame({"ID_Patient": eval_df["ID_Patient"]})
    prediction_df["split"] = "server_eval"
    for target in targets:
        threshold = best_thresholds[target]
        eval_proba = global_models[target].predict_proba(x_eval)
        prediction_df[f"true_{target}"] = y_eval_dict[target]
        prediction_df[f"pred_{target}"] = (eval_proba >= threshold).astype(int)
        prediction_df[f"proba_{target}"] = eval_proba
        prediction_df[f"threshold_{target}"] = threshold

    result = {
        "objective": metadata["objective"] + " Version équilibrée : le serveur choisit un seuil par maladie pour améliorer ensemble précision, recall et F1, plutôt que de pousser uniquement la précision.",
        "n_clients": metadata["n_clients"],
        "client_sizes": metadata["client_sizes"],
        "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "alpha": ALPHA,
        "seed": RANDOM_SEED,
        "learning_rate": LEARNING_RATE,
        "eta0": ETA0,
        "average_weights": USE_AVERAGING,
        "calibration_size": CALIBRATION_SIZE,
        "threshold_grid": [float(x) for x in THRESHOLD_GRID.tolist()],
        "history": history,
        "final_thresholds": best_thresholds,
        "threshold_selection_reports": best_threshold_reports,
        "why_it_is_federated": (
            "Chaque hôpital virtuel entraîne localement quatre modèles binaires (Cardio, Respi, Infectieux, Neuro). "
            "Le serveur n'agrège que les poids de ces modèles avec FedAvg, puis ajuste un seuil par maladie sur un jeu serveur de calibration."
        ),
        "outputs_description": {
            "history.csv": "évolution des scores par round fédéré, avec précision moyenne, recall moyen et F1 moyen",
            "server_predictions.csv": "prédictions finales sur le jeu serveur d'évaluation avec probabilité et seuil retenu par maladie",
            "result.json": "résumé complet de l'expérience et seuils optimisés pour un meilleur compromis global",
            "performance.png": "courbe de la précision moyenne, du recall moyen et du F1 moyen",
        },
    }
    out_dir = folder_root() / "generated_federated_data" / "early_disease_prediction_non_iid" / "outputs_balanced"
    save_outputs(result, prediction_df, out_dir)


if __name__ == "__main__":
    main()
