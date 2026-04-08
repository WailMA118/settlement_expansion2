# ═══════════════════════════════════════════════════════════════
# train_model.py — XGBoost training + serialisation
# ═══════════════════════════════════════════════════════════════

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

from config import (
    FEATURE_COLS, MODEL_PATH,
    SCALER_PATH, FEATURES_PATH,
)


# ═══════════════════════════════════════════════════════════════
# Silver-label generator
# ═══════════════════════════════════════════════════════════════

class SilverLabelGenerator:
    """
    Creates binary training labels using expert rules.

    A settlement is labelled 1 (high expansion risk) when ≥ 2 of
    7 conditions are met.  This is the silver-label strategy used
    before verified ground-truth is available.

    When real historical expansion records become available,
    replace this class with a GroundTruthLabelLoader.
    """

    MIN_CONDITIONS = 2

    def generate(self, df: pd.DataFrame) -> pd.Series:
        conditions = pd.DataFrame({
            "fast_growth"   : df["growth_rate_m2yr"]    > df["growth_rate_m2yr"].median(),
            "has_confiscation": df["n_conf_total"]       > 0,
            "high_leaked"   : df["leaked_ratio"]         > 0.10,
            "zone_c_overlap": df["zone_c_coverage"]      > 0.30,
            "recent_decade" : df["decade_growth_frac"]   > 0.15,
            "recent_conf"   : df["n_conf_recent"]        > 0,
            "road_pressure" : df["road_expansion_index"] > df["road_expansion_index"].median(),
        })
        score = conditions.sum(axis=1)
        labels = (score >= self.MIN_CONDITIONS).astype(int)

        pos = labels.sum()
        neg = (labels == 0).sum()
        print(f"    Silver labels → positive={pos}  negative={neg}")
        return labels


# ═══════════════════════════════════════════════════════════════
# XGBoost trainer
# ═══════════════════════════════════════════════════════════════

class SettlementExpansionModel:
    """
    Wraps XGBoost training, evaluation, and serialisation.

    Designed for Palestine-wide use.
    Trained here on Ramallah data as a pilot.
    """

    def __init__(self):
        self.model   = None
        self.scaler  = None
        self._label_gen = SilverLabelGenerator()

    # ── Training ──────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> np.ndarray:
        """
        Train on feature matrix df.
        Returns predicted probabilities for the training set.
        """
        print("[train_model] Training XGBoost …")

        y = self._label_gen.generate(df)
        X = df[FEATURE_COLS].copy()

        self.scaler = StandardScaler()
        X_sc = self.scaler.fit_transform(X)

        pos = y.sum()
        neg = (y == 0).sum()
        spw = max(neg / max(pos, 1), 1.0)

        self.model = xgb.XGBClassifier(
            n_estimators      = 300,
            max_depth         = 4,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            scale_pos_weight  = spw,
            eval_metric       = "logloss",
            random_state      = 42,
        )

        self._cross_validate(X_sc, y)

        self.model.fit(X_sc, y, eval_set=[(X_sc, y)], verbose=False)

        proba = self.model.predict_proba(X_sc)[:, 1]
        self._print_report(y, proba)

        return proba

    def _cross_validate(self, X_sc, y):
        """
        Leave-One-Out CV — correct choice for small datasets (≤ 30 rows).
        Standard k-fold would leave only 3 samples per fold with 15 rows.
        """
        n = len(y)
        if n < 4:
            print("    Too few samples for CV — skipping")
            return

        cv = LeaveOneOut()
        try:
            scores = cross_val_score(
                self.model, X_sc, y,
                cv=cv, scoring="roc_auc"
            )
            print(f"    LOO-CV AUC = {scores.mean():.3f} ± {scores.std():.3f}  "
                  f"(n={n} settlements)")
        except Exception as e:
            print(f"    CV warning: {e}")

    def _print_report(self, y, proba):
        y_pred = (proba >= 0.5).astype(int)
        print("\n    Classification report (train set):")
        print(classification_report(y, y_pred,
                                    target_names=["low_risk", "high_risk"],
                                    zero_division=0))
        if len(np.unique(y)) > 1:
            auc = roc_auc_score(y, proba)
            print(f"    Train AUC = {auc:.3f}")

    # ── Inference ─────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return risk probabilities for a feature matrix."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained. Call train() or load().")
        X    = df[FEATURE_COLS].copy().fillna(0)
        X_sc = self.scaler.transform(X)
        return self.model.predict_proba(X_sc)[:, 1]

    def predict_single(self, feature_dict: dict) -> float:
        """Return risk probability for a single settlement (as dict)."""
        row = {col: feature_dict.get(col, 0) for col in FEATURE_COLS}
        df  = pd.DataFrame([row])
        return float(self.predict(df)[0])

    # ── Feature importance ────────────────────────────────────

    def feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        imp = pd.DataFrame({
            "feature"   : FEATURE_COLS,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).head(top_n)
        return imp

    # ── Persistence ───────────────────────────────────────────

    def save(self):
        """Save model, scaler, and feature list to disk."""
        with open(MODEL_PATH,    "wb") as f: pickle.dump(self.model,   f)
        with open(SCALER_PATH,   "wb") as f: pickle.dump(self.scaler,  f)
        with open(FEATURES_PATH, "wb") as f: pickle.dump(FEATURE_COLS, f)
        print(f"    Model  saved → {MODEL_PATH}")
        print(f"    Scaler saved → {SCALER_PATH}")

    def load(self):
        """Load model and scaler from disk."""
        with open(MODEL_PATH,    "rb") as f: self.model  = pickle.load(f)
        with open(SCALER_PATH,   "rb") as f: self.scaler = pickle.load(f)
        print(f"    Model  loaded ← {MODEL_PATH}")
        print(f"    Scaler loaded ← {SCALER_PATH}")
        return self


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def run_training():
    """
    Full training pipeline:
      1. Load data
      2. Build feature matrix
      3. Train XGBoost
      4. Save model to disk
    """
    from data_loader import get_engine, load_all
    from feature_engineering import FeatureMatrix

    print("\n" + "═" * 55)
    print("  SETTLEMENT EXPANSION MODEL — TRAINING")
    print("  Pilot data: Ramallah Governorate")
    print("  Model scope: Palestine-wide")
    print("═" * 55 + "\n")

    engine = get_engine()
    data   = load_all(engine)
    df     = FeatureMatrix().build(data)

    model  = SettlementExpansionModel()
    proba  = model.train(df)
    model.save()

    print("\n  Top 10 feature importances:")
    print(model.feature_importance(10).to_string(index=False))

    print("\n" + "═" * 55)
    print("  Training complete. Model saved.")
    print("═" * 55 + "\n")

    return model, df, proba


if __name__ == "__main__":
    run_training()