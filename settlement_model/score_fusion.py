# ═══════════════════════════════════════════════════════════════
# score_fusion.py — Composite risk scoring + DB alert writer
# ═══════════════════════════════════════════════════════════════

import json
import numpy as np
import pandas as pd
from sqlalchemy import text

from config import (
    W_XGBOOST, W_GROWTH, W_SPATIAL,
    THR_CRITICAL, THR_HIGH, THR_MEDIUM,
)


def _severity(score: float) -> str:
    if score >= THR_CRITICAL: return "critical"
    if score >= THR_HIGH:     return "high"
    if score >= THR_MEDIUM:   return "medium"
    return "low"


class RiskFusion:
    """
    Merges three risk signals into one composite score:
      - xgb_risk     : XGBoost probability    (weight 0.50)
      - ts_risk      : Time-series growth norm (weight 0.30)
      - spatial_risk : Spatial pressure index  (weight 0.20)
    """

    def fuse(self, df: pd.DataFrame,
             xgb_proba: np.ndarray,
             ts_df: pd.DataFrame) -> pd.DataFrame:

        df = df.merge(
            ts_df[["settlement_id", "forecast_norm_growth",
                   "forecast_5yr_area", "forecast_series"]],
            on="settlement_id", how="left"
        )

        df["xgb_risk"] = xgb_proba
        df["ts_risk"]  = df["forecast_norm_growth"].fillna(0)

        # Spatial: normalise legal + leakage + road to [0, 1]
        sp = (df["legal_pressure_index"].fillna(0) +
              df["leakage_pressure"].fillna(0) * 5 +
              df["road_expansion_index"].fillna(0))
        sp_range = sp.max() - sp.min() + 1e-6
        df["spatial_risk"] = (sp - sp.min()) / sp_range

        df["composite_risk"] = (
            W_XGBOOST * df["xgb_risk"]    +
            W_GROWTH  * df["ts_risk"]     +
            W_SPATIAL * df["spatial_risk"]
        )

        df["severity"] = df["composite_risk"].apply(_severity)

        return df.sort_values("composite_risk", ascending=False).reset_index(drop=True)


class AlertWriter:
    """Writes high/critical risk results to ai_alerts table."""

    def write(self, df: pd.DataFrame, engine) -> int:
        alerts = df[df["severity"].isin(["critical", "high"])].copy()
        count  = 0

        with engine.begin() as conn:
            for _, row in alerts.iterrows():
                payload = {
                    "settlement_id"    : int(row["settlement_id"]),
                    "composite_risk"   : round(float(row["composite_risk"]), 4),
                    "xgb_risk"         : round(float(row["xgb_risk"]), 4),
                    "ts_risk"          : round(float(row["ts_risk"]), 4),
                    "growth_rate_m2yr" : float(row.get("growth_rate_m2yr", 0)),
                    "n_conf_total"     : int(row.get("n_conf_total", 0)),
                    "leaked_ratio"     : round(float(row.get("leaked_ratio", 0)), 4),
                    "zone_c_coverage"  : round(float(row.get("zone_c_coverage", 0)), 4),
                    "forecast_5yr_area": float(row.get("forecast_5yr_area", 0)),
                    "model_version"    : "settlement_expansion_v1.0",
                }
                conn.execute(text("""
                    INSERT INTO ai_alerts
                        (alert_type, title, description,
                         severity, ai_payload, status, created_by)
                    VALUES
                        ('high_activity', :title, :desc,
                         :sev, :payload, 'new', 'AI_SETTLEMENT_MODEL')
                """), {
                    "title"  : f"Settlement expansion risk: {row.get('name', '?')}",
                    "desc"   : (f"Composite risk: {row['composite_risk']:.2%}. "
                                f"Predicted expansion based on confiscation orders, "
                                f"leaked parcels, road growth and time-series forecast."),
                    "sev"    : row["severity"],
                    "payload": json.dumps(payload),
                })
                count += 1

        print(f"    {count} alerts written to ai_alerts")
        return count