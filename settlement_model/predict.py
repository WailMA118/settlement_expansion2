# ═══════════════════════════════════════════════════════════════
# predict.py — Predict expansion risk for ONE specific settlement
# Usage:
#   python predict.py --id 3
#   python predict.py --name "Beit El"
# ═══════════════════════════════════════════════════════════════

import argparse
import json
import pandas as pd
import os
from pathlib import Path

from config import THR_CRITICAL, THR_HIGH, THR_MEDIUM, SEVERITY_COLOR, OUTPUT_DIR


# ── Prediction runner ─────────────────────────────────────────

class SingleSettlementPredictor:
    """
    Loads a saved model and runs the full prediction pipeline
    for exactly one settlement.

    Outputs:
      - Console report with all scores and top risk indicators
      - Forecast series (area per year)
    """

    def __init__(self):
        from data_loader import get_engine, load_all
        from feature_engineering import FeatureMatrix
        from train_model import SettlementExpansionModel
        from timeseries import TimeSeriesForecaster
        from score_fusion import RiskFusion

        self.engine      = get_engine()
        self._data       = None
        self._df         = None
        self.model       = SettlementExpansionModel().load()
        self.ts          = TimeSeriesForecaster()
        self.fusion      = RiskFusion()
        self._data_loader_cls    = load_all
        self._feature_matrix_cls = FeatureMatrix

    def _ensure_data(self):
        if self._data is None:
            self._data = self._data_loader_cls(self.engine)
            self._df   = self._feature_matrix_cls().build(self._data)

    def predict(self, settlement_id: int = None,
                name: str = None) -> dict:
        """
        Run prediction for a single settlement.
        Identify it by settlement_id OR name (case-insensitive).
        """
        self._ensure_data()
        df = self._df

        # ── Identify row ──────────────────────────────────────
        if settlement_id is not None:
            mask = df["settlement_id"] == settlement_id
        elif name is not None:
            mask = df["name"].str.lower() == name.lower()
        else:
            raise ValueError("Provide settlement_id or name.")

        if not mask.any():
            raise ValueError(f"Settlement not found: id={settlement_id} name={name}")

        row_df = df[mask].copy().reset_index(drop=True)
        sid    = int(row_df["settlement_id"].iloc[0])

        # ── XGBoost score ─────────────────────────────────────
        xgb_risk = float(self.model.predict(row_df)[0])

        # ── Time-series forecast ──────────────────────────────
        area_map = {sid: float(row_df["area_latest_m2"].iloc[0])}
        ts_df    = self.ts.forecast_all(
            [sid], self._data["expansion_history"], area_map
        )

        # ── Score fusion ──────────────────────────────────────
        import numpy as np
        result_df = self.fusion.fuse(row_df, np.array([xgb_risk]), ts_df)
        result    = result_df.iloc[0].to_dict()

        # Parse forecast series
        fc_series = {}
        raw = result.get("forecast_series", "{}")
        if isinstance(raw, str):
            fc_series = {int(k): v for k, v in json.loads(raw).items()}

        return {
            "settlement_id"    : sid,
            "name"             : result.get("name", "?"),
            "type"             : result.get("type", "?"),
            "established_year" : result.get("established_year"),
            "composite_risk"   : round(result["composite_risk"], 4),
            "xgb_risk"         : round(result["xgb_risk"], 4),
            "ts_risk"          : round(result["ts_risk"], 4),
            "spatial_risk"     : round(result["spatial_risk"], 4),
            "severity"         : result["severity"],
            # Key features
            "area_latest_m2"   : round(result.get("area_latest_m2", 0), 1),
            "growth_rate_m2yr" : round(result.get("growth_rate_m2yr", 0), 1),
            "forecast_5yr_area": round(result.get("forecast_5yr_area", 0), 1),
            "n_conf_total"     : int(result.get("n_conf_total", 0)),
            "n_conf_recent"    : int(result.get("n_conf_recent", 0)),
            "leaked_ratio"     : round(result.get("leaked_ratio", 0), 4),
            "zone_c_coverage"  : round(result.get("zone_c_coverage", 0), 4),
            "n_transactions"   : int(result.get("n_transactions", 0)),
            "road_length_m"    : round(result.get("road_length_m", 0), 1),
            # Derived
            "expansion_momentum"    : round(result.get("expansion_momentum", 0), 6),
            "legal_pressure_index"  : round(result.get("legal_pressure_index", 0), 4),
            "leakage_pressure"      : round(result.get("leakage_pressure", 0), 4),
            # Forecast
            "forecast_series"  : fc_series,
        }

    def top_features(self, n: int = 10) -> pd.DataFrame:
        """Return top-n feature importances from the loaded model."""
        return self.model.feature_importance(n)


# ── Console report ────────────────────────────────────────────

def _print_report(result: dict):
    sev   = result["severity"]
    color_map = {"critical":"🔴","high":"🟠","medium":"🟡","low":"🟢"}
    icon  = color_map.get(sev, "⚪")

    print("\n" + "═" * 60)
    print(f"  SETTLEMENT EXPANSION RISK REPORT")
    print("═" * 60)
    print(f"  Name            : {result['name']}")
    print(f"  Type            : {result['type']}")
    print(f"  Established     : {result['established_year']}")
    print(f"  ID              : {result['settlement_id']}")
    print("─" * 60)
    print(f"  {icon} Severity        : {sev.upper()}")
    print(f"  Composite risk  : {result['composite_risk']:.1%}")
    print(f"    ├─ XGBoost    : {result['xgb_risk']:.3f}")
    print(f"    ├─ TimeSeries : {result['ts_risk']:.3f}")
    print(f"    └─ Spatial    : {result['spatial_risk']:.3f}")
    print("─" * 60)
    print("  SPATIAL INDICATORS")
    print(f"  Current area    : {result['area_latest_m2']:>12,.0f} m²  "
          f"({result['area_latest_m2']/10000:.2f} ha)")
    print(f"  Growth rate     : {result['growth_rate_m2yr']:>12,.0f} m²/yr")
    print(f"  Forecast (5yr)  : {result['forecast_5yr_area']:>12,.0f} m²")
    print(f"  Confiscations   : {result['n_conf_total']:>3}  "
          f"(recent: {result['n_conf_recent']})")
    print(f"  Leaked ratio    : {result['leaked_ratio']:.1%} of nearby parcels")
    print(f"  Zone-C coverage : {result['zone_c_coverage']:.1%}")
    print(f"  Road length     : {result['road_length_m']:>10,.0f} m nearby")
    print(f"  Transactions    : {result['n_transactions']:>3}")
    print("─" * 60)
    print("  COMPOSITE INDICATORS")
    print(f"  Expansion momentum   : {result['expansion_momentum']:.6f}")
    print(f"  Legal pressure index : {result['legal_pressure_index']:.4f}")
    print(f"  Leakage pressure     : {result['leakage_pressure']:.4f}")
    print("─" * 60)

    fc = result.get("forecast_series", {})
    if fc:
        print("  AREA FORECAST (m²)")
        for yr in sorted(fc.keys()):
            bar = "█" * int(fc[yr] / max(fc.values()) * 20)
            print(f"    {yr} │ {fc[yr]:>12,.0f}  {bar}")

    print("═" * 60 + "\n")


# ── HTML Report Generator ────────────────────────────────────

def _generate_html_report(result: dict) -> str:
    """
    Generate an embeddable HTML section with settlement expansion risk summary.
    Returns HTML string that can be embedded in other pages.
    """
    sev = result["severity"]
    severity_colors = {
        "critical": "#d32f2f",
        "high": "#f57c00",
        "medium": "#fbc02d",
        "low": "#388e3c"
    }
    sev_color = severity_colors.get(sev, "#9e9e9e")
    
    # Calculate area growth percentage
    area_growth_pct = 0
    if result["area_latest_m2"] > 0:
        area_growth_pct = (result["forecast_5yr_area"] - result["area_latest_m2"]) / result["area_latest_m2"] * 100
    
    html = f"""
<div class="settlement-report" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 24px; background: #f5f5f5; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
    
    <!-- Header -->
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; border-bottom: 3px solid {sev_color}; padding-bottom: 16px;">
        <div>
            <h1 style="margin: 0 0 4px 0; font-size: 28px; color: #1a1a1a;">{result['name']}</h1>
            <p style="margin: 0; font-size: 14px; color: #666;">ID: {result['settlement_id']} • {result['type']} • Est. {result['established_year']}</p>
        </div>
        <div style="background: {sev_color}; color: white; padding: 12px 20px; border-radius: 6px; text-align: center;">
            <div style="font-size: 24px; font-weight: bold;">{result['composite_risk']:.1%}</div>
            <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">{sev}</div>
        </div>
    </div>
    
    <!-- Risk Components -->
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 24px;">
        <div style="background: white; padding: 12px; border-radius: 6px; border-left: 4px solid #2196F3;">
            <div style="font-size: 12px; color: #666; text-transform: uppercase; font-weight: 500;">XGBoost Risk</div>
            <div style="font-size: 18px; font-weight: bold; color: #1a1a1a; margin-top: 4px;">{result['xgb_risk']:.3f}</div>
        </div>
        <div style="background: white; padding: 12px; border-radius: 6px; border-left: 4px solid #4CAF50;">
            <div style="font-size: 12px; color: #666; text-transform: uppercase; font-weight: 500;">TimeSeries Risk</div>
            <div style="font-size: 18px; font-weight: bold; color: #1a1a1a; margin-top: 4px;">{result['ts_risk']:.3f}</div>
        </div>
        <div style="background: white; padding: 12px; border-radius: 6px; border-left: 4px solid #FF9800;">
            <div style="font-size: 12px; color: #666; text-transform: uppercase; font-weight: 500;">Spatial Risk</div>
            <div style="font-size: 18px; font-weight: bold; color: #1a1a1a; margin-top: 4px;">{result['spatial_risk']:.3f}</div>
        </div>
    </div>
    
    <!-- Key Metrics -->
    <div style="background: white; padding: 16px; border-radius: 6px; margin-bottom: 24px;">
        <h3 style="margin: 0 0 12px 0; font-size: 14px; text-transform: uppercase; color: #666; font-weight: 600;">Spatial Indicators</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div>
                <div style="font-size: 12px; color: #999;">Current Area</div>
                <div style="font-size: 16px; font-weight: bold; color: #1a1a1a; margin-top: 2px;">{result['area_latest_m2']/10000:.2f} ha</div>
            </div>
            <div>
                <div style="font-size: 12px; color: #999;">Growth Rate</div>
                <div style="font-size: 16px; font-weight: bold; color: #1a1a1a; margin-top: 2px;">{result['growth_rate_m2yr']:,.0f} m²/yr</div>
            </div>
            <div>
                <div style="font-size: 12px; color: #999;">5-Year Forecast</div>
                <div style="font-size: 16px; font-weight: bold; color: #1a1a1a; margin-top: 2px;">{result['forecast_5yr_area']/10000:.2f} ha</div>
                <div style="font-size: 11px; color: #666; margin-top: 2px;">+{area_growth_pct:.1f}% growth</div>
            </div>
            <div>
                <div style="font-size: 12px; color: #999;">Confiscations</div>
                <div style="font-size: 16px; font-weight: bold; color: #1a1a1a; margin-top: 2px;">{result['n_conf_total']} total</div>
                <div style="font-size: 11px; color: #666; margin-top: 2px;">{result['n_conf_recent']} recent</div>
            </div>
        </div>
    </div>
    
    <!-- Pressure Indicators -->
    <div style="background: white; padding: 16px; border-radius: 6px; margin-bottom: 24px;">
        <h3 style="margin: 0 0 12px 0; font-size: 14px; text-transform: uppercase; color: #666; font-weight: 600;">Risk Factors</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px;">
            <div>
                <div style="font-size: 12px; color: #999;">Leaked Ratio</div>
                <div style="font-size: 14px; font-weight: bold; color: #1a1a1a;">{result['leaked_ratio']:.1%}</div>
            </div>
            <div>
                <div style="font-size: 12px; color: #999;">Zone-C Coverage</div>
                <div style="font-size: 14px; font-weight: bold; color: #1a1a1a;">{result['zone_c_coverage']:.1%}</div>
            </div>
            <div>
                <div style="font-size: 12px; color: #999;">Transactions</div>
                <div style="font-size: 14px; font-weight: bold; color: #1a1a1a;">{result['n_transactions']}</div>
            </div>
        </div>
    </div>
    
    <!-- Report Footer -->
    <div style="font-size: 11px; color: #999; text-align: center; padding-top: 12px; border-top: 1px solid #e0e0e0;">
        Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} • Settlement Expansion Risk Analysis
    </div>
    
</div>
"""
    return html


# ── Save HTML Report ─────────────────────────────────────────

def _save_html_report(result: dict, html_content: str) -> str:
    """
    Save HTML report to outputs folder.
    Returns the file path where it was saved.
    """
    filename = f"{result['name'].replace(' ', '_').lower()}_report.html"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Create minimal HTML wrapper
    full_html = f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{result['name']} - Settlement Expansion Report</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #f0f0f0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Predict expansion risk for one settlement"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id",   type=int,   help="settlement_id")
    group.add_argument("--name", type=str,   help="Settlement name")
    args = parser.parse_args()

    predictor = SingleSettlementPredictor()

    result = predictor.predict(
        settlement_id = args.id,
        name          = args.name,
    )
    
    # Print console report
    _print_report(result)

    print("  Top 10 model feature importances:")
    print(predictor.top_features(10).to_string(index=False))
    print()
    
    # Generate and save HTML report
    html_content = _generate_html_report(result)
    html_path = _save_html_report(result, html_content)
    
    print(f"\n✓ HTML Report saved to: {html_path}")
    print(f"  You can embed this report in other pages by including the HTML section.\n")


if __name__ == "__main__":
    main()