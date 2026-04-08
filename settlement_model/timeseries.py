# ═══════════════════════════════════════════════════════════════
# timeseries.py — Area growth forecasting (Prophet / ARIMA fallback)
# ═══════════════════════════════════════════════════════════════

import json
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime

from config import PALESTINE_CRS, FORECAST_YEARS

CURRENT_YEAR = datetime.now().year

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except ImportError:
    _HAS_PROPHET = False
    try:
        from statsmodels.tsa.arima.model import ARIMA
        _HAS_ARIMA = True
    except ImportError:
        _HAS_ARIMA = False


class TimeSeriesForecaster:
    """
    Fits an area-growth model per settlement.

    Priority order:
      1. Prophet  (best — handles short series + uncertainty intervals)
      2. ARIMA    (fallback if Prophet not installed)
      3. Linear extrapolation (last resort)

    Input:  expansion_history rows for one settlement_id
    Output: {year: predicted_area_m2, …}
    """

    def forecast_one(self, settlement_id: int,
                     expansion_history: gpd.GeoDataFrame,
                     forecast_years: int = FORECAST_YEARS) -> dict:
        """
        Returns a dict {year (int): area_m2 (float)} for future years.
        Returns {} if insufficient data.
        """
        hist = (expansion_history[
                    expansion_history["settlement_id"] == settlement_id]
                .copy()
                .pipe(lambda df: df[df["geometry"].notna()])
                .sort_values("recorded_year"))

        if len(hist) < 3:
            return {}

        areas = hist.to_crs(PALESTINE_CRS)["geometry"].area.values.astype(float)
        years = hist["recorded_year"].values.astype(int)

        if _HAS_PROPHET:
            return self._prophet_forecast(years, areas, forecast_years)
        elif _HAS_ARIMA:
            return self._arima_forecast(years, areas, forecast_years)
        else:
            return self._linear_forecast(years, areas, forecast_years)

    # ── Backends ──────────────────────────────────────────────

    def _prophet_forecast(self, years, areas, n) -> dict:
        ts_df = pd.DataFrame({
            "ds": pd.to_datetime([f"{y}-06-01" for y in years]),
            "y" : areas,
        })
        m = Prophet(
            yearly_seasonality    = False,
            weekly_seasonality    = False,
            daily_seasonality     = False,
            changepoint_prior_scale = 0.3,
        )
        m.fit(ts_df)
        future = m.make_future_dataframe(periods=n, freq="YE")
        fc = m.predict(future)
        return {
            int(row["ds"].year): max(float(row["yhat"]), 0.0)
            for _, row in fc.iterrows()
            if row["ds"].year >= CURRENT_YEAR
        }

    def _arima_forecast(self, years, areas, n) -> dict:
        try:
            fit = ARIMA(areas, order=(1, 1, 0)).fit()
            fc  = fit.forecast(steps=n)
            return {
                CURRENT_YEAR + i: max(float(v), 0.0)
                for i, v in enumerate(fc)
            }
        except Exception:
            return self._linear_forecast(years, areas, n)

    def _linear_forecast(self, years, areas, n) -> dict:
        slope     = float(np.polyfit(years, areas, 1)[0])
        last_area = float(areas[-1])
        return {
            CURRENT_YEAR + i: max(last_area + slope * (i + 1), 0.0)
            for i in range(n)
        }

    # ── Batch forecasting ─────────────────────────────────────

    def forecast_all(self, settlement_ids,
                     expansion_history: gpd.GeoDataFrame,
                     area_latest_map: dict) -> pd.DataFrame:
        """
        Run forecast for all settlements.
        Returns DataFrame with growth metrics per settlement.
        """
        print("[timeseries] Running forecasts …")
        rows = []

        for sid in settlement_ids:
            fc = self.forecast_one(sid, expansion_history)

            if fc:
                last_area         = area_latest_map.get(sid, 0.0)
                max_yr            = max(fc.keys())
                max_area          = fc[max_yr]
                growth_rate_fc    = (max_area - last_area) / max(last_area, 1)
                norm_growth       = min(growth_rate_fc, 3.0) / 3.0
                fc_5yr            = fc.get(CURRENT_YEAR + 5, 0.0)
            else:
                growth_rate_fc = 0.0
                norm_growth    = 0.0
                fc_5yr         = 0.0
                fc             = {}

            rows.append({
                "settlement_id"      : sid,
                "forecast_growth_rate": growth_rate_fc,
                "forecast_norm_growth": norm_growth,
                "forecast_5yr_area"  : fc_5yr,
                "forecast_series"    : json.dumps(
                    {str(k): round(v, 1) for k, v in fc.items()}
                ),
            })

        return pd.DataFrame(rows)