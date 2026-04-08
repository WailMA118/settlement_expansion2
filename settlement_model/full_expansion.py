# ═══════════════════════════════════════════════════════════════
# full_expansion.py — Full pipeline: all outputs for all settlements
# Run this after train_model.py has saved the model.
# ═══════════════════════════════════════════════════════════════

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap

from config import OUTPUT_DIR, SEVERITY_COLOR

# ── Output writers ────────────────────────────────────────────

class ResultTableWriter:

    def write(self, df: pd.DataFrame) -> str:
        cols = [
            "settlement_id", "name", "type", "established_year",
            "composite_risk", "xgb_risk", "ts_risk", "spatial_risk",
            "severity",
            "growth_rate_m2yr", "area_latest_m2", "forecast_5yr_area",
            "n_conf_total", "n_conf_recent",
            "leaked_ratio", "zone_c_coverage",
            "road_length_m", "n_transactions", "price_slope",
        ]
        out  = df[[c for c in cols if c in df.columns]].copy()
        for col in ["composite_risk", "xgb_risk", "ts_risk", "spatial_risk"]:
            if col in out: out[col] = out[col].round(4)

        path = os.path.join(OUTPUT_DIR, "settlement_expansion_risk.csv")
        out.to_csv(path, index=False)
        print(f"    CSV     → {path}")
        return path


class GeoJSONWriter:

    def write(self, df: pd.DataFrame,
              settlements_gdf: gpd.GeoDataFrame) -> str:
        geo = settlements_gdf[["settlement_id", "geometry"]].merge(
            df[["settlement_id", "name", "composite_risk", "severity",
                "xgb_risk", "ts_risk", "growth_rate_m2yr",
                "area_latest_m2", "n_conf_total"]],
            on="settlement_id", how="right"
        )
        gdf_out = gpd.GeoDataFrame(geo, geometry="geometry", crs="EPSG:4326")
        path    = os.path.join(OUTPUT_DIR, "settlement_expansion_risk.geojson")
        gdf_out.to_file(path, driver="GeoJSON")
        print(f"    GeoJSON → {path}")
        return path


class FoliumMapWriter:

    def write(self, df: pd.DataFrame,
              settlements_gdf: gpd.GeoDataFrame,
              confiscation: gpd.GeoDataFrame,
              parcels: gpd.GeoDataFrame) -> str:

        center = [31.9, 35.2]
        m = folium.Map(location=center, zoom_start=11,
                       tiles="CartoDB dark_matter")

        self._add_confiscation_heatmap(m, confiscation)
        self._add_leaked_parcels(m, parcels)
        self._add_settlements(m, df, settlements_gdf)
        self._add_legend(m)

        folium.LayerControl().add_to(m)

        path = os.path.join(OUTPUT_DIR, "settlement_expansion_map.html")
        m.save(path)
        print(f"    Map     → {path}")
        return path

    # ── Layer builders ────────────────────────────────────────

    def _add_confiscation_heatmap(self, m, confiscation):
        conf_clean = confiscation[confiscation["geometry"].notna()].copy()
        conf_pts = [[g.centroid.y, g.centroid.x]
                    for g in conf_clean["geometry"] if g and not g.is_empty]
        if conf_pts:
            HeatMap(conf_pts, name="Confiscation orders heatmap",
                    radius=15, blur=20, max_zoom=14).add_to(m)

    def _add_leaked_parcels(self, m, parcels):
        leaked_layer = folium.FeatureGroup(name="Leaked parcels")
        leaked = parcels[parcels["leakage_label"].isin(["leaked", "suspected"])].copy()

        for _, p_row in leaked.iterrows():
            geom = p_row["geometry"]
            if geom is None or geom.is_empty:
                continue
            c = geom.centroid
            color = "#e74c3c" if p_row["leakage_label"] == "leaked" else "#e67e22"
            folium.CircleMarker(
                location=[c.y, c.x],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"Parcel {p_row['parcel_id']} - {p_row['leakage_label']}"
            ).add_to(leaked_layer)
        leaked_layer.add_to(m)

    def _add_settlements(self, m, df, settlements_gdf):
        settle_layer = folium.FeatureGroup(name="Settlements (risk)")
        
        # Merge settlements geometry with risk scores using settlements_gdf as base
        merged = settlements_gdf.merge(
            df[["settlement_id", "composite_risk", "severity", "xgb_risk",
                "ts_risk", "spatial_risk", "growth_rate_m2yr",
                "n_conf_total", "leaked_ratio",
                "zone_c_coverage", "forecast_5yr_area", "name"]],
            on="settlement_id", how="right"
        )

        for _, row in merged.iterrows():
            geom = row["geometry"]
            if geom is None or geom.is_empty:
                continue
            
            color = SEVERITY_COLOR.get(row["severity"], "#95a5a6")
            risk_pct = f"{row['composite_risk'] * 100:.1f}%"
            area_5yr = (f"{row['forecast_5yr_area'] / 10_000:.1f} ha"
                        if row.get("forecast_5yr_area", 0) > 0 else "N/A")
            
            popup_html = f"""
            <div style='font-family:sans-serif;min-width:200px'>
              <b style='font-size:14px'>{row.get('name', 'Unknown')}</b><br>
              <span style='color:{color};font-weight:bold'>
                Warning {row['severity'].upper()} RISK - {risk_pct}</span><br><hr>
              <table style='font-size:12px'>
                <tr><td>XGBoost risk</td><td>{row['xgb_risk']:.3f}</td></tr>
                <tr><td>Time-series risk</td><td>{row['ts_risk']:.3f}</td></tr>
                <tr><td>Spatial pressure</td><td>{row['spatial_risk']:.3f}</td></tr>
                <tr><td>Growth rate</td>
                    <td>{row['growth_rate_m2yr']:,.0f} m2/yr</td></tr>
                <tr><td>Confiscation orders</td>
                    <td>{int(row['n_conf_total'])}</td></tr>
                <tr><td>Leaked parcel ratio</td>
                    <td>{row['leaked_ratio']:.1%}</td></tr>
                <tr><td>Zone-C coverage</td>
                    <td>{row['zone_c_coverage']:.1%}</td></tr>
                <tr><td>Forecast area (5yr)</td><td>{area_5yr}</td></tr>
              </table>
            </div>
            """
            
            # Draw polygon
            if geom.geom_type == "Polygon":
                coords = [[c[1], c[0]] for c in geom.exterior.coords]
            elif geom.geom_type == "MultiPolygon":
                coords = [[c[1], c[0]] for c in
                          max(geom.geoms, key=lambda g: g.area).exterior.coords]
            else:
                continue

            folium.Polygon(
                locations=coords,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.45,
                weight=2,
                tooltip=row.get("name", ""),
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(settle_layer)

            # Centroid label
            centroid = geom.centroid
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(
                    html=f'<div style="color:{color};font-size:9px;'
                         f'font-weight:bold;white-space:nowrap">'
                         f'{row.get("name","")}</div>',
                    icon_size=(120, 18)
                )
            ).add_to(settle_layer)

        settle_layer.add_to(m)

    def _add_legend(self, m):
        legend_html = """
        <div style='position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:rgba(0,0,0,0.8);color:white;padding:12px;
                    border-radius:8px;font-family:sans-serif;font-size:12px'>
          <b>Settlement Expansion Risk</b><br>
          <span style='color:#c0392b'>●</span> Critical (&gt;75%)<br>
          <span style='color:#e67e22'>●</span> High (&gt;55%)<br>
          <span style='color:#f1c40f'>●</span> Medium (&gt;35%)<br>
          <span style='color:#2ecc71'>●</span> Low<br><hr>
          <span style='color:#e74c3c'>●</span> Leaked parcel<br>
          <span style='color:#e67e22'>●</span> Suspected parcel
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))


# ── Console table ─────────────────────────────────────────────

def print_ranking_table(df: pd.DataFrame):
    print("\n" + "═" * 80)
    print("  SETTLEMENT EXPANSION RISK RANKING")
    print("═" * 80)
    print(f"{'#':<4} {'Name':<28} {'Type':<12} {'Risk':<8} "
          f"{'Severity':<10} {'Growth m²/yr':<14} {'Confisc.'}")
    print("─" * 80)
    for i, row in df.head(20).iterrows():
        print(f"{i+1:<4} {str(row.get('name','?')):<28} "
              f"{str(row.get('type','?')):<12} "
              f"{row['composite_risk']:.3f}   "
              f"{row.get('severity','?'):<10} "
              f"{row.get('growth_rate_m2yr',0):>12,.0f}   "
              f"{int(row.get('n_conf_total',0))}")
    print("═" * 80)


# ═══════════════════════════════════════════════════════════════
# Full pipeline entry point
# ═══════════════════════════════════════════════════════════════

def run_full_pipeline():
    from data_loader import get_engine, load_all
    from feature_engineering import FeatureMatrix
    from train_model import SettlementExpansionModel
    from timeseries import TimeSeriesForecaster
    from score_fusion import RiskFusion, AlertWriter

    print("\n" + "═" * 60)
    print("  SETTLEMENT EXPANSION — FULL PIPELINE")
    print("  Ramallah data | Palestine-wide model")
    print("═" * 60 + "\n")

    # 1. Data
    engine = get_engine()
    data   = load_all(engine)

    # 2. Features
    df = FeatureMatrix().build(data)

    if df.empty:
        print("ERROR: No settlements found.")
        return

    # 3a. XGBoost (load saved model)
    model = SettlementExpansionModel().load()
    xgb_proba = model.predict(df)

    # 3b. Time series
    area_map = dict(zip(df["settlement_id"], df["area_latest_m2"]))
    ts_df    = TimeSeriesForecaster().forecast_all(
        df["settlement_id"].tolist(),
        data["expansion_history"],
        area_map
    )

    # 4. Score fusion
    print("[score_fusion] Fusing risk scores …")
    df_final = RiskFusion().fuse(df, xgb_proba, ts_df)
    print_ranking_table(df_final)

    # 5. Outputs
    print("\n[outputs] Writing files …")
    ResultTableWriter().write(df_final)
    GeoJSONWriter().write(df_final, data["settlements"])
    FoliumMapWriter().write(
        df_final, data["settlements"],
        data["confiscation"], data["parcels"]
    )

    # 6. DB alerts
    n_alerts = AlertWriter().write(df_final, engine)

    # Summary
    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Settlements analysed : {len(df_final)}")
    print(f"  Critical             : {(df_final['severity']=='critical').sum()}")
    print(f"  High                 : {(df_final['severity']=='high').sum()}")
    print(f"  Alerts written       : {n_alerts}")
    print(f"  Outputs dir          : {OUTPUT_DIR}")
    print("═" * 60 + "\n")

    return df_final


if __name__ == "__main__":
    run_full_pipeline()