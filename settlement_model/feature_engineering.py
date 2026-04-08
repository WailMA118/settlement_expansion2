# ═══════════════════════════════════════════════════════════════
# feature_engineering.py — Spatial + temporal feature builders
# ═══════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
from shapely.ops import transform, unary_union
from pyproj import Transformer

from config import (
    WGS84, PALESTINE_CRS,
    BUFFER_CONFISCATION, BUFFER_ROADS,
    BUFFER_PARCELS, BUFFER_OSLO,
)


# ── CRS transformers (module-level singletons) ────────────────
_to_metric = Transformer.from_crs(WGS84, PALESTINE_CRS, always_xy=True)
_to_wgs84  = Transformer.from_crs(PALESTINE_CRS, WGS84,  always_xy=True)

CURRENT_YEAR = datetime.now().year


# ── Geometry helpers ──────────────────────────────────────────

def to_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject GeoDataFrame to Palestinian metric CRS (EPSG:2039)."""
    return gdf.to_crs(PALESTINE_CRS)


def buffer_metric(geom, radius_m: float):
    """
    Buffer a WGS-84 geometry using metric distance.
    Converts to EPSG:2039 → buffer → back to WGS-84.
    """
    g_m    = transform(_to_metric.transform, geom)
    buf_m  = g_m.buffer(radius_m)
    return transform(_to_wgs84.transform, buf_m)


def area_in_metric(geom) -> float:
    """Return area (m²) of a WGS-84 geometry after projecting to metric CRS."""
    if geom is None or geom.is_empty:
        return 0.0
    g_m = transform(_to_metric.transform, geom)
    return g_m.area


# ═══════════════════════════════════════════════════════════════
# 1. Temporal expansion features
# ═══════════════════════════════════════════════════════════════

class ExpansionFeatureBuilder:
    """
    Builds time-based growth features from settlement_expansion_history.
    One row per settlement.
    """

    def build(self, settlements: gpd.GeoDataFrame,
              expansion_history: gpd.GeoDataFrame) -> pd.DataFrame:

        rows = []
        for _, row in settlements.iterrows():
            rows.append(self._settlement_row(row, expansion_history))
        return pd.DataFrame(rows)

    def _settlement_row(self, s_row, expansion_history) -> dict:
        sid  = s_row["settlement_id"]
        hist = (expansion_history[expansion_history["settlement_id"] == sid]
                .copy()
                .pipe(lambda df: df[df["geometry"].notna()])
                .sort_values("recorded_year"))

        if len(hist) >= 2:
            areas       = hist.to_crs(PALESTINE_CRS)["geometry"].area.values
            years       = hist["recorded_year"].values
            area_first  = float(areas[0])
            area_latest = float(areas[-1])
            yr_span     = max(int(years[-1]) - int(years[0]), 1)
            growth_rate = (area_latest - area_first) / yr_span
            n_snaps     = len(hist)
            yr_record   = yr_span

            # Growth in last decade relative to total area
            last_decade = hist[hist["recorded_year"] >= CURRENT_YEAR - 10]
            if len(last_decade) >= 2:
                ld_areas = last_decade.to_crs(PALESTINE_CRS)["geometry"].area.values
                dec_frac = (float(ld_areas[-1]) - float(ld_areas[0])) / max(area_latest, 1)
            else:
                dec_frac = 0.0

        elif len(hist) == 1:
            area_latest = float(hist.to_crs(PALESTINE_CRS)["geometry"].area.values[0])
            area_first  = area_latest
            growth_rate = 0.0
            n_snaps     = 1
            yr_record   = 0
            dec_frac    = 0.0

        else:
            # Fallback: use settlement's own geometry
            area_latest = area_in_metric(s_row["geometry"])
            area_first  = area_latest
            growth_rate = 0.0
            n_snaps     = 0
            yr_record   = 0
            dec_frac    = 0.0

        return {
            "settlement_id"        : sid,
            "area_latest_m2"       : area_latest,
            "area_first_m2"        : area_first,
            "growth_rate_m2yr"     : growth_rate,
            "n_expansion_snapshots": n_snaps,
            "years_of_record"      : yr_record,
            "decade_growth_frac"   : dec_frac,
        }


# ═══════════════════════════════════════════════════════════════
# 2. Spatial pressure features
# ═══════════════════════════════════════════════════════════════

class SpatialPressureBuilder:
    """
    Computes buffer-based spatial pressure features per settlement.
    Each metric captures a different dimension of settlement expansion risk.
    """

    def build(self,
              settlements: gpd.GeoDataFrame,
              confiscation: gpd.GeoDataFrame,
              roads: gpd.GeoDataFrame,
              parcels: gpd.GeoDataFrame,
              oslo: gpd.GeoDataFrame,
              transactions: pd.DataFrame,
              leakage_cases: pd.DataFrame,
              owner_risk: pd.DataFrame) -> pd.DataFrame:

        # ── Pre-process once ──────────────────────────────────
        parcels_ext = self._enrich_parcels(parcels, leakage_cases, owner_risk)
        conf        = self._prepare_confiscation(confiscation)
        trans       = self._prepare_transactions(transactions)
        zone_c      = oslo[oslo["class"].str.upper().str.contains("C", na=False)]

        rows = []
        for _, s_row in settlements.iterrows():
            rows.append(
                self._settlement_row(
                    s_row, conf, roads, parcels_ext,
                    zone_c, trans
                )
            )
        return pd.DataFrame(rows)

    # ── Pre-processors ────────────────────────────────────────

    def _enrich_parcels(self, parcels, leakage_cases, owner_risk) -> gpd.GeoDataFrame:
        """Attach max suspicion score and avg owner risk to each parcel."""
        lc = (leakage_cases
              .groupby("parcel_id")["suspicion_score"]
              .max()
              .reset_index()
              .rename(columns={"suspicion_score": "max_suspicion"}))
        or_agg = (owner_risk
                  .groupby("parcel_id")["risk_score"]
                  .mean()
                  .reset_index()
                  .rename(columns={"risk_score": "avg_owner_risk"}))

        p = parcels.merge(lc,     on="parcel_id", how="left")
        p = p.merge(or_agg,       on="parcel_id", how="left")
        p["max_suspicion"]  = p["max_suspicion"].fillna(0)
        p["avg_owner_risk"] = p["avg_owner_risk"].fillna(0)
        return p

    def _prepare_confiscation(self, confiscation) -> gpd.GeoDataFrame:
        conf = confiscation.copy()
        conf["issue_date"] = pd.to_datetime(conf["issue_date"], errors="coerce")
        conf["is_recent"]  = (conf["issue_date"].dt.year >= CURRENT_YEAR - 5).astype(int)
        conf["conf_weight"]= conf["is_recent"].map({1: 2, 0: 1})
        return conf

    def _prepare_transactions(self, transactions) -> pd.DataFrame:
        trans = transactions.copy()
        trans["transaction_date"] = pd.to_datetime(
            trans["transaction_date"], errors="coerce")
        trans["year"] = trans["transaction_date"].dt.year
        return trans

    # ── Per-settlement row ────────────────────────────────────

    def _settlement_row(self, s_row, conf, roads,
                        parcels_ext, zone_c, trans) -> dict:
        sid  = s_row["settlement_id"]
        geom = s_row["geometry"]

        base = {"settlement_id": sid}
        if geom is None or geom.is_empty:
            return base

        # 1. Confiscation pressure (3 km)
        conf_feats = self._confiscation_features(geom, conf)

        # 2. Road expansion (2 km)
        road_feats = self._road_features(geom, roads)

        # 3. Parcel pressure (2 km)
        parc_feats = self._parcel_features(geom, parcels_ext)

        # 4. Zone C coverage (1 km)
        zone_c_cov = self._zone_c_feature(geom, zone_c)

        # 5. Transaction activity (on parcels in 2 km buffer)
        nearby_pids = parc_feats.pop("_nearby_pids")
        trans_feats = self._transaction_features(nearby_pids, trans)

        return {**base,
                **conf_feats, **road_feats, **parc_feats,
                "zone_c_coverage": zone_c_cov,
                **trans_feats}

    # ── Feature sub-builders ──────────────────────────────────

    def _confiscation_features(self, geom, conf: gpd.GeoDataFrame) -> dict:
        buf  = buffer_metric(geom, BUFFER_CONFISCATION)
        sub  = conf[conf["geometry"].intersects(buf)]

        type_counts   = sub["order_type"].value_counts().to_dict()
        return {
            "n_conf_total"       : len(sub),
            "n_conf_recent"      : int(sub["is_recent"].sum()),
            "conf_weighted_score": float(sub["conf_weight"].sum()),
            "n_state_land"       : type_counts.get("State Land Declaration", 0),
            "n_military_order"   : type_counts.get("Military Order", 0),
            "n_road_order"       : type_counts.get("Road Construction Order", 0),
            "n_absentee_law"     : type_counts.get("Absentee Property Law", 0),
        }

    def _road_features(self, geom, roads: gpd.GeoDataFrame) -> dict:
        buf  = buffer_metric(geom, BUFFER_ROADS)
        sub  = roads[roads["geometry"].intersects(buf)].to_crs(PALESTINE_CRS)

        length = float(sub["geometry"].length.sum())
        width  = float(sub["width_meters"].mean()) if len(sub) > 0 else 0.0
        score  = float(
            (sub["geometry"].length * sub["width_meters"].fillna(5)).sum()
        )
        return {
            "road_length_m"      : length,
            "road_avg_width_m"   : width,
            "road_capacity_score": score,
        }

    def _parcel_features(self, geom, parcels_ext: gpd.GeoDataFrame) -> dict:
        buf  = buffer_metric(geom, BUFFER_PARCELS)
        sub  = parcels_ext[parcels_ext["geometry"].intersects(buf)]
        n    = len(sub)

        n_leaked    = int((sub["leakage_label"] == "leaked").sum())
        n_suspected = int((sub["leakage_label"] == "suspected").sum())

        return {
            "n_parcels_nearby"  : n,
            "leaked_ratio"      : (n_leaked + n_suspected) / max(n, 1),
            "avg_suspicion_score": float(sub["max_suspicion"].mean()) if n > 0 else 0.0,
            "avg_owner_risk"    : float(sub["avg_owner_risk"].mean()) if n > 0 else 0.0,
            "n_israeli_register": int((sub["registration_status"] == "Israeli_Register").sum()),
            "n_in_settlement"   : int((sub["registration_status"] == "In_Settlement").sum()),
            "_nearby_pids"      : set(sub["parcel_id"].tolist()),   # internal, popped later
        }

    def _zone_c_feature(self, geom, zone_c: gpd.GeoDataFrame) -> float:
        if len(zone_c) == 0:
            return 0.0
        buf          = buffer_metric(geom, BUFFER_OSLO)
        zone_c_union = unary_union(zone_c["geometry"])
        intersect    = buf.intersection(zone_c_union)
        buf_area     = area_in_metric(buf)
        int_area     = area_in_metric(intersect)
        return int_area / max(buf_area, 1)

    def _transaction_features(self, nearby_pids: set,
                               trans: pd.DataFrame) -> dict:
        sub = trans[trans["parcel_id"].isin(nearby_pids)]
        n   = len(sub)

        slope = 0.0
        if n >= 3:
            yt = sub.dropna(subset=["year", "price"])
            if len(yt) >= 2:
                slope = float(np.polyfit(yt["year"], yt["price"], 1)[0])

        return {
            "n_transactions" : n,
            "price_slope"    : slope,
            "avg_price"      : float(sub["price"].mean()) if n > 0 else 0.0,
            "n_recent_trans" : int(sub[sub["year"] >= CURRENT_YEAR - 5].shape[0]),
        }


# ═══════════════════════════════════════════════════════════════
# 3. Settlement metadata features
# ═══════════════════════════════════════════════════════════════

class MetaFeatureBuilder:
    """Simple settlement-level metadata."""

    def build(self, settlements: gpd.GeoDataFrame) -> pd.DataFrame:
        rows = []
        for _, row in settlements.iterrows():
            age = (CURRENT_YEAR - int(row["established_year"])
                   if row["established_year"] else 30)
            rows.append({
                "settlement_id": row["settlement_id"],
                "settlement_age": age,
                "is_outpost"   : 1 if str(row.get("type", "")).lower() == "outpost" else 0,
            })
        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 4. Master feature assembler
# ═══════════════════════════════════════════════════════════════

class FeatureMatrix:
    """
    Assembles the full feature matrix by merging all builders.
    Also computes derived composite features.
    """

    def __init__(self):
        self.expansion_builder = ExpansionFeatureBuilder()
        self.spatial_builder   = SpatialPressureBuilder()
        self.meta_builder      = MetaFeatureBuilder()

    def build(self, data: dict) -> pd.DataFrame:
        print("[feature_engineering] Building feature matrix …")

        exp_feat   = self.expansion_builder.build(
            data["settlements"], data["expansion_history"])
        spatial_pr = self.spatial_builder.build(
            data["settlements"], data["confiscation"], data["roads"],
            data["parcels"],     data["oslo"],         data["transactions"],
            data["leakage_cases"], data["owner_risk"])
        meta       = self.meta_builder.build(data["settlements"])

        base = data["settlements"][
            ["settlement_id", "name", "type", "established_year", "geometry"]
        ].copy()

        df = (base
              .merge(meta,       on="settlement_id", how="left")
              .merge(exp_feat,   on="settlement_id", how="left")
              .merge(spatial_pr, on="settlement_id", how="left"))

        df = self._add_derived_features(df)
        df = self._fill_nulls(df)

        print(f"    Feature matrix: {df.shape[0]} rows × {df.shape[1]} cols")
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["expansion_momentum"] = (
            df["growth_rate_m2yr"].fillna(0) *
            df["decade_growth_frac"].fillna(0) /
            df["settlement_age"].clip(lower=1)
        )
        df["legal_pressure_index"] = (
            df["conf_weighted_score"].fillna(0) * 0.4 +
            df["n_state_land"].fillna(0)         * 0.3 +
            df["n_military_order"].fillna(0)      * 0.3
        )
        df["leakage_pressure"] = (
            df["leaked_ratio"].fillna(0)        * 0.5 +
            df["avg_suspicion_score"].fillna(0) / 100 * 0.3 +
            df["avg_owner_risk"].fillna(0)      / 100 * 0.2
        )
        df["road_expansion_index"] = (
            np.log1p(df["road_length_m"].fillna(0)) *
            df["road_avg_width_m"].fillna(5) / 10
        )
        return df

    def _fill_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df