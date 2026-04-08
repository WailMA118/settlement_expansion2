# ═══════════════════════════════════════════════════════════════
# data_loader.py — MySQL extraction → GeoDataFrames
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import geopandas as gpd
from shapely import wkt
from sqlalchemy import create_engine, text

from config import DB_URL, WGS84


# ── Engine factory ────────────────────────────────────────────

def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)


# ── Generic loader ────────────────────────────────────────────

def load_geodataframe(engine, table: str,
                      geom_col: str = "geom",
                      extra_cols: str = "*") -> gpd.GeoDataFrame:
    """
    Load any spatial table from MySQL.
    Uses ST_AsText() to convert binary geometry → WKT → Shapely.
    Returns GeoDataFrame in EPSG:4326.
    """
    query = f"""
        SELECT {extra_cols},
               ST_AsText(`{geom_col}`) AS geom_wkt
        FROM   `{table}`
    """
    df = pd.read_sql(text(query), engine)
    df["geometry"] = df["geom_wkt"].apply(
        lambda x: wkt.loads(x) if x else None
    )
    df.drop(columns=["geom_wkt"], inplace=True)
    return gpd.GeoDataFrame(df, geometry="geometry", crs=WGS84)


# ── Specific table loaders ────────────────────────────────────

def load_settlements(engine) -> gpd.GeoDataFrame:
    return load_geodataframe(
        engine, "settlements",
        extra_cols="settlement_id, name, type, established_year"
    )


def load_expansion_history(engine) -> gpd.GeoDataFrame:
    return load_geodataframe(
        engine, "settlement_expansion_history",
        extra_cols="expansion_id, settlement_id, recorded_year"
    )


def load_roads(engine) -> gpd.GeoDataFrame:
    return load_geodataframe(
        engine, "settlement_roads",
        extra_cols="road_id, name, width_meters"
    )


def load_confiscation(engine) -> gpd.GeoDataFrame:
    """Confiscation orders joined to parcel geometry."""
    query = text("""
        SELECT co.order_id,
               co.parcel_id,
               co.order_type,
               co.issue_date,
               co.issued_by,
               ST_AsText(lp.geom) AS geom_wkt
        FROM   confiscation_orders co
        JOIN   land_parcels lp ON lp.parcel_id = co.parcel_id
    """)
    df = pd.read_sql(query, engine)
    df["geometry"] = df["geom_wkt"].apply(
        lambda x: wkt.loads(x) if x else None
    )
    df.drop(columns=["geom_wkt"], inplace=True)
    return gpd.GeoDataFrame(df, geometry="geometry", crs=WGS84)


def load_parcels(engine) -> gpd.GeoDataFrame:
    return load_geodataframe(
        engine, "land_parcels",
        extra_cols=(
            "parcel_id, locality_id, oslo_id, "
            "leakage_label, registration_status, area_m2"
        )
    )


def load_oslo(engine) -> gpd.GeoDataFrame:
    return load_geodataframe(
        engine, "oslo_zones",
        extra_cols="zone_id, class"
    )


def load_leakage_cases(engine) -> pd.DataFrame:
    query = text("""
        SELECT case_id, parcel_id, case_status, suspicion_score
        FROM   leakage_cases
    """)
    return pd.read_sql(query, engine)


def load_transactions(engine) -> pd.DataFrame:
    query = text("""
        SELECT transaction_id, parcel_id,
               transaction_date, price, transaction_type
        FROM   land_transactions
        WHERE  transaction_type IN ('sale', 'court_transfer', 'confiscation')
    """)
    return pd.read_sql(query, engine)


def load_owner_risk(engine) -> pd.DataFrame:
    """Owner risk scores joined through parcel_ownership."""
    query = text("""
        SELECT po.parcel_id,
               orp.risk_score,
               orp.risk_type,
               o.identity_type,
               o.residence_country
        FROM   parcel_ownership    po
        JOIN   owners              o   ON o.owner_id   = po.owner_id
        JOIN   owner_risk_profiles orp ON orp.owner_id = po.owner_id
    """)
    return pd.read_sql(query, engine)


# ── Master loader ─────────────────────────────────────────────

def load_all(engine) -> dict:
    """
    Load every table required by the pipeline.
    Returns a dict keyed by table name.
    """
    print("[data_loader] Extracting data from MySQL …")

    data = {
        "settlements"      : load_settlements(engine),
        "expansion_history": load_expansion_history(engine),
        "roads"            : load_roads(engine),
        "confiscation"     : load_confiscation(engine),
        "parcels"          : load_parcels(engine),
        "oslo"             : load_oslo(engine),
        "leakage_cases"    : load_leakage_cases(engine),
        "transactions"     : load_transactions(engine),
        "owner_risk"       : load_owner_risk(engine),
    }

    for name, obj in data.items():
        print(f"    {name:<22}: {len(obj)} rows")

    return data