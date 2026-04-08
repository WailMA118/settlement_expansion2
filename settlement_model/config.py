# ═══════════════════════════════════════════════════════════════
# config.py — Central configuration for Settlement Expansion Model
# ═══════════════════════════════════════════════════════════════

import os

# ── Database ──────────────────────────────────────────────────
DB_URL = "mysql+mysqlconnector://root:wail@localhost/palestine_land_system_v4"

# ── Coordinate Reference Systems ──────────────────────────────
WGS84        = "EPSG:4326"   # stored in DB
PALESTINE_CRS = "EPSG:2039"  # metric CRS for distance calculations

# ── Spatial buffer radii (metres) ─────────────────────────────
BUFFER_CONFISCATION = 3_000
BUFFER_ROADS        = 2_000
BUFFER_PARCELS      = 2_000
BUFFER_OSLO         = 1_000

# ── Score fusion weights (must sum to 1.0) ────────────────────
W_XGBOOST = 0.50
W_GROWTH   = 0.30
W_SPATIAL  = 0.20

# ── Alert thresholds ──────────────────────────────────────────
THR_CRITICAL = 0.75
THR_HIGH     = 0.55
THR_MEDIUM   = 0.35

# ── Time series ───────────────────────────────────────────────
FORECAST_YEARS = 10

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODEL_DIR   = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

MODEL_PATH   = os.path.join(MODEL_DIR, "xgb_settlement.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler_settlement.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_cols.pkl")

# ── Feature columns (single source of truth) ─────────────────
FEATURE_COLS = [
    # Temporal expansion
    "area_latest_m2",
    "area_first_m2",
    "growth_rate_m2yr",
    "n_expansion_snapshots",
    "years_of_record",
    "decade_growth_frac",
    # Settlement metadata
    "settlement_age",
    "is_outpost",
    # Confiscation pressure
    "n_conf_total",
    "n_conf_recent",
    "conf_weighted_score",
    "n_state_land",
    "n_military_order",
    "n_road_order",
    "n_absentee_law",
    # Road expansion
    "road_length_m",
    "road_avg_width_m",
    "road_capacity_score",
    # Parcel pressure
    "n_parcels_nearby",
    "leaked_ratio",
    "avg_suspicion_score",
    "avg_owner_risk",
    "n_israeli_register",
    "n_in_settlement",
    # Oslo Zone C
    "zone_c_coverage",
    # Transactions
    "n_transactions",
    "price_slope",
    "avg_price",
    "n_recent_trans",
    # Derived composites
    "expansion_momentum",
    "legal_pressure_index",
    "leakage_pressure",
    "road_expansion_index",
]

# ── Severity colours (used in map + reports) ──────────────────
SEVERITY_COLOR = {
    "critical": "#c0392b",
    "high"    : "#e67e22",
    "medium"  : "#f1c40f",
    "low"     : "#2ecc71",
}