# f1_predictor.py

import fastf1
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

# Optional models (we'll guard with try/except)
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


TRACK_DATA = {
    "Bahrain Grand Prix":  {"Downforce": "medium", "Category": "permanent", "LengthKM": 5.4, "Corners": 15, "StopGo": 3},
    "Saudi Arabian Grand Prix": {"Downforce": "low", "Category": "street", "LengthKM": 6.1, "Corners": 27, "StopGo": 1},
    "Australian Grand Prix": {"Downforce": "medium", "Category": "semi-street", "LengthKM": 5.3, "Corners": 14, "StopGo": 2},
    "Azerbaijan Grand Prix": {"Downforce": "low", "Category": "street", "LengthKM": 6.0, "Corners": 20, "StopGo": 2},
    "Miami Grand Prix": {"Downforce": "medium", "Category": "street", "LengthKM": 5.4, "Corners": 19, "StopGo": 3},
    "Monaco Grand Prix": {"Downforce": "high", "Category": "street", "LengthKM": 3.3, "Corners": 19, "StopGo": 1},
    "Spanish Grand Prix": {"Downforce": "high", "Category": "permanent", "LengthKM": 4.7, "Corners": 14, "StopGo": 1},
    "Canadian Grand Prix": {"Downforce": "medium", "Category": "semi-street", "LengthKM": 4.4, "Corners": 14, "StopGo": 3},
    "Austrian Grand Prix": {"Downforce": "medium", "Category": "permanent", "LengthKM": 4.3, "Corners": 10, "StopGo": 2},
    "British Grand Prix": {"Downforce": "high", "Category": "permanent", "LengthKM": 5.9, "Corners": 18, "StopGo": 0},
    "Hungarian Grand Prix": {"Downforce": "high", "Category": "permanent", "LengthKM": 4.4, "Corners": 14, "StopGo": 1},
    "Belgian Grand Prix": {"Downforce": "low", "Category": "permanent", "LengthKM": 7.0, "Corners": 19, "StopGo": 0},
    "Dutch Grand Prix": {"Downforce": "high", "Category": "permanent", "LengthKM": 4.2, "Corners": 14, "StopGo": 1},
    "Italian Grand Prix": {"Downforce": "low", "Category": "permanent", "LengthKM": 5.8, "Corners": 11, "StopGo": 2},
    "Singapore Grand Prix": {"Downforce": "high", "Category": "street", "LengthKM": 5.0, "Corners": 19, "StopGo": 3},
    "Japanese Grand Prix": {"Downforce": "high", "Category": "permanent", "LengthKM": 5.8, "Corners": 18, "StopGo": 0},
    "United States Grand Prix": {"Downforce": "medium", "Category": "permanent", "LengthKM": 5.5, "Corners": 20, "StopGo": 2},
    "Mexico City Grand Prix": {"Downforce": "high-altitude", "Category": "permanent", "LengthKM": 4.3, "Corners": 17, "StopGo": 2},
    "Brazilian Grand Prix": {"Downforce": "high", "Category": "permanent", "LengthKM": 4.3, "Corners": 15, "StopGo": 2},
    "Las Vegas Grand Prix": {"Downforce": "low", "Category": "street", "LengthKM": 6.2, "Corners": 17, "StopGo": 2},
    "Qatar Grand Prix": {"Downforce": "medium", "Category": "permanent", "LengthKM": 5.4, "Corners": 16, "StopGo": 1},
    "Abu Dhabi Grand Prix": {"Downforce": "medium", "Category": "permanent", "LengthKM": 5.3, "Corners": 16, "StopGo": 2},
}

# ---------------------------
# 1. FastF1 setup + data loading
# ---------------------------

def enable_fastf1_cache(cache_dir: str = "./fastf1_cache") -> None:
    """
    Enable FastF1 on-disk cache to avoid re-downloading data.
    """
    fastf1.Cache.enable_cache(cache_dir)


def load_qualifying_results(year: int, round_number: int) -> pd.DataFrame:
    """
    Load qualifying session data for a given race.
    Returns a DataFrame with driver abbreviation, team, quali time, and quali position.
    """
    session = fastf1.get_session(year, round_number, "Q")
    session.load(laps=False, telemetry=False, weather=False)

    quali = session.results[[
        "Abbreviation", "DriverNumber", "TeamName",
        "Position", "Q1", "Q2", "Q3"
    ]].copy()

    quali.rename(columns={
        "Abbreviation": "Driver",
        "Position": "QualiPosition"
    }, inplace=True)

    # Convert Q1/Q2/Q3 times to a single representative lap time in seconds
    def convert_time(t):
        if pd.isna(t):
            return np.nan
        try:
            return t.total_seconds()
        except Exception:
            return np.nan

    quali["Q1s"] = quali["Q1"].apply(convert_time)
    quali["Q2s"] = quali["Q2"].apply(convert_time)
    quali["Q3s"] = quali["Q3"].apply(convert_time)

    # Representative lap time = best of Q1/Q2/Q3
    quali["BestQualiLap"] = quali[["Q1s", "Q2s", "Q3s"]].min(axis=1)

    quali = quali[["Driver", "TeamName", "DriverNumber", "QualiPosition", "BestQualiLap"]]

    return quali


def load_season_results(year: int) -> pd.DataFrame:
    """
    Load race results for a given season using FastF1.

    Returns a DataFrame with one row per driver per race, including:
    Year, RoundNumber, RaceName, Driver, TeamName, GridPosition, Position, Points, Quali data.
    """
    print(f"\n=== Loading season {year} ===")
    schedule = fastf1.get_event_schedule(year, include_testing=False)

    all_results = []

    # Only keep rounds that have a race ("R" session)
    race_events = schedule[schedule["EventFormat"].notna()]

    for _, event in race_events.iterrows():
        rnd = int(event["RoundNumber"])
        race_name = event["EventName"]

        print(f"  -> Loading {year} Round {rnd}: {race_name}")
        session = fastf1.get_session(year, rnd, "R")
        session.load(laps=False, telemetry=False, weather=False)

        res = session.results
        df_res = res[[
            "Abbreviation", "DriverNumber", "TeamName",
            "GridPosition", "Position", "Points"
        ]].copy()

        # Rename first so "Driver" exists for merging
        df_res.rename(columns={
            "Abbreviation": "Driver",
        }, inplace=True)

        # Add qualifying data
        try:
            quali_df = load_qualifying_results(year, rnd)
            df_res = df_res.merge(
                quali_df[["Driver", "BestQualiLap", "QualiPosition"]],
                on="Driver",
                how="left",
            )
        except Exception as e:
            print(f"  !! Qualifying data missing for {year} Round {rnd}: {e}")
            df_res["BestQualiLap"] = np.nan
            df_res["QualiPosition"] = np.nan

        df_res["Year"] = year
        df_res["RoundNumber"] = rnd
        df_res["RaceName"] = race_name

        all_results.append(df_res)

    season_df = pd.concat(all_results, ignore_index=True)
    print(f"Loaded {len(season_df)} driver-race rows for {year}.")
    return season_df


def build_dataset(start_year: int = 2022, end_year: int = 2024) -> pd.DataFrame:
    """
    Build a combined dataset from start_year to end_year (inclusive).
    """
    all_years = []
    for year in range(start_year, end_year + 1):
        season_df = load_season_results(year)
        all_years.append(season_df)

    df_all = pd.concat(all_years, ignore_index=True)
    print(f"\nTotal rows from {start_year}-{end_year}: {len(df_all)}")
    return df_all


# ---------------------------
# 2. Feature engineering
# ---------------------------

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling performance features for drivers and teams:
      - Driver_Last3_AvgFinish
      - Driver_EWMA5_AvgFinish
      - Team_Last3_AvgFinish
      - Team_EWMA5_AvgFinish

    These are computed in a leak-free way by shifting before rolling.
    """
    df = df.sort_values(["Year", "RoundNumber", "Driver"]).reset_index(drop=True)

    # Driver-based features
    df["Driver_Last3_AvgFinish"] = (
        df.groupby("Driver")["Position"]
          .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    df["Driver_EWMA5_AvgFinish"] = (
        df.groupby("Driver")["Position"]
          .transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
    )

    # Team-based features
    df["Team_Last3_AvgFinish"] = (
        df.groupby("TeamName")["Position"]
          .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    df["Team_EWMA5_AvgFinish"] = (
        df.groupby("TeamName")["Position"]
          .transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
    )

    return df


def add_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add season-level cumulative features per driver:
      - Season_AvgFinish (up to previous race)
      - Season_Wins_SoFar
      - Season_Podiums_SoFar
    """
    df = df.sort_values(["Year", "RoundNumber", "Driver"]).reset_index(drop=True)

    # Average finish so far in the season
    df["Season_AvgFinish"] = (
        df.groupby(["Year", "Driver"])["Position"]
          .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Cumulative wins and podiums so far
    df["IsWin"] = (df["Position"] == 1).astype(int)
    df["IsPodium"] = (df["Position"] <= 3).astype(int)

    df["Season_Wins_SoFar"] = (
        df.groupby(["Year", "Driver"])["IsWin"]
          .transform(lambda x: x.shift(1).cumsum())
    )

    df["Season_Podiums_SoFar"] = (
        df.groupby(["Year", "Driver"])["IsPodium"]
          .transform(lambda x: x.shift(1).cumsum())
    )

    # We don't need these as features
    df.drop(columns=["IsWin", "IsPodium"], inplace=True)

    return df


def add_team_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add long-term constructor strength features:

      - Team_AvgFinish_AllTime
      - Team_EWMAFinish_AllTime
      - TeamStrengthNorm in [0,1] (higher = stronger team)

    This is leak-free (only past races per team).
    """
    df = df.sort_values(["Year", "RoundNumber", "TeamName"]).reset_index(drop=True)

    # Expanding average of finish for each team (all-time)
    df["Team_AvgFinish_AllTime"] = (
        df.groupby("TeamName")["Position"]
          .transform(lambda x: x.shift(1).expanding().mean())
    )

    # EWMA of finish for each team (all-time)
    df["Team_EWMAFinish_AllTime"] = (
        df.groupby("TeamName")["Position"]
          .transform(lambda x: x.shift(1).ewm(span=8, min_periods=1).mean())
    )

    # Fallbacks: use short-term team metrics if all-time is NaN at the very start
    df["Team_AvgFinish_AllTime"] = df["Team_AvgFinish_AllTime"].fillna(df["Team_Last3_AvgFinish"])
    df["Team_EWMAFinish_AllTime"] = df["Team_EWMAFinish_AllTime"].fillna(df["Team_EWMA5_AvgFinish"])

    # Final fallback: global mean position
    global_mean_pos = df["Position"].mean()
    df["Team_AvgFinish_AllTime"] = df["Team_AvgFinish_AllTime"].fillna(global_mean_pos)
    df["Team_EWMAFinish_AllTime"] = df["Team_EWMAFinish_AllTime"].fillna(global_mean_pos)

    # Raw strength: lower avg finish => stronger => higher strength
    raw_strength = 1.0 / (df["Team_EWMAFinish_AllTime"] + 0.5)

    # Normalize to [0,1]
    min_raw = raw_strength.min()
    max_raw = raw_strength.max()
    df["TeamStrengthNorm"] = (raw_strength - min_raw) / (max_raw - min_raw + 1e-9)

    return df


def get_street_track_flag(race_name: str) -> int:
    """
    Very simple heuristic for street circuits.
    Extend this mapping as needed.
    """
    street_keywords = [
        "Monaco", "Singapore", "Azerbaijan", "Baku", "Saudi", "Jeddah",
        "Las Vegas", "Miami", "Melbourne", "Australia", "Canada"
    ]
    race_name = str(race_name)
    return int(any(kw.lower() in race_name.lower() for kw in street_keywords))


def add_track_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Track metadata lookup
    df["Downforce"] = df["RaceName"].apply(lambda x: TRACK_DATA.get(x, {}).get("Downforce", "medium"))
    df["TrackCategory"] = df["RaceName"].apply(lambda x: TRACK_DATA.get(x, {}).get("Category", "permanent"))
    df["TrackLengthKM"] = df["RaceName"].apply(lambda x: TRACK_DATA.get(x, {}).get("LengthKM", 5.0))
    df["Corners"] = df["RaceName"].apply(lambda x: TRACK_DATA.get(x, {}).get("Corners", 15))
    df["StopGo"] = df["RaceName"].apply(lambda x: TRACK_DATA.get(x, {}).get("StopGo", 1))

    # Add normalized round number (important!)
    max_round_per_year = df.groupby("Year")["RoundNumber"].transform("max")
    df["NormRound"] = df["RoundNumber"] / max_round_per_year

    # One-hot encode track attributes
    df = pd.get_dummies(df, columns=["Downforce", "TrackCategory"], prefix=["DF", "TC"])

    return df


def add_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add recency-focused EMAs for drivers and teams:

      - Driver_RecentFinish_EMA
      - Driver_RecentGrid_EMA
      - Driver_RecentRaceVsTeammate_EMA
      - Driver_RecentQualiDelta_EMA
      - Team_RecentFinish_EMA
    """
    df = df.sort_values(["Year", "RoundNumber", "Driver"]).reset_index(drop=True)

    df["Driver_RecentFinish_EMA"] = (
        df.groupby("Driver")["Position"]
          .transform(lambda x: x.shift(1).ewm(span=6, min_periods=1).mean())
    )

    df["Driver_RecentGrid_EMA"] = (
        df.groupby("Driver")["GridPosition"]
          .transform(lambda x: x.shift(1).ewm(span=6, min_periods=1).mean())
    )

    df["Driver_RecentRaceVsTeammate_EMA"] = (
        df.groupby("Driver")["RaceDeltaToTeammate"]
          .transform(lambda x: x.shift(1).ewm(span=6, min_periods=1).mean())
    )

    df["Driver_RecentQualiDelta_EMA"] = (
        df.groupby("Driver")["QualiDeltaToPole"]
          .transform(lambda x: x.shift(1).ewm(span=6, min_periods=1).mean())
    )

    df["Team_RecentFinish_EMA"] = (
        df.groupby("TeamName")["Position"]
          .transform(lambda x: x.shift(1).ewm(span=6, min_periods=1).mean())
    )

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature-engineering function.
    """
    df = df.copy()

    # Make sure 'Position' is numeric (FastF1 sometimes has strings for DNFs)
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")

    # Drop rows with no position (DNF, DNS, etc.) for now
    df = df.dropna(subset=["Position"]).reset_index(drop=True)

    # Rolling + season features
    df = add_rolling_features(df)
    df = add_season_features(df)

    # ---------------------------
    # Qualifying features
    # ---------------------------
    df["QualiDeltaToPole"] = (
        df["BestQualiLap"] - df.groupby(["Year", "RoundNumber"])["BestQualiLap"].transform("min")
    )

    df["QualiDeltaToMedian"] = (
        df["BestQualiLap"] - df.groupby(["Year", "RoundNumber"])["BestQualiLap"].transform("median")
    )

    df["Driver_RollingQualiDelta"] = (
        df.groupby("Driver")["QualiDeltaToPole"]
          .transform(lambda x: x.shift(1).ewm(span=4, min_periods=1).mean())
    )

    # Compute teammate quali deltas
    teams = df.groupby(["Year", "RoundNumber", "TeamName"])

    df["TeammateQualiDeltaToPole"] = teams["QualiDeltaToPole"].transform(
        lambda x: x.replace(np.nan, x.max()).max() - x
    )

    df["QualiVsTeammate"] = df["QualiDeltaToPole"] - df["TeammateQualiDeltaToPole"]

    df["GridPosition_Adjusted"] = np.log1p(df["GridPosition"])

    # ---------------------------
    # Teammate race-based features
    # ---------------------------
    team_groups = df.groupby(["Year", "RoundNumber", "TeamName"])

    df["TeammateRacePosition"] = team_groups["Position"].transform(lambda x: x.replace(np.nan, x.max()).max())
    df["RaceDeltaToTeammate"] = df["Position"] - df["TeammateRacePosition"]

    df["Driver_RollingRaceVsTeammate"] = (
        df.groupby("Driver")["RaceDeltaToTeammate"]
        .transform(lambda x: x.shift(1).ewm(span=4, min_periods=1).mean())
    )

    df["TeammateGridPosition"] = team_groups["GridPosition"].transform(lambda x: x.replace(np.nan, x.max()).max())
    df["Driver_GridVsTeammate"] = df["GridPosition"] - df["TeammateGridPosition"]

    df["Driver_RollingGridVsTeammate"] = (
        df.groupby("Driver")["Driver_GridVsTeammate"]
        .transform(lambda x: x.shift(1).ewm(span=4, min_periods=1).mean())
    )

    df["BeatTeammateInRace"] = (df["Position"] < df["TeammateRacePosition"]).astype(int)
    df["BeatTeammateInQuali"] = (df["QualiDeltaToPole"] < df["TeammateQualiDeltaToPole"]).astype(int)

    df["Season_RaceBeatTeammate"] = df.groupby(["Year", "Driver"])["BeatTeammateInRace"].transform(lambda x: x.shift(1).cumsum())
    df["Season_QualiBeatTeammate"] = df.groupby(["Year", "Driver"])["BeatTeammateInQuali"].transform(lambda x: x.shift(1).cumsum())

    # ---------------------------
    # Weighted teammate effect (Fix #2)
    # ---------------------------
    df["Weighted_RaceDeltaToTeammate"] = df["RaceDeltaToTeammate"] * 0.5
    df["Weighted_QualiVsTeammate"] = df["QualiVsTeammate"] * 0.5
    df["Weighted_RollingRaceVsTeammate"] = df["Driver_RollingRaceVsTeammate"] * 0.5
    df["Weighted_RollingGridVsTeammate"] = df["Driver_RollingGridVsTeammate"] * 0.5
    df["Weighted_RollingQualiDelta"] = df["Driver_RollingQualiDelta"] * 0.5

    # ---------------------------
    # Constructor strength (Fix #3)
    # ---------------------------
    df = add_team_strength_features(df)

    # ---------------------------
    # Recency model
    # ---------------------------
    df = add_recency_features(df)

    # ---------------------------
    # Track features
    # ---------------------------
    df = add_track_features(df)

    return df



# ---------------------------
# 3. Model training & evaluation
# ---------------------------

def get_feature_columns(df: pd.DataFrame | None = None):
    base_cols = [
        "GridPosition_Adjusted",
        "Year", "RoundNumber", "NormRound",

        # Rolling + season
        "Driver_Last3_AvgFinish",
        "Driver_EWMA5_AvgFinish",
        "Team_Last3_AvgFinish",
        "Team_EWMA5_AvgFinish",
        "Season_AvgFinish",
        "Season_Wins_SoFar",
        "Season_Podiums_SoFar",

        # Qualifying
        "QualiPosition",
        "BestQualiLap",
        "QualiDeltaToPole",
        "QualiDeltaToMedian",

        # Team strength
        "Team_AvgFinish_AllTime",
        "Team_EWMAFinish_AllTime",
        "TeamStrengthNorm",

        # Teammate (weighted)
        "Weighted_RaceDeltaToTeammate",
        "Weighted_QualiVsTeammate",
        "Weighted_RollingRaceVsTeammate",
        "Weighted_RollingGridVsTeammate",
        "Weighted_RollingQualiDelta",

        # Recency
        "Driver_RecentFinish_EMA",
        "Driver_RecentGrid_EMA",
        "Driver_RecentRaceVsTeammate_EMA",
        "Driver_RecentQualiDelta_EMA",
        "Team_RecentFinish_EMA",

        # Track
        "TrackLengthKM",
        "Corners",
        "StopGo",
    ]

    if df is not None:
        base_cols += [c for c in df.columns if c.startswith("DF_") or c.startswith("TC_")]

    return base_cols


def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Compute sample weights based on:
      - recency within a season (more recent races -> higher weight)
      - season importance (2024 > 2023 > 2022)
    """
    df = df.copy()

    # Recency within year: newer rounds get higher weight via exponential decay
    max_round = df.groupby("Year")["RoundNumber"].transform("max")
    races_ago = max_round - df["RoundNumber"]

    # Decay factor: tweak alpha if you want steeper / flatter recency curve
    alpha = 0.25
    recency_weight = np.exp(-alpha * races_ago)

    # Season-level weighting
    season_weight_map = {
        2024: 2.0,
        2023: 1.0,
        2022: 0.5,
    }
    season_weight = df["Year"].map(season_weight_map).fillna(1.0)

    sample_weight = recency_weight * season_weight
    return sample_weight.values


def prepare_train_val_split(df: pd.DataFrame):
    """
    Split into train and validation sets:

    - Train: 2022, 2023, and first half of 2024 (RoundNumber <= 10)
    - Val:   second half of 2024 (RoundNumber > 10)
    """
    feature_cols = get_feature_columns(df)

    # Fill any NaNs in features with column means (simple, but ok)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    # Define masks
    train_mask = (df["Year"] < 2024) | ((df["Year"] == 2024) & (df["RoundNumber"] <= 10))
    val_mask = (df["Year"] == 2024) & (df["RoundNumber"] > 10)

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()

    X_train = df_train[feature_cols]
    y_train = df_train["Position"]

    X_val = df_val[feature_cols]
    y_val = df_val["Position"]

    print(f"\nTrain size: {len(X_train)} rows")
    print(f"Val size:   {len(X_val)} rows")

    return X_train, y_train, X_val, y_val, feature_cols, df_train, df_val


def train_random_forest(X_train, y_train, sample_weight) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor with reasonable default hyperparameters.
    """
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining RandomForestRegressor...")
    model.fit(X_train, y_train, sample_weight=sample_weight)
    print("Training complete.")
    return model


def train_and_evaluate_multiple_models(df_features: pd.DataFrame):
    """
    Train several models (RF, XGB, LGBM, CatBoost if available) and
    evaluate them using race-level ranking metrics on the validation set.
    """
    # Prepare split
    X_train, y_train, X_val, y_val, feature_cols, df_train, df_val = prepare_train_val_split(df_features)

    # Sample weights (recency + season importance)
    train_sample_weights = compute_sample_weights(df_train)

    results = []
    trained_models = {}

    # 1) RandomForest (baseline)
    rf_model = train_random_forest(X_train, y_train, sample_weight=train_sample_weights)
    rf_metrics = evaluate_model_race_ranking(rf_model, df_val, feature_cols, model_name="RandomForest")
    results.append(rf_metrics)
    trained_models["RandomForest"] = (rf_model, feature_cols)

    # 2) XGBoost Regressor
    if xgb is not None:
        print("\nTraining XGBRegressor...")
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train, sample_weight=train_sample_weights)
        print("XGBRegressor training complete.")
        xgb_metrics = evaluate_model_race_ranking(xgb_model, df_val, feature_cols, model_name="XGBRegressor")
        results.append(xgb_metrics)
        trained_models["XGBRegressor"] = (xgb_model, feature_cols)
    else:
        print("\n[INFO] xgboost not installed; skipping XGBRegressor.")

    # 3) LightGBM Regressor
    if lgb is not None:
        print("\nTraining LGBMRegressor...")
        lgbm_model = lgb.LGBMRegressor(
            n_estimators=800,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        lgbm_model.fit(X_train, y_train, sample_weight=train_sample_weights)
        print("LGBMRegressor training complete.")
        lgbm_metrics = evaluate_model_race_ranking(lgbm_model, df_val, feature_cols, model_name="LGBMRegressor")
        results.append(lgbm_metrics)
        trained_models["LGBMRegressor"] = (lgbm_model, feature_cols)
    else:
        print("\n[INFO] lightgbm not installed; skipping LGBMRegressor.")

    # 4) CatBoost Regressor (optional)
    if CatBoostRegressor is not None:
        print("\nTraining CatBoostRegressor...")
        cat_model = CatBoostRegressor(
            iterations=800,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
        )
        cat_model.fit(X_train, y_train, sample_weight=train_sample_weights)
        print("CatBoostRegressor training complete.")
        cat_metrics = evaluate_model_race_ranking(cat_model, df_val, feature_cols, model_name="CatBoostRegressor")
        results.append(cat_metrics)
        trained_models["CatBoostRegressor"] = (cat_model, feature_cols)
    else:
        print("\n[INFO] catboost not installed; skipping CatBoostRegressor.")

    # Make a simple leaderboard (sorted by MAE)
    if results:
        print("\n==================== Model Leaderboard (race-level) ====================")
        results_sorted = sorted(results, key=lambda m: m["MAE"])
        for m in results_sorted:
            print(
                f"{m['model']:20s} | "
                f"MAE: {m['MAE']:.3f} | "
                f"RMSE: {m['RMSE']:.3f} | "
                f"Spearman: {m['Spearman']:.3f} | "
                f"Top3: {m['Top3Acc']:.3f} | "
                f"Top10: {m['Top10Acc']:.3f}"
            )

    return trained_models, results


def evaluate_model(model: RandomForestRegressor, X_val, y_val) -> float:
    """
    Evaluate model on validation set using Mean Absolute Error.
    """
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"\nValidation MAE (positions): {mae:.3f}")
    return mae


# ---------------------------
# 4. Race-level prediction helper
# ---------------------------

def predict_race_order(
    df_all: pd.DataFrame,
    model,
    feature_cols: list[str],
    year: int,
    round_number: int,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Predict finishing order for a specific race (year, round_number).

    Returns a DataFrame with:
      Driver, TeamName, GridPosition, PredictedPosition, RaceName
    sorted by PredictedPosition ascending.
    """
    df_race = df_all[(df_all["Year"] == year) & (df_all["RoundNumber"] == round_number)].copy()

    if df_race.empty:
        raise ValueError(f"No data found for {year} Round {round_number} in df_all.")

    # Fill NaNs just as in training
    df_race[feature_cols] = df_race[feature_cols].fillna(df_race[feature_cols].mean())

    X_race = df_race[feature_cols]
    preds = model.predict(X_race)

    df_race["PredictedPosition"] = preds

    df_race_sorted = df_race.sort_values("PredictedPosition").reset_index(drop=True)

    df_out = df_race_sorted[["Driver", "TeamName", "GridPosition", "PredictedPosition", "RaceName"]]
    if top_n is not None:
        df_out = df_out.head(top_n)

    print(f"\nPredicted order for {year} Round {round_number} ({df_out['RaceName'].iloc[0]}):")
    for i, row in df_out.iterrows():
        print(
            f"  P{i+1:2d}: {row['Driver']:>3s} "
            f"(Team: {row['TeamName']}, Grid: {int(row['GridPosition'])}, Score: {row['PredictedPosition']:.2f})"
        )

    return df_out


def evaluate_model_race_ranking(
    model,
    df_val: pd.DataFrame,
    feature_cols: list[str],
    model_name: str = "model",
):
    """
    Evaluate a model on race-level ranking metrics (validation set only).

    For each race in df_val:
      - predict positions for all drivers in that race
      - sort predictions to get a predicted order
      - compare to true finishing positions

    Metrics (averaged across races):
      - MAE (driver-level, for context)
      - RMSE
      - Spearman rank correlation
      - Top-3 accuracy (fraction of true top-3 drivers that appear in predicted top-3)
      - Top-10 accuracy (same but for top 10)
    """
    races = df_val[["Year", "RoundNumber"]].drop_duplicates().sort_values(["Year", "RoundNumber"])
    mae_list = []
    rmse_list = []
    spearman_list = []
    top3_acc_list = []
    top10_acc_list = []

    for _, row in races.iterrows():
        year = row["Year"]
        rnd = row["RoundNumber"]
        df_race = df_val[(df_val["Year"] == year) & (df_val["RoundNumber"] == rnd)].copy()
        if df_race.empty:
            continue

        # Features, true labels
        X_race = df_race[feature_cols]
        y_true = df_race["Position"].values

        # Predictions
        y_pred = model.predict(X_race)

        # Driver-level MAE/RMSE for this race
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae_list.append(mae)
        rmse_list.append(rmse)

        # Spearman rank correlation (ranking quality)
        try:
            rho, _ = spearmanr(y_true, y_pred)
        except Exception:
            rho = np.nan
        spearman_list.append(rho)

        # Top-3 / Top-10 accuracy (which drivers are in top-N)
        df_race = df_race.assign(PredictedPosition=y_pred)
        df_true_sorted = df_race.sort_values("Position")
        df_pred_sorted = df_race.sort_values("PredictedPosition")

        true_top3 = set(df_true_sorted["Driver"].head(3))
        pred_top3 = set(df_pred_sorted["Driver"].head(3))
        top3_acc = len(true_top3.intersection(pred_top3)) / 3.0

        true_top10 = set(df_true_sorted["Driver"].head(10))
        pred_top10 = set(df_pred_sorted["Driver"].head(10))
        top10_acc = len(true_top10.intersection(pred_top10)) / 10.0

        top3_acc_list.append(top3_acc)
        top10_acc_list.append(top10_acc)

    metrics = {
        "model": model_name,
        "MAE": float(np.nanmean(mae_list)) if mae_list else np.nan,
        "RMSE": float(np.nanmean(rmse_list)) if mae_list else np.nan,
        "Spearman": float(np.nanmean(spearman_list)) if spearman_list else np.nan,
        "Top3Acc": float(np.nanmean(top3_acc_list)) if top3_acc_list else np.nan,
        "Top10Acc": float(np.nanmean(top10_acc_list)) if top10_acc_list else np.nan,
        "NumRaces": len(mae_list),
    }

    print(f"\n=== Race-level metrics for {model_name} ===")
    print(f"Races evaluated:   {metrics['NumRaces']}")
    print(f"Avg MAE:           {metrics['MAE']:.3f}")
    print(f"Avg RMSE:          {metrics['RMSE']:.3f}")
    print(f"Avg Spearman:      {metrics['Spearman']:.3f}")
    print(f"Avg Top-3 Acc:     {metrics['Top3Acc']:.3f}")
    print(f"Avg Top-10 Acc:    {metrics['Top10Acc']:.3f}")

    return metrics


# ---------------------------
# 5. Main entry point
# ---------------------------

def main():
    # 1. Enable cache
    enable_fastf1_cache("./fastf1_cache")

    # 2. Build dataset for 2022-2024
    df_raw = build_dataset(start_year=2022, end_year=2024)

    # 3. Add all features
    df_features = add_features(df_raw)

    # 4. Train and evaluate multiple models (race-level)
    trained_models, results = train_and_evaluate_multiple_models(df_features)

    # 5. Optionally: pick the best model (lowest MAE) and show predictions for last 2024 race
    if results:
        best = sorted(results, key=lambda m: m["MAE"])[0]
        best_name = best["model"]
        print(f"\nBest model by MAE: {best_name}")

        model, feature_cols = trained_models[best_name]

        # use df_features to predict a specific race (e.g., last 2024 round)
        df_2024 = df_features[df_features["Year"] == 2024]
        if not df_2024.empty:
            last_round = int(df_2024["RoundNumber"].max())
            predict_race_order(
                df_features,
                model,
                feature_cols,
                year=2024,
                round_number=last_round,
                top_n=10,
            )


if __name__ == "__main__":
    main()
