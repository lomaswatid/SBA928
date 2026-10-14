from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "online_shoppers_intention.csv"
RESULTS_DIR = ROOT / "results"
PROMPTS_DIR = ROOT / "prompts"
MODELS_DIR = ROOT / "models"

def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)

    if normalized.isna().any():
        bad = series[normalized.isna()].drop_duplicates().tolist()
        raise ValueError(f"Could not convert values to boolean: {bad}")

    return normalized.astype(bool)

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} does not exist.\n"
            "Run: uv run python src/download_dataset.py"
        )

    df = pd.read_csv(DATA_PATH)

    if "Revenue" not in df.columns:
        raise ValueError("Expected a Revenue column in the UCI dataset.")

    df["Revenue"] = normalize_bool(df["Revenue"])

    if "Weekend" in df.columns:
        df["Weekend"] = normalize_bool(df["Weekend"])

    return df

def pct(value: float) -> str:
    return f"{value * 100:.1f}%"
