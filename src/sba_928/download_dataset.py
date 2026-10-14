from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT = DATA_DIR / "online_shoppers_intention.csv"

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = fetch_ucirepo(id=468)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if isinstance(y, pd.Series):
        y = y.to_frame()

    df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    df.to_csv(OUTPUT, index=False)

    print(f"Saved dataset to: {OUTPUT}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Columns:")
    print(", ".join(df.columns))

if __name__ == "__main__":
    main()
