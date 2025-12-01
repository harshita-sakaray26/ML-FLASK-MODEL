# crop_details.py
import pandas as pd
import pickle
from typing import List, Dict, Any

CSV_PATH = "Dataset/eggplant_details.csv"  # ಕನಿಷ್ಠವಾಗಿ ಪರಿಶೀಲಿಸಿ, ಅಗತ್ಯವಿದ್ದು ಹಾದಿ ಬದಲಾಯಿಸಿ
PICKLE_OUT = "crop_details_model.pkl"

def load_csv_flex(path: str) -> pd.DataFrame:
    """
    Robust CSV loader:
    - Tries multiple encodings and separators
    - Uses engine='python' which is more tolerant for weird files
    - Raises a clear error if none of the attempts succeed
    """
    encodings = ["utf-8", "utf-8-sig", "utf-16", "latin1", "cp1252", "iso-8859-1"]
    seps = [",", ";", "\t", "|"]
    last_err = None

    for enc in encodings:
        for sep in seps:
            try:
                # engine='python' is slower but more tolerant for mixed encodings/delimiters
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                # Basic sanity check: must have at least 1 column and 1 row
                if df.shape[0] == 0 or df.shape[1] == 0:
                    raise ValueError(f"Empty dataframe when tried encoding={enc} sep={repr(sep)}")
                print(f"[INFO] Loaded with encoding={enc!r}, sep={repr(sep)}; shape={df.shape}")
                return df
            except Exception as e:
                last_err = e
                # continue trying other combos
    # If we reach here, nothing worked — raise with helpful message and last error detail
    raise RuntimeError(
        f"Failed to read CSV at {path!r} with common encodings/separators. "
        f"Last error: {last_err}"
    ) from last_err

def ensure_column(df: pd.DataFrame, col_candidates: List[str]) -> str:
    """
    Return the name of the first matching column from candidates (case-insensitive).
    Raises if none found.
    """
    lower_cols = {c.lower(): c for c in df.columns.astype(str)}
    for cand in col_candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    raise KeyError(f"None of the expected columns {col_candidates} found in dataframe. Found columns: {list(df.columns)}")

def get_crop_info(df: pd.DataFrame, crop_name: str) -> List[Dict[str, Any]]:
    """
    Returns matching records for the given crop_name.
    - Case-insensitive and trims whitespace
    - If no exact match, returns close matches using 'contains' (substring) search
    """
    # Determine likely column name(s)
    possible_cols = ["Crop Name", "crop_name", "crop", "Name", "name"]
    try:
        crop_col = ensure_column(df, possible_cols)
    except KeyError:
        # fallback: use first column
        crop_col = df.columns[0]
        print(f"[WARN] Could not find a standard 'crop name' column. Using first column: {crop_col!r}")

    crop_name_clean = str(crop_name).strip().lower()
    # create a normalized series for matching
    series_norm = df[crop_col].astype(str).str.strip().str.lower()

    # exact match first
    exact_mask = series_norm == crop_name_clean
    if exact_mask.any():
        result = df[exact_mask]
        return result.to_dict(orient="records")

    # substring contains match next
    contains_mask = series_norm.str.contains(crop_name_clean, na=False)
    if contains_mask.any():
        result = df[contains_mask]
        return result.to_dict(orient="records")

    # approximate fallback: startswith
    starts_mask = series_norm.str.startswith(crop_name_clean, na=False)
    if starts_mask.any():
        result = df[starts_mask]
        return result.to_dict(orient="records")

    # nothing found
    return []

def main():
    try:
        data = load_csv_flex(CSV_PATH)
    except Exception as e:
        print("[ERROR] Could not load dataset:", e)
        return

    print("Dataset loaded successfully. Columns:", list(data.columns))
    print("Sample rows:")
    print(data.head(5).to_string(index=False))

    # Example usage
    crop = "Pusa Uttam"
    info = get_crop_info(data, crop)
    if not info:
        print(f"No records found for crop: {crop!r}")
    else:
        print(f"Found {len(info)} record(s) for crop: {crop!r}")
        for rec in info:
            print(rec)

    # Save DataFrame as pickle (optional)
    try:
        with open(PICKLE_OUT, "wb") as f:
            pickle.dump(data, f)
        print(f"[INFO] Pickle saved to {PICKLE_OUT}")
    except Exception as e:
        print("[WARN] Could not save pickle:", e)

if __name__ == "__main__":
    main()
