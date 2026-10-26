from pathlib import Path
import argparse
import json

import pandas as pd

FEATURES = ["danceability", "energy", "tempo", "acousticness", "instrumentalness", "valence"]
REQUIRED_COLUMNS = ["track_name", "artists", "duration_ms"] + FEATURES


def clean_column_name(column):
    return str(column).strip().lower().replace(" ", "_").replace("-", "_")


def load_data(path):
    df = pd.read_csv(path)
    df.columns = [clean_column_name(column) for column in df.columns]
    return df


def clean_data(df, mood_column=None, moods=None):
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    before = len(df)

    df["_track"] = df["track_name"].astype("string").str.strip().str.casefold()
    df["_artist"] = df["artists"].astype("string").str.strip().str.casefold()
    df = df.dropna(subset=["_track", "_artist"])
    df = df.drop_duplicates(subset=["_track", "_artist"])
    df = df.drop(columns=["_track", "_artist"])

    for column in FEATURES + ["duration_ms"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=FEATURES + ["duration_ms"])

    # tempo가 0인 곡, 50초 미만, 400초(6분 40초) 초과 곡 제거
    df = df[(df["tempo"] > 0) & (df["duration_ms"] >= 50_000) & (df["duration_ms"] <= 400_000)]
    df = df.copy()

    # 무드 컬럼이 있으면 원하는 무드만 남김
    if mood_column:
        mood_column = clean_column_name(mood_column)
        if mood_column not in df.columns:
            raise ValueError(f"무드 컬럼이 없습니다: {mood_column}")

        df[mood_column] = df[mood_column].astype("string").str.strip().str.casefold()
        df = df.dropna(subset=[mood_column])

        if moods:
            moods = [mood.strip().casefold() for mood in moods]
            df = df[df[mood_column].isin(moods)]

    df = df.reset_index(drop=True)

    summary = {
        "before": before,
        "after": len(df),
        "removed": before - len(df),
        "columns": list(df.columns),
    }
    return df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/dataset.csv")
    parser.add_argument("--output", default="data/processed/tracks_clean.csv")
    parser.add_argument("--summary", default="reports/preprocess_summary.json")
    parser.add_argument("--mood-column")
    parser.add_argument("--moods", nargs="+")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    if not input_path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {input_path}")

    df = load_data(input_path)
    cleaned_df, summary = clean_data(df, mood_column=args.mood_column, moods=args.moods)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"원본 데이터: {summary['before']:,}곡")
    print(f"정제 데이터: {summary['after']:,}곡")
    print(f"저장 위치: {output_path}")


if __name__ == "__main__":
    main()