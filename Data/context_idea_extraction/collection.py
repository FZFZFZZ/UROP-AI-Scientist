#!/usr/bin/env python3
"""
extract.py — Randomly sample 100 instances (id + abstract) from the ICLR2025 parquet file
and export to JSONL format.
"""

import pandas as pd
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Sample 100 instances from ICLR2025 parquet")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input parquet file (e.g., iclr2025.parquet)")
    parser.add_argument("--output", type=str, default="sample_iclr2025.jsonl",
                        help="Output JSONL filename")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of samples (default=100)")
    parser.add_argument("--seed", type=int, default=20020515,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File not found: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"✅ Loaded {len(df)} rows from {args.input}")

    for col in ["id", "abstract"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}'")

    sample_df = df.sample(n=min(args.n, len(df)), random_state=args.seed)[["id", "abstract"]]
    print(f"🎯 Sampled {len(sample_df)} rows")

    with open(args.output, "w", encoding="utf-8") as f:
        for _, row in sample_df.iterrows():
            json.dump({"id": row["id"], "abstract": row["abstract"]}, f, ensure_ascii=False)
            f.write("\n")

    print(f"💾 Saved {len(sample_df)} samples to {args.output}")

if __name__ == "__main__":
    main()
