"""Generate synthetic StoreOps data with the same schema as the seed CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Seed CSV path.")
    parser.add_argument("--output", required=True, help="Output synthetic CSV path.")
    parser.add_argument("--rows", type=int, default=5000, help="Number of synthetic rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    seed_df = pd.read_csv(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stores = [f"Store_{index:03d}" for index in range(1, 31)]
    cities = ["Delhi", "Mumbai", "Bengaluru", "Pune", "Hyderabad"]
    items = [f"Inventory_{index:03d}" for index in range(1, 41)]
    products = [f"Product_{index:03d}" for index in range(1, 26)]

    sampled_rows = []
    for index in range(args.rows):
        base_row = seed_df.iloc[index % len(seed_df)].copy()
        qty = max(0.5, float(base_row["QTY"]) * rng.uniform(0.7, 1.4))
        ideals = max(0.5, float(base_row["IDEALS"]) * rng.uniform(0.7, 1.3))
        count = max(1, int(round(float(base_row["COUNT"]) * rng.uniform(0.8, 1.2))))
        prod_quantity = max(
            1,
            int(round(float(base_row["PROD_QUANTITY"]) * rng.uniform(0.8, 1.3))),
        )

        base_row["STORE_NAME"] = rng.choice(stores)
        base_row["STORE_CODE"] = f"ST{rng.randint(100, 999)}"
        base_row["CITY"] = rng.choice(cities)
        base_row["STATE"] = "SyntheticState"
        base_row["PRODUCT_NAME"] = rng.choice(products)
        base_row["PRODUCT_CODE"] = f"PRD{rng.randint(100000, 999999)}"
        base_row["INVENTORY_NAME"] = rng.choice(items)
        base_row["INVENTORY_ITEM_CODE"] = f"INV{rng.randint(1000, 9999)}"
        base_row["ORDER_CRN"] = f"SYN{1000000000 + index}"
        base_row["ORDER_LINE_CODE"] = f"SUB{2000000000 + index}"
        base_row["PRODUCT_DETAIL_CODE"] = base_row["PRODUCT_CODE"]
        base_row["QTY"] = round(qty, 2)
        base_row["IDEALS"] = round(ideals, 2)
        base_row["COUNT"] = count
        base_row["PROD_QUANTITY"] = prod_quantity
        base_row["ITEM_PRICE"] = round(float(base_row["ITEM_PRICE"]) * rng.uniform(0.9, 1.1), 2)
        base_row["INVENTORY_ITEM_PRICE"] = round(
            float(base_row["INVENTORY_ITEM_PRICE"]) * rng.uniform(0.9, 1.15),
            2,
        )
        base_row["INVENTORY_ITEM_BASIC_PRICE"] = round(
            float(base_row["INVENTORY_ITEM_BASIC_PRICE"]) * rng.uniform(0.9, 1.15),
            2,
        )
        base_row["IDEAL_WACC"] = round(float(base_row["IDEAL_WACC"]) * rng.uniform(0.85, 1.1), 2)
        base_row["Area Manager"] = f"AreaManager_{rng.randint(1, 8):02d}"
        base_row["Zonal Manager"] = f"ZonalManager_{rng.randint(1, 5):02d}"
        sampled_rows.append(base_row)

    synthetic_df = pd.DataFrame(sampled_rows)
    synthetic_df.to_csv(output_path, index=False)
    print(f"Wrote {len(synthetic_df)} synthetic rows to {output_path}")


if __name__ == "__main__":
    main()
