"""Generates a synthetic placeholder dataset matching the real UK online
retail export's schema (InvoiceNo, StockCode, Description, Quantity,
InvoiceDate, UnitPrice, CustomerID, Country), including the kinds of dirty
rows (missing values, duplicates, cancellations, anomalous stock codes,
zero prices) the cleaning pipeline is built to handle.

Run once to (re)create data/placeholder_data.csv:
    python scripts/generate_placeholder_data.py
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "placeholder_data.csv"

N_CUSTOMERS = 120
COUNTRIES = (
    ["United Kingdom"] * 85
    + ["Germany"] * 5
    + ["France"] * 4
    + ["EIRE"] * 3
    + ["Spain"] * 2
    + ["Netherlands"] * 1
)

PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER"),
    ("71053", "WHITE METAL LANTERN"),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER"),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE"),
    ("84029E", "RED WOOLLY HOTTIE WHITE HEART"),
    ("22752", "SET 7 BABUSHKA NESTING BOXES"),
    ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER"),
    ("22633", "HAND WARMER UNION JACK"),
    ("22632", "HAND WARMER RED POLKA DOT"),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT"),
    ("22745", "POPPY'S PLAYHOUSE BEDROOM"),
    ("22748", "POPPY'S PLAYHOUSE KITCHEN"),
    ("22749", "FELTCRAFT PRINCESS CHARLOTTE DOLL"),
    ("22310", "IVORY KNITTED MUG COSY"),
    ("84969", "BOX OF 6 ASSORTED COLOUR TEASPOONS"),
    ("22623", "BOX OF VINTAGE JIGSAW BLOCKS"),
    ("22622", "BOX OF VINTAGE ALPHABET BLOCKS"),
    ("21754", "HOME BUILDING BLOCK WORD"),
    ("21755", "LOVE BUILDING BLOCK WORD"),
    ("21777", "RECIPE BOX WITH METAL HEART"),
    ("48187", "DOORMAT NEW ENGLAND"),
    ("22961", "JAM MAKING SET WITH JARS"),
    ("22960", "JAM MAKING SET PRINTED"),
    ("22138", "BAKING SET 9 PIECE RETROSPOT"),
    ("22139", "RETROSPOT TEA SET CERAMIC 11 PC"),
    ("23203", "JUMBO BAG DOILEY PATTERNS"),
    ("23204", "GYMKHANA TREASURE BOOK BOX"),
    ("21212", "PACK OF 72 RETROSPOT CAKE CASES"),
    ("20725", "LUNCH BAG RED RETROSPOT"),
    ("20726", "LUNCH BAG WOODLAND"),
    ("85099B", "JUMBO BAG RED RETROSPOT"),
    ("85099C", "JUMBO BAG BAROQUE BLACK WHITE"),
    ("23298", "SPOTTY BUNTING"),
    ("23299", "FOOD COVER WITH BEADS SET 2"),
]

ANOMALOUS_STOCK_CODES = ["POST", "DOT", "M", "D", "BANK CHARGES"]
SERVICE_DESCRIPTIONS = ["Next Day Carriage", "High Resolution Image"]

START_DATE = datetime(2011, 1, 4, 8, 0)
END_DATE = datetime(2011, 12, 9, 18, 0)


def random_datetime() -> datetime:
    span = (END_DATE - START_DATE).total_seconds()
    return START_DATE + timedelta(seconds=random.uniform(0, span))


def main() -> None:
    rows = []
    invoice_counter = 536365
    customer_ids = np.round(np.random.uniform(12000, 18500, N_CUSTOMERS))

    for customer_id in customer_ids:
        country = random.choice(COUNTRIES)
        n_invoices = random.randint(3, 12)

        # Spread invoices across distinct days so recency/behavioral features
        # (e.g. Average_Days_Between_Purchases) have real values to compute.
        invoice_days = sorted(
            {random_datetime().replace(hour=0, minute=0, second=0) for _ in range(n_invoices)}
        )
        while len(invoice_days) < 2:
            invoice_days.append(invoice_days[0] + timedelta(days=random.randint(1, 20)))

        for day in invoice_days:
            invoice_counter += 1
            invoice_no = str(invoice_counter)
            is_cancelled = random.random() < 0.03
            if is_cancelled:
                invoice_no = "C" + invoice_no

            n_items = random.randint(1, 6)
            invoice_time = day + timedelta(hours=random.randint(6, 19), minutes=random.randint(0, 59))

            for _ in range(n_items):
                stock_code, description = random.choice(PRODUCTS)
                quantity = random.randint(1, 20)
                if is_cancelled:
                    quantity = -quantity
                unit_price = round(random.uniform(0.42, 24.90), 2)

                rows.append(
                    {
                        "InvoiceNo": invoice_no,
                        "StockCode": stock_code,
                        "Description": description,
                        "Quantity": quantity,
                        "InvoiceDate": invoice_time.strftime("%m/%d/%Y %H:%M"),
                        "UnitPrice": unit_price,
                        "CustomerID": customer_id,
                        "Country": country,
                    }
                )

    df = pd.DataFrame(rows)

    # --- sprinkle in the kinds of dirty data the cleaning pipeline handles ---
    n = len(df)
    rng = np.random.default_rng(42)

    # Missing CustomerID / Description
    missing_idx = rng.choice(n, size=max(1, int(n * 0.02)), replace=False)
    df.loc[missing_idx[: len(missing_idx) // 2], "CustomerID"] = np.nan
    df.loc[missing_idx[len(missing_idx) // 2 :], "Description"] = np.nan

    # Duplicate rows
    dup_sample = df.sample(n=max(1, int(n * 0.01)), random_state=1)
    df = pd.concat([df, dup_sample], ignore_index=True)

    # Anomalous stock codes (non-product adjustment codes)
    anomalous_idx = rng.choice(len(df), size=max(1, int(n * 0.01)), replace=False)
    for i, idx in enumerate(anomalous_idx):
        df.loc[idx, "StockCode"] = ANOMALOUS_STOCK_CODES[i % len(ANOMALOUS_STOCK_CODES)]

    # Service-related descriptions
    service_idx = rng.choice(len(df), size=max(1, int(n * 0.005)), replace=False)
    for i, idx in enumerate(service_idx):
        df.loc[idx, "Description"] = SERVICE_DESCRIPTIONS[i % len(SERVICE_DESCRIPTIONS)]

    # Zero unit price rows
    zero_price_idx = rng.choice(len(df), size=max(1, int(n * 0.01)), replace=False)
    df.loc[zero_price_idx, "UnitPrice"] = 0.0

    # Lowercase description noise
    lowercase_idx = rng.choice(len(df), size=max(1, int(n * 0.005)), replace=False)
    df.loc[lowercase_idx, "Description"] = df.loc[lowercase_idx, "Description"].str.lower()

    df = df.sample(frac=1, random_state=7).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
