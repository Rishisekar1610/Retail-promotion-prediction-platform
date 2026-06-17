import os
import logging
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("SUPABASE_DB_URL")
engine = create_engine(DATABASE_URL)


def clean_transactions():
    logger.info("Reading raw_transactions from Supabase...")

    df = pd.read_sql("SELECT * FROM raw_transactions", engine)

    logger.info(f"Raw rows: {len(df):,}")

    df = df.dropna(subset=["customer_id"])
    df = df[~df["invoice"].astype(str).str.startswith("C")]
    df = df[df["quantity"] > 0]
    df = df[df["price"] > 0]

    df["customer_id"] = df["customer_id"].astype(int)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["revenue"] = df["quantity"] * df["price"]

    q1 = df["revenue"].quantile(0.25)
    q3 = df["revenue"].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df = df[(df["revenue"] >= lower) & (df["revenue"] <= upper)]

    logger.info(f"Clean rows: {len(df):,}")

    df.to_sql(
        "clean_transactions",
        engine,
        if_exists="replace",
        index=False,
        chunksize=50000,
        method="multi"
    )

    logger.info("clean_transactions table created successfully.")


if __name__ == "__main__":
    clean_transactions()