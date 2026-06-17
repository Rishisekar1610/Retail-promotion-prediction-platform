
import os
import logging
from pathlib import Path
from io import StringIO

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("SUPABASE_DB_URL")

print("Using DB URL:", DATABASE_URL.split("@")[-1])

if not DATABASE_URL:
    raise RuntimeError(
        "SUPABASE_DB_URL not found. Add it to your .env file."
    )


BASE_DIR = Path(__file__).resolve().parent

RAW_FILE = BASE_DIR / "Data" / "online_retail_II.xlsx"

TABLE_NAME = "raw_transactions"

COLUMN_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def load_raw_excel() -> pd.DataFrame:
    logger.info("Reading Online Retail II Excel file...")

    df_2009 = pd.read_excel(
        RAW_FILE,
        sheet_name="Year 2009-2010",
        engine="openpyxl"
    )

    df_2010 = pd.read_excel(
        RAW_FILE,
        sheet_name="Year 2010-2011",
        engine="openpyxl"
    )

    df = pd.concat([df_2009, df_2010], ignore_index=True)

    logger.info(f"Loaded {len(df):,} rows from Excel")

    df = df.rename(columns=COLUMN_MAP)
    df = df[list(COLUMN_MAP.values())]

    df["customer_id"] = df["customer_id"].astype("Int64")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    return df


def create_table(engine) -> None:
    logger.info(f"Creating table if not exists: {TABLE_NAME}")

    query = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        invoice TEXT,
        stock_code TEXT,
        description TEXT,
        quantity INTEGER,
        invoice_date TIMESTAMP,
        price NUMERIC,
        customer_id INTEGER,
        country TEXT
    );
    """

    with engine.begin() as conn:
        conn.execute(text(query))


def bulk_insert(df: pd.DataFrame, engine, truncate_first: bool = True) -> None:
    logger.info("Starting bulk insert into Supabase...")

    raw_conn = engine.raw_connection()

    try:
        cursor = raw_conn.cursor()

        if truncate_first:
            logger.info(f"Truncating existing table: {TABLE_NAME}")
            cursor.execute(f"TRUNCATE TABLE {TABLE_NAME};")

        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False, na_rep="")
        buffer.seek(0)

        cursor.copy_expert(
            f"""
            COPY {TABLE_NAME}
            FROM STDIN
            WITH (FORMAT CSV, NULL '')
            """,
            buffer
        )

        raw_conn.commit()

        logger.info(f"Inserted {len(df):,} rows into {TABLE_NAME}")

    finally:
        raw_conn.close()


def verify_load(engine) -> int:
    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT COUNT(*) FROM {TABLE_NAME};")
        )
        count = result.scalar()

    logger.info(f"Final row count in {TABLE_NAME}: {count:,}")
    return count


def main() -> None:
    logger.info("=" * 60)
    logger.info("SUPABASE RETAIL DATA LOAD")
    logger.info("=" * 60)

    engine = create_engine(DATABASE_URL)

    df = load_raw_excel()

    create_table(engine)

    bulk_insert(df, engine)

    verify_load(engine)

    logger.info("Data load completed successfully.")


if __name__ == "__main__":
    main()