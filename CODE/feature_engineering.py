import os
from pathlib import Path
from dotenv import load_dotenv

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, max as spark_max, countDistinct, count, sum as spark_sum,
    avg, datediff, date_add, lit, rand, when
)
from pyspark.sql.window import Window
from pyspark.sql.functions import ntile
from sqlalchemy.engine import make_url

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("SUPABASE_DB_URL")

url = make_url(DATABASE_URL)

DB_HOST = url.host
DB_PORT = url.port
DB_USER = url.username
DB_PASSWORD = url.password
DB_NAME = url.database

JDBC_URL = f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}"


spark = (
    SparkSession.builder
    .appName("RetailPromoPySparkFeatureEngineering")
    .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3")
    .getOrCreate()
)


def read_clean_transactions():
    df = (
        spark.read.format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", "clean_transactions")
        .option("user", DB_USER)
        .option("password", DB_PASSWORD)
        .option("driver", "org.postgresql.Driver")
        .load()
    )

    return df


def build_features(df):
    snapshot_date = df.select(
        date_add(spark_max("invoice_date"), 1).alias("snapshot_date")
    ).collect()[0]["snapshot_date"]

    rfm = (
        df.groupBy("customer_id")
        .agg(
            datediff(lit(snapshot_date), spark_max("invoice_date")).alias("recency"),
            countDistinct("invoice").alias("frequency"),
            spark_sum("revenue").alias("monetary")
        )
    )

    r_window = Window.orderBy(col("recency").asc())
    f_window = Window.orderBy(col("frequency").asc())
    m_window = Window.orderBy(col("monetary").asc())

    rfm = (
        rfm
        .withColumn("r_score_raw", ntile(4).over(r_window))
        .withColumn("f_score", ntile(4).over(f_window))
        .withColumn("m_score", ntile(4).over(m_window))
        .withColumn("r_score", 5 - col("r_score_raw"))
        .withColumn("rfm_score", col("r_score") + col("f_score") + col("m_score"))
    )

    purchase_features = (
        df.groupBy("customer_id")
        .agg(
            spark_sum("revenue").alias("total_spend"),
            avg("revenue").alias("avg_basket_size"),
            countDistinct("invoice").alias("purchase_frequency"),
            count("invoice").alias("total_transactions"),
            countDistinct("stock_code").alias("unique_products"),
            avg("quantity").alias("avg_quantity"),
            spark_sum("quantity").alias("total_quantity"),
            datediff(lit(snapshot_date), spark_max("invoice_date")).alias("days_since_purchase")
        )
    )

    feature_df = (
        purchase_features
        .join(
            rfm.select(
                "customer_id",
                "recency",
                "frequency",
                "monetary",
                "rfm_score"
            ),
            on="customer_id",
            how="left"
        )
    )

    feature_df = feature_df.withColumn(
        "customer_lifetime_value",
        col("avg_basket_size") * col("purchase_frequency") * lit(0.65)
    )

    max_spend = feature_df.agg(spark_max("total_spend")).collect()[0][0]
    max_frequency = feature_df.agg(spark_max("purchase_frequency")).collect()[0][0]

    feature_df = feature_df.withColumn(
        "redemption_probability",
        (
            lit(0.10)
            + (col("total_spend") / lit(max_spend)) * lit(0.20)
            + (col("purchase_frequency") / lit(max_frequency)) * lit(0.20)
        )
    )

    feature_df = feature_df.withColumn(
        "redeemed_offer",
        when(rand(seed=42) < col("redemption_probability"), 1).otherwise(0)
    )

    feature_df = feature_df.withColumn(
        "segment",
        when(col("rfm_score") >= 10, "High Value")
        .when((col("rfm_score") >= 7) & (col("rfm_score") < 10), "Occasional")
        .when((col("rfm_score") >= 4) & (col("rfm_score") < 7), "At Risk")
        .otherwise("New Customer")
    )

    return feature_df


def write_customer_features(feature_df):
    (
        feature_df.write.format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", "customer_features")
        .option("user", DB_USER)
        .option("password", DB_PASSWORD)
        .option("driver", "org.postgresql.Driver")
        .mode("overwrite")
        .save()
    )


if __name__ == "__main__":
    clean_df = read_clean_transactions()
    feature_df = build_features(clean_df)

    print("Customer feature rows:", feature_df.count())
    feature_df.show(10)

    write_customer_features(feature_df)

    print("customer_features table created successfully.")

    spark.stop()