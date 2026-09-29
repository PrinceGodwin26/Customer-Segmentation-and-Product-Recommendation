"""Data loading and cleaning, ported from the notebook's Step 1-3 and 4."""

import numpy as np
import pandas as pd
from scipy.stats import linregress


def load_raw_data(path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="ISO-8859-1")


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw transactional data.

    Mirrors the notebook's Step 3 (missing values, duplicates, cancelled
    transactions, anomalous stock codes, description text, zero prices).
    Returns the cleaned dataframe plus a report of the data-quality issues
    found along the way, so the API/dashboard can surface them.
    """
    df = df.copy()
    report: dict = {"initial_rows": int(len(df))}

    missing = df.isnull().sum()
    report["missing_value_pct"] = (
        (missing[missing > 0] / len(df) * 100).round(2).to_dict()
    )

    df = df.dropna(subset=["CustomerID", "Description"])

    report["duplicate_rows"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    df["Transaction_Status"] = np.where(
        df["InvoiceNo"].astype(str).str.startswith("C"), "Cancelled", "Completed"
    )
    report["cancelled_pct"] = round((df["Transaction_Status"] == "Cancelled").mean() * 100, 2)

    unique_stock_codes = df["StockCode"].unique()
    anomalous_stock_codes = [
        code for code in unique_stock_codes if sum(c.isdigit() for c in str(code)) in (0, 1)
    ]
    report["anomalous_stockcode_pct"] = round(
        df["StockCode"].isin(anomalous_stock_codes).mean() * 100, 2
    )
    df = df[~df["StockCode"].isin(anomalous_stock_codes)]

    service_related_descriptions = ["Next Day Carriage", "High Resolution Image"]
    report["service_related_pct"] = round(
        df["Description"].isin(service_related_descriptions).mean() * 100, 2
    )
    df = df[~df["Description"].isin(service_related_descriptions)]
    df["Description"] = df["Description"].str.upper()

    df = df[df["UnitPrice"] > 0]

    df = df.reset_index(drop=True)
    report["final_rows"] = int(len(df))
    return df, report


def _calculate_trend(spend_data: pd.Series) -> float:
    if len(spend_data) > 1:
        x = np.arange(len(spend_data))
        slope, _, _, _, _ = linregress(x, spend_data)
        return slope
    return 0.0


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the customer-centric feature table (notebook Step 4)."""
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceDay"] = df["InvoiceDate"].dt.date

    # Recency
    customer_data = df.groupby("CustomerID")["InvoiceDay"].max().reset_index()
    most_recent_date = pd.to_datetime(df["InvoiceDay"].max())
    customer_data["InvoiceDay"] = pd.to_datetime(customer_data["InvoiceDay"])
    customer_data["Days_Since_Last_Purchase"] = (
        most_recent_date - customer_data["InvoiceDay"]
    ).dt.days
    customer_data = customer_data.drop(columns=["InvoiceDay"])

    # Frequency
    total_transactions = (
        df.groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .reset_index()
        .rename(columns={"InvoiceNo": "Total_Transactions"})
    )
    total_products_purchased = (
        df.groupby("CustomerID")["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"Quantity": "Total_Products_Purchased"})
    )
    customer_data = customer_data.merge(total_transactions, on="CustomerID").merge(
        total_products_purchased, on="CustomerID"
    )

    # Monetary
    df["Total_Spend"] = df["UnitPrice"] * df["Quantity"]
    total_spend = df.groupby("CustomerID")["Total_Spend"].sum().reset_index()
    average_transaction_value = total_spend.merge(total_transactions, on="CustomerID")
    average_transaction_value["Average_Transaction_Value"] = (
        average_transaction_value["Total_Spend"] / average_transaction_value["Total_Transactions"]
    )
    customer_data = customer_data.merge(total_spend, on="CustomerID").merge(
        average_transaction_value[["CustomerID", "Average_Transaction_Value"]], on="CustomerID"
    )

    # Product diversity
    unique_products_purchased = (
        df.groupby("CustomerID")["StockCode"]
        .nunique()
        .reset_index()
        .rename(columns={"StockCode": "Unique_Products_Purchased"})
    )
    customer_data = customer_data.merge(unique_products_purchased, on="CustomerID")

    # Behavioral features
    df["Day_Of_Week"] = df["InvoiceDate"].dt.dayofweek
    df["Hour"] = df["InvoiceDate"].dt.hour

    days_between_purchases = df.groupby("CustomerID")["InvoiceDay"].apply(
        lambda x: (x.diff().dropna()).apply(lambda y: y.days)
    )
    average_days_between_purchases = (
        days_between_purchases.groupby("CustomerID")
        .mean()
        .reset_index()
        .rename(columns={"InvoiceDay": "Average_Days_Between_Purchases"})
    )

    favorite_shopping_day = df.groupby(["CustomerID", "Day_Of_Week"]).size().reset_index(name="Count")
    favorite_shopping_day = favorite_shopping_day.loc[
        favorite_shopping_day.groupby("CustomerID")["Count"].idxmax()
    ][["CustomerID", "Day_Of_Week"]]

    favorite_shopping_hour = df.groupby(["CustomerID", "Hour"]).size().reset_index(name="Count")
    favorite_shopping_hour = favorite_shopping_hour.loc[
        favorite_shopping_hour.groupby("CustomerID")["Count"].idxmax()
    ][["CustomerID", "Hour"]]

    customer_data = (
        customer_data.merge(average_days_between_purchases, on="CustomerID")
        .merge(favorite_shopping_day, on="CustomerID")
        .merge(favorite_shopping_hour, on="CustomerID")
    )

    # Geography
    customer_country = df.groupby(["CustomerID", "Country"]).size().reset_index(name="Number_of_Transactions")
    customer_main_country = customer_country.sort_values(
        "Number_of_Transactions", ascending=False
    ).drop_duplicates("CustomerID")
    customer_main_country["Is_UK"] = customer_main_country["Country"].apply(
        lambda x: 1 if x == "United Kingdom" else 0
    )
    customer_data = customer_data.merge(
        customer_main_country[["CustomerID", "Is_UK"]], on="CustomerID", how="left"
    )

    # Cancellation insights
    cancelled_transactions = df[df["Transaction_Status"] == "Cancelled"]
    cancellation_frequency = (
        cancelled_transactions.groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .reset_index()
        .rename(columns={"InvoiceNo": "Cancellation_Frequency"})
    )
    customer_data = customer_data.merge(cancellation_frequency, on="CustomerID", how="left")
    customer_data["Cancellation_Frequency"] = customer_data["Cancellation_Frequency"].fillna(0)
    customer_data = customer_data.merge(
        total_transactions.rename(columns={"Total_Transactions": "_Total_Transactions_For_Rate"}),
        on="CustomerID",
    )
    customer_data["Cancellation_Rate"] = (
        customer_data["Cancellation_Frequency"] / customer_data["_Total_Transactions_For_Rate"]
    )
    customer_data = customer_data.drop(columns=["_Total_Transactions_For_Rate"])

    # Seasonality & trends
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    monthly_spending = df.groupby(["CustomerID", "Year", "Month"])["Total_Spend"].sum().reset_index()
    seasonal_buying_patterns = (
        monthly_spending.groupby("CustomerID")["Total_Spend"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "Monthly_Spending_Mean", "std": "Monthly_Spending_Std"})
    )
    seasonal_buying_patterns["Monthly_Spending_Std"] = seasonal_buying_patterns[
        "Monthly_Spending_Std"
    ].fillna(0)

    spending_trends = (
        monthly_spending.groupby("CustomerID")["Total_Spend"]
        .apply(_calculate_trend)
        .reset_index()
        .rename(columns={"Total_Spend": "Spending_Trend"})
    )

    customer_data = customer_data.merge(seasonal_buying_patterns, on="CustomerID").merge(
        spending_trends, on="CustomerID"
    )

    customer_data["CustomerID"] = customer_data["CustomerID"].astype(str)
    customer_data = customer_data.convert_dtypes()
    return customer_data
