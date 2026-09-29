"""Per-cluster, per-customer product recommendations (notebook Step 12)."""

import pandas as pd


def generate_recommendations(
    df_filtered: pd.DataFrame,
    customer_data_cleaned: pd.DataFrame,
    top_n_products: int = 10,
    top_n_recommendations: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recommend each customer's cluster's best-sellers they haven't bought yet.

    `df_filtered` must already exclude outlier customers, and both frames must
    use the same (numeric) CustomerID dtype.
    """
    merged_data = df_filtered.merge(
        customer_data_cleaned[["CustomerID", "cluster"]], on="CustomerID", how="inner"
    )

    best_selling_products = (
        merged_data.groupby(["cluster", "StockCode", "Description"])["Quantity"]
        .sum()
        .reset_index()
        .sort_values(by=["cluster", "Quantity"], ascending=[True, False])
    )
    top_products_per_cluster = best_selling_products.groupby("cluster").head(top_n_products)

    customer_purchases = (
        merged_data.groupby(["CustomerID", "cluster", "StockCode"])["Quantity"].sum().reset_index()
    )

    columns = ["CustomerID", "cluster"]
    for i in range(1, top_n_recommendations + 1):
        columns += [f"Rec{i}_StockCode", f"Rec{i}_Description"]

    recommendations = []
    for cluster in top_products_per_cluster["cluster"].unique():
        top_products = top_products_per_cluster[top_products_per_cluster["cluster"] == cluster]
        customers_in_cluster = customer_data_cleaned[
            customer_data_cleaned["cluster"] == cluster
        ]["CustomerID"]

        for customer in customers_in_cluster:
            purchased = customer_purchases[
                (customer_purchases["CustomerID"] == customer)
                & (customer_purchases["cluster"] == cluster)
            ]["StockCode"].tolist()

            not_purchased = top_products[~top_products["StockCode"].isin(purchased)]
            top_recs = not_purchased.head(top_n_recommendations)

            row = [customer, cluster]
            for _, rec in top_recs.iterrows():
                row += [rec["StockCode"], rec["Description"]]
            # Pad so every row has the same length even if a cluster doesn't
            # have enough unbought best-sellers left to fill all N slots.
            while len(row) < len(columns):
                row += [None, None]
            recommendations.append(row[: len(columns)])

    recommendations_df = pd.DataFrame(recommendations, columns=columns)
    return recommendations_df, top_products_per_cluster
