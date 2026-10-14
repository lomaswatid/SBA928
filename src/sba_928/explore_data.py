import matplotlib.pyplot as plt

from .common import RESULTS_DIR, load_data, pct


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    overall_conversion = df["Revenue"].mean()
    revenue_counts = df["Revenue"].value_counts().sort_index()

    visitor = (
        df.groupby("VisitorType", dropna=False)["Revenue"]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
    )

    month = (
        df.groupby("Month", dropna=False)["Revenue"]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
    )

    engagement = (
        df.groupby("Revenue")[["BounceRates", "ExitRates", "PageValues",
                               "ProductRelated", "ProductRelated_Duration"]]
        .mean()
    )

    summary = [
        "SBA 928 - Dataset Summary",
        "=" * 32,
        f"Rows: {len(df):,}",
        f"Columns: {len(df.columns)}",
        f"Overall conversion rate: {pct(overall_conversion)}",
        "",
        "Revenue counts:",
        revenue_counts.to_string(),
        "",
        "Visitor type:",
        visitor.to_string(),
        "",
        "Monthly conversion:",
        month.to_string(),
        "",
        "Average engagement metrics by Revenue:",
        engagement.to_string(),
    ]

    (RESULTS_DIR / "dataset_summary.txt").write_text(
        "\n".join(summary), encoding="utf-8"
    )

    # Plot 1: Revenue distribution
    ax = revenue_counts.rename(index={False: "No Purchase", True: "Purchase"}).plot(
        kind="bar", title="Revenue Outcome Distribution"
    )
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Sessions")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "revenue_distribution.png", dpi=160)
    plt.close()

    # Plot 2: Visitor conversion
    ax = (visitor["mean"] * 100).plot(
        kind="bar", title="Conversion Rate by Visitor Type"
    )
    ax.set_xlabel("Visitor Type")
    ax.set_ylabel("Conversion Rate (%)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "conversion_by_visitor_type.png", dpi=160)
    plt.close()

    # Plot 3: Monthly conversion
    ax = (month["mean"] * 100).plot(
        kind="bar", title="Observed Conversion Rate by Month"
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Conversion Rate (%)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "conversion_by_month.png", dpi=160)
    plt.close()

    # Plot 4: Selected engagement variables normalized for display
    selected = engagement[["BounceRates", "ExitRates"]].copy() * 100
    selected.index = selected.index.map({False: "No Purchase", True: "Purchase"})
    ax = selected.plot(kind="bar", title="Average Bounce and Exit Rates by Revenue")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Average Rate (%)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "engagement_by_revenue.png", dpi=160)
    plt.close()

    print(f"Saved analysis outputs to: {RESULTS_DIR}")
    print(f"Overall conversion rate: {pct(overall_conversion)}")

if __name__ == "__main__":
    main()
