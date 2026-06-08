from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_DIR = Path.cwd()
BASE_DIR = PROJECT_DIR / "outputs" / "【保留】ifb214_full_feature_k_interpretability" / "tables"
PROFILE_PATH = BASE_DIR / "cluster_profile_k5.csv"
ASSIGNMENTS_PATH = BASE_DIR / "cluster_assignments_k3_to_k6.csv"
FEATURES_PATH = BASE_DIR / "review_level_feature_table_full_input.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "drunk_elephant_cluster_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BRANDS = ["Drunk Elephant", "Tatcha", "The Ordinary"]
EQUAL_SHARE = 100 / len(TARGET_BRANDS)


def main() -> None:
    profile = pd.read_csv(PROFILE_PATH)
    assignments = pd.read_csv(ASSIGNMENTS_PATH)
    features = pd.read_csv(FEATURES_PATH)

    merged = pd.merge(
        features[["review_id", "brand", "rating", "is_recommended", "sentiment_score"]],
        assignments[["review_id", "cluster_k5"]],
        on="review_id",
        how="left",
    ).rename(columns={"cluster_k5": "cluster"})

    cluster_name_map = {
        int(row.cluster): str(row.draft_business_label)
        for row in profile.itertuples(index=False)
    }
    cluster_rating_map = {
        int(row.cluster): float(row.average_rating)
        for row in profile.itertuples(index=False)
    }
    cluster_rec_map = {
        int(row.cluster): float(row.recommendation_rate)
        for row in profile.itertuples(index=False)
    }

    brand_cluster = (
        merged.groupby(["cluster", "brand"])
        .agg(
            review_count=("review_id", "size"),
            avg_rating=("rating", "mean"),
            recommendation_rate=("is_recommended", lambda s: float(s.mean() * 100)),
            avg_sentiment=("sentiment_score", "mean"),
        )
        .reset_index()
    )
    brand_cluster["cluster_total"] = brand_cluster.groupby("cluster")["review_count"].transform("sum")
    brand_cluster["brand_share_pct"] = brand_cluster["review_count"] / brand_cluster["cluster_total"] * 100
    brand_cluster["over_index_vs_equal_share_pct"] = brand_cluster["brand_share_pct"] - EQUAL_SHARE
    brand_cluster["cluster_name"] = brand_cluster["cluster"].map(cluster_name_map)
    brand_cluster["cluster_avg_rating"] = brand_cluster["cluster"].map(cluster_rating_map)
    brand_cluster["cluster_rec_rate"] = brand_cluster["cluster"].map(cluster_rec_map)
    brand_cluster["rating_vs_cluster_avg"] = brand_cluster["avg_rating"] - brand_cluster["cluster_avg_rating"]
    brand_cluster["rec_vs_cluster_avg"] = brand_cluster["recommendation_rate"] - brand_cluster["cluster_rec_rate"]

    brand_cluster = brand_cluster[
        [
            "cluster",
            "cluster_name",
            "brand",
            "review_count",
            "brand_share_pct",
            "over_index_vs_equal_share_pct",
            "avg_rating",
            "recommendation_rate",
            "avg_sentiment",
            "cluster_avg_rating",
            "cluster_rec_rate",
            "rating_vs_cluster_avg",
            "rec_vs_cluster_avg",
        ]
    ].copy()

    brand_cluster = brand_cluster.sort_values(["cluster", "brand"]).reset_index(drop=True)
    brand_cluster.to_csv(OUT_DIR / "brand_cluster_profile_k5.csv", index=False, encoding="utf-8-sig")

    de_only = brand_cluster[brand_cluster["brand"] == "Drunk Elephant"].copy().reset_index(drop=True)
    de_only = de_only.sort_values("brand_share_pct", ascending=False).reset_index(drop=True)
    de_only.to_csv(OUT_DIR / "drunk_elephant_share_overindex_k5.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for cluster_id, cluster_df in brand_cluster.groupby("cluster"):
        cluster_df = cluster_df.sort_values("brand_share_pct", ascending=False).reset_index(drop=True)
        de_row = cluster_df[cluster_df["brand"] == "Drunk Elephant"].iloc[0]
        top_brand = cluster_df.iloc[0]
        tatcha_row = cluster_df[cluster_df["brand"] == "Tatcha"].iloc[0]
        ordinary_row = cluster_df[cluster_df["brand"] == "The Ordinary"].iloc[0]

        summary_rows.append(
            {
                "cluster": int(cluster_id),
                "cluster_name": cluster_name_map[int(cluster_id)],
                "drunk_elephant_share_pct": round(float(de_row["brand_share_pct"]), 2),
                "drunk_elephant_over_index_pct": round(float(de_row["over_index_vs_equal_share_pct"]), 2),
                "tatcha_share_pct": round(float(tatcha_row["brand_share_pct"]), 2),
                "the_ordinary_share_pct": round(float(ordinary_row["brand_share_pct"]), 2),
                "leading_brand": str(top_brand["brand"]),
                "drunk_elephant_rating": round(float(de_row["avg_rating"]), 2),
                "drunk_elephant_rec_rate": round(float(de_row["recommendation_rate"]), 1),
                "tatcha_rating": round(float(tatcha_row["avg_rating"]), 2),
                "tatcha_rec_rate": round(float(tatcha_row["recommendation_rate"]), 1),
                "the_ordinary_rating": round(float(ordinary_row["avg_rating"]), 2),
                "the_ordinary_rec_rate": round(float(ordinary_row["recommendation_rate"]), 1),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("drunk_elephant_share_pct", ascending=False).reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / "drunk_elephant_cluster_comparison_summary.csv", index=False, encoding="utf-8-sig")

    weakness_points = []
    for row in summary_df.itertuples(index=False):
        peers = [
            ("Tatcha", float(row.tatcha_rating), float(row.tatcha_rec_rate), float(row.tatcha_share_pct)),
            ("The Ordinary", float(row.the_ordinary_rating), float(row.the_ordinary_rec_rate), float(row.the_ordinary_share_pct)),
        ]
        de_rating = float(row.drunk_elephant_rating)
        de_rec = float(row.drunk_elephant_rec_rate)
        lower_than_both = all(de_rating < p[1] for p in peers) or all(de_rec < p[2] for p in peers)
        de_high_share = float(row.drunk_elephant_share_pct) >= 33.33
        if lower_than_both or de_high_share:
            weakness_points.append(
                {
                    "cluster": int(row.cluster),
                    "cluster_name": row.cluster_name,
                    "drunk_elephant_share_pct": row.drunk_elephant_share_pct,
                    "drunk_elephant_over_index_pct": row.drunk_elephant_over_index_pct,
                    "key_issue": (
                        "Drunk Elephant is relatively concentrated here and/or underperforms peer brands on rating or recommendation within this experience type."
                    ),
                }
            )
    weakness_df = pd.DataFrame(weakness_points)
    weakness_df.to_csv(OUT_DIR / "drunk_elephant_relative_shortcomings.csv", index=False, encoding="utf-8-sig")

    interpretation_lines = [
        "# Drunk Elephant cluster diagnosis",
        "",
        "## What the over-index means",
        "Over-index is calculated relative to an equal three-brand baseline of 33.3%.",
        "A positive value means Drunk Elephant is more concentrated in that cluster than expected under an even brand mix.",
        "",
        "## Main findings",
    ]

    for row in summary_df.itertuples(index=False):
        if row.drunk_elephant_over_index_pct >= 5:
            interpretation_lines.append(
                f"- `{row.cluster_name}`: Drunk Elephant is over-indexed at {row.drunk_elephant_share_pct:.1f}% ({row.drunk_elephant_over_index_pct:+.1f} pts vs equal share)."
            )
        elif row.drunk_elephant_over_index_pct <= -5:
            interpretation_lines.append(
                f"- `{row.cluster_name}`: Drunk Elephant is under-indexed at {row.drunk_elephant_share_pct:.1f}% ({row.drunk_elephant_over_index_pct:+.1f} pts), suggesting this experience space is led more by competitors."
            )

    interpretation_lines += [
        "",
        "## Shortcoming summary",
        "Drunk Elephant's clearest relative gap is not in hydration, but in the clusters dominated by cleansing and lightweight finish experiences, where Tatcha is more prevalent and often associated with stronger consumer approval.",
        "Drunk Elephant is also less dominant in the large value-and-results-led segment, where The Ordinary shapes the consumer experience through stronger low-price visibility.",
        "This suggests Drunk Elephant may need to improve either perceived value-for-money or the clarity of sensory/use experience in order to compete more directly with Tatcha and The Ordinary in these experience spaces.",
    ]
    (OUT_DIR / "drunk_elephant_cluster_diagnosis.md").write_text("\n".join(interpretation_lines), encoding="utf-8")

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
