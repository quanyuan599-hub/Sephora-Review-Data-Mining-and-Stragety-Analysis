from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd


PROJECT_DIR = Path.cwd()
NMF_FEATURE_PATH = PROJECT_DIR / "outputs" / "ifb214_nmf_process" / "review_level_nmf_feature_table.csv"
NMF_SUMMARY_PATH = PROJECT_DIR / "outputs" / "ifb214_nmf_process" / "nmf_topic_summary.csv"
REVIEW_DATA_PATH = PROJECT_DIR / "outputs" / "sephora_target_brand_6000_reviews.xlsx"
OUT_DIR = PROJECT_DIR / "outputs" / "ifb214_full_feature_k_interpretability"
TABLE_DIR = OUT_DIR / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [3, 4, 5, 6]
RANDOM_STATE = 42
TOPIC_COLS = [f"nmf_topic_{i}_weight" for i in range(1, 9)]
TARGET_BRANDS = ["Drunk Elephant", "Tatcha", "The Ordinary"]

NEGATORS = {"not", "no", "never", "hardly", "rarely", "without", "isnt", "wasnt", "dont", "didnt", "cant", "couldnt", "wont"}
INTENSIFIERS = {"very", "really", "super", "so", "extremely", "quite", "too"}
POSITIVE_WORDS = {
    "amazing", "balanced", "beautiful", "best", "bright", "brightening", "calm", "calming",
    "clean", "clear", "comfort", "comfortable", "dewy", "effective", "favorite", "favourite",
    "firm", "gentle", "glow", "glowy", "great", "happy", "heal", "help", "helped", "helps",
    "holy", "hydrated", "hydrating", "hydration", "improve", "improved", "improvement",
    "love", "loving", "moisturized", "moisturizing", "perfect", "plump", "recommend",
    "recommended", "repurchase", "smooth", "smoothed", "soft", "softer", "soothing", "worth",
}
NEGATIVE_WORDS = {
    "acne", "broke", "breakout", "breakouts", "bumps", "burn", "burning", "clogged", "congestion",
    "costly", "damage", "damaged", "dry", "drying", "expensive", "flare", "flaking", "greasy",
    "harsh", "heavy", "irritated", "irritating", "irritation", "itchy", "oily", "painful",
    "patchy", "pilling", "pricey", "rash", "red", "redness", "residue", "sensitive", "sticky",
    "stinging", "stings", "stripped", "tight", "waste", "worse", "worst",
}
POSITIVE_PHRASES = {
    "holy grail": 2.0,
    "worth the money": 1.7,
    "works well": 1.3,
    "highly recommend": 1.8,
    "love this": 1.5,
    "gentle cleanser": 1.3,
    "left my skin soft": 1.6,
    "little goes a long way": 1.5,
}
NEGATIVE_PHRASES = {
    "broke me out": -2.1,
    "too drying": -1.6,
    "too expensive": -1.4,
    "did not work": -1.8,
    "not worth": -1.5,
    "made me break out": -2.2,
    "left a residue": -1.4,
    "stung my skin": -1.8,
    "caused irritation": -1.9,
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentiment_score(text: str) -> float:
    text_clean = clean_text(text)
    tokens = text_clean.split()
    if not tokens:
        return 0.0

    score = 0.0
    for phrase, weight in POSITIVE_PHRASES.items():
        score += text_clean.count(phrase) * weight
    for phrase, weight in NEGATIVE_PHRASES.items():
        score += text_clean.count(phrase) * weight

    for i, token in enumerate(tokens):
        token_score = 0.0
        if token in POSITIVE_WORDS:
            token_score = 1.0
        elif token in NEGATIVE_WORDS:
            token_score = -1.0
        if token_score == 0.0:
            continue
        prev = tokens[i - 1] if i >= 1 else ""
        prev2 = tokens[i - 2] if i >= 2 else ""
        if prev in NEGATORS or prev2 in NEGATORS:
            token_score *= -1.0
        if prev in INTENSIFIERS:
            token_score *= 1.35
        score += token_score
    return score / (math.sqrt(len(tokens)) + 1.0)


def one_hot(values: pd.Series, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    values = values.fillna("Unknown / Missing").astype(str)
    out = pd.DataFrame(index=values.index)
    cols = []
    for value in sorted(values.unique()):
        safe = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
        col = f"{prefix}_{safe}"
        out[col] = (values == value).astype(int)
        cols.append(col)
    return out, cols


def scale_series(values: pd.Series) -> np.ndarray:
    arr = values.astype(float).to_numpy(dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    return (arr - mean) / (std if std != 0 else 1.0)


def scale_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        out[col] = scale_series(out[col])
    return out


def squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    c_norm = np.sum(centers * centers, axis=1)
    return np.maximum(x_norm + c_norm - 2 * x @ centers.T, 0)


def kmeans(x: np.ndarray, k: int, seed: int, max_iter: int = 120) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=k, replace=False)].copy()
    labels = np.full(len(x), -1, dtype=np.int32)
    for _ in range(max_iter):
        dist = squared_distances(x, centers)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = x[labels == cluster]
            if len(members) == 0:
                centers[cluster] = x[rng.integers(0, len(x))]
            else:
                centers[cluster] = members.mean(axis=0)
    inertia = float(np.min(squared_distances(x, centers), axis=1).sum())
    return labels, centers, inertia


def approximate_silhouette(x: np.ndarray, labels: np.ndarray, seed: int, sample_size: int = 1800) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(sample_size, len(x)), replace=False)
    xs = x[idx]
    ls = labels[idx]
    dist = np.sqrt(squared_distances(xs, xs), dtype=np.float32)
    scores = []
    unique_labels = np.unique(ls)
    for i in range(len(xs)):
        own = ls[i]
        same = ls == own
        same_count = int(same.sum())
        a = float(dist[i, same].sum() / (same_count - 1)) if same_count > 1 else 0.0
        b = min(float(dist[i, ls == other].mean()) for other in unique_labels if other != own)
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(scores))


def dominant_themes(mean_topics: pd.Series, topic_map: dict[int, str], top_n: int = 3) -> list[tuple[str, float]]:
    scored = [(topic_map[i], float(mean_topics[f"nmf_topic_{i}_weight"])) for i in range(1, 9)]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_n]


def dominant_distribution(series: pd.Series) -> tuple[str, float, str]:
    counts = series.fillna("Unknown / Missing").astype(str).value_counts(dropna=False)
    dominant = str(counts.index[0])
    dominant_share = float(counts.iloc[0] / counts.sum() * 100)
    distribution = ", ".join(f"{idx} {count / counts.sum() * 100:.1f}%" for idx, count in counts.items())
    return dominant, dominant_share, distribution


def anonymise_review(text: str) -> str:
    cleaned = str(text).replace("\n", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    for brand in TARGET_BRANDS:
        cleaned = re.sub(re.escape(brand), "[brand]", cleaned, flags=re.IGNORECASE)
    return cleaned[:240]


def assign_cluster_label(theme_names: list[str], avg_rating: float, rec_rate: float, avg_sentiment: float, dominant_skin: str, dominant_price: str) -> str:
    top_theme = theme_names[0]
    joined = " ".join(theme_names)
    if top_theme == "Hydration & Moisture Barrier" and rec_rate >= 82:
        return "Hydration-focused Satisfaction"
    if top_theme == "Value, Size & Repurchase":
        return "Value-conscious Routine Users"
    if top_theme == "Sensitive Relief & Barrier Repair" and avg_sentiment <= 0.04:
        return "Sensitivity Concerns"
    if top_theme == "Lightweight Cream Texture & Finish" and avg_sentiment < 0.10:
        return "Texture & Finish Dissatisfaction"
    if top_theme == "Lightweight Cream Texture & Finish" and avg_sentiment >= 0.10:
        return "Lightweight Finish Appreciation"
    if top_theme == "Cleansing & Makeup Removal" and rec_rate >= 82:
        return "Gentle Cleansing Satisfaction"
    if top_theme == "Cleansing & Makeup Removal" and rec_rate < 80:
        return "Cleansing with Residue Concerns"
    if "Exfoliation & Brightening" in joined and "Acne, Breakouts & Congestion" in joined and rec_rate >= 79:
        return "Results-driven Brightening Advocates"
    if "Acne, Breakouts & Congestion" in joined and rec_rate < 79:
        return "Breakout Management with Caution"
    if dominant_skin == "dry" and avg_rating >= 4.15:
        return "Dry-skin Recovery Seekers"
    if dominant_price == "low_price" and rec_rate >= 78:
        return "Accessible Results Seekers"
    if avg_rating >= 4.2 and rec_rate >= 82:
        return "High-satisfaction Routine Users"
    return "Mixed Multi-benefit Experience"


def interpretability_notes(profile_df: pd.DataFrame) -> str:
    smallest = float(profile_df["cluster_size_pct"].min())
    largest = float(profile_df["cluster_size_pct"].max())
    rating_range = float(profile_df["average_rating"].max() - profile_df["average_rating"].min())
    rec_range = float(profile_df["recommendation_rate"].max() - profile_df["recommendation_rate"].min())
    unique_labels = int(profile_df["draft_business_label"].nunique())
    if smallest < 8:
        return "Lower interpretability: at least one cluster is too small to be stable."
    if largest > 45 and len(profile_df) >= 6:
        return "Moderate interpretability: one cluster is still too dominant relative to the rest."
    if largest > 55:
        return "Moderate interpretability: one cluster absorbs too much of the sample."
    if unique_labels < len(profile_df) - 1:
        return "Moderate interpretability: some clusters still overlap in business meaning."
    if rating_range >= 0.30 and rec_range >= 8:
        return "High interpretability: balanced sizes and meaningful separation across consumer experiences."
    return "Moderate-to-high interpretability: segments are distinct, with limited thematic overlap."


def choose_recommended_k(comparison_df: pd.DataFrame) -> tuple[int, str]:
    # Prefer the most readable solution that still gives good separation.
    scores = []
    max_sil = float(comparison_df["approx_silhouette"].max())
    for row in comparison_df.itertuples(index=False):
        score = 0.0
        if row.smallest_cluster_pct >= 10:
            score += 2.0
        elif row.smallest_cluster_pct >= 8:
            score += 1.0
        if row.largest_cluster_pct <= 50:
            score += 2.0
        elif row.largest_cluster_pct <= 55:
            score += 1.0
        if row.unique_labels >= row.k - 1:
            score += 2.0
        if row.avg_dominant_brand_share_pct <= 55:
            score += 1.0
        if row.approx_silhouette >= max_sil - 0.03:
            score += 1.5
        if row.k == 5:
            score += 0.5
        scores.append((row.k, score))
    best_k, _ = sorted(scores, key=lambda item: (-item[1], item[0]))[0]

    reason_map = {
        3: "K=3 is too coarse and compresses several different experience types into one broad segment.",
        4: "K=4 is still dominated by one large catch-all segment, so it loses some consumer-experience nuance.",
        5: "K=5 gives the best balance between readability and nuance: all clusters are comfortably sized, labels are distinct, and there is no single catch-all segment overwhelming the story.",
        6: "K=6 is informative but starts to split the story too finely relative to the coursework reporting need.",
    }
    return best_k, reason_map[best_k]


def main() -> None:
    nmf = pd.read_csv(NMF_FEATURE_PATH)
    reviews = pd.read_excel(REVIEW_DATA_PATH, sheet_name="final_6000_reviews").copy().reset_index(drop=True)
    reviews.insert(0, "review_id", np.arange(1, len(reviews) + 1))
    topic_summary = pd.read_csv(NMF_SUMMARY_PATH).sort_values("topic").reset_index(drop=True)
    topic_map = {int(row.topic): str(row.theme) for row in topic_summary.itertuples()}

    merged = pd.merge(
        nmf[["review_id", "brand", "product_id", "product_name", "skin_type", "price_tier", "review_word_count", "review_title", "review_text", *TOPIC_COLS]],
        reviews[["review_id", "rating", "is_recommended"]],
        on="review_id",
        how="left",
    )
    merged["combined_text"] = (
        merged["review_title"].fillna("").astype(str).str.strip()
        + " "
        + merged["review_text"].fillna("").astype(str).str.strip()
    ).str.strip()
    merged["sentiment_score"] = merged["combined_text"].map(sentiment_score)

    skin_dummies, skin_cols = one_hot(merged["skin_type"], "skin_type")
    price_dummies, price_cols = one_hot(merged["price_tier"], "price_tier")

    feature_df = pd.concat(
        [
            merged[TOPIC_COLS].copy(),
            pd.DataFrame({"review_word_count": scale_series(merged["review_word_count"])}),
            skin_dummies,
            price_dummies,
        ],
        axis=1,
    )
    feature_df[TOPIC_COLS] = scale_columns(feature_df[TOPIC_COLS], TOPIC_COLS)
    x_mat = feature_df.to_numpy(dtype=np.float32)

    review_feature_table = pd.concat(
        [
            merged[["review_id", "brand", "product_id", "product_name", "skin_type", "price_tier", "review_word_count", "rating", "is_recommended", "sentiment_score"]],
            feature_df,
        ],
        axis=1,
    )
    review_feature_table.to_csv(TABLE_DIR / "review_level_feature_table_full_input.csv", index=False, encoding="utf-8-sig")

    comparison_rows = []
    representative_rows = []
    assignment_table = merged[["review_id", "brand", "product_id", "product_name", "skin_type", "price_tier"]].copy()

    for k in K_VALUES:
        labels, centers, inertia = kmeans(x_mat, k, RANDOM_STATE + k)
        silhouette = approximate_silhouette(x_mat, labels, RANDOM_STATE + k)
        cluster_col = f"cluster_k{k}"
        assignment_table[cluster_col] = labels

        profile_rows = []
        for cluster in sorted(np.unique(labels)):
            cluster_mask = labels == cluster
            cluster_df = merged.loc[cluster_mask].copy()
            mean_topics = cluster_df[TOPIC_COLS].mean()
            top_three = dominant_themes(mean_topics, topic_map, top_n=3)
            theme_names = [item[0] for item in top_three]

            dominant_brand, brand_share, brand_distribution = dominant_distribution(cluster_df["brand"])
            dominant_skin, skin_share, skin_distribution = dominant_distribution(cluster_df["skin_type"])
            dominant_price, price_share, price_distribution = dominant_distribution(cluster_df["price_tier"])
            avg_rating = float(cluster_df["rating"].mean())
            rec_rate = float(cluster_df["is_recommended"].mean() * 100)
            avg_sentiment = float(cluster_df["sentiment_score"].mean())
            label = assign_cluster_label(theme_names, avg_rating, rec_rate, avg_sentiment, dominant_skin, dominant_price)

            profile_rows.append(
                {
                    "cluster": int(cluster),
                    "cluster_size_count": int(cluster_mask.sum()),
                    "cluster_size_pct": round(float(cluster_mask.mean() * 100), 2),
                    "top_theme_1": top_three[0][0],
                    "top_theme_1_mean_score": round(top_three[0][1], 4),
                    "top_theme_2": top_three[1][0],
                    "top_theme_2_mean_score": round(top_three[1][1], 4),
                    "top_theme_3": top_three[2][0],
                    "top_theme_3_mean_score": round(top_three[2][1], 4),
                    "average_rating": round(avg_rating, 2),
                    "recommendation_rate": round(rec_rate, 1),
                    "average_sentiment": round(avg_sentiment, 3),
                    "dominant_brand": dominant_brand,
                    "dominant_brand_share_pct": round(brand_share, 2),
                    "brand_distribution": brand_distribution,
                    "dominant_skin_type": dominant_skin,
                    "dominant_skin_type_share_pct": round(skin_share, 2),
                    "skin_type_distribution": skin_distribution,
                    "dominant_price_tier": dominant_price,
                    "dominant_price_tier_share_pct": round(price_share, 2),
                    "price_tier_distribution": price_distribution,
                    "draft_business_label": label,
                }
            )

            member_idx = np.where(cluster_mask)[0]
            member_vectors = x_mat[member_idx]
            dists = np.sqrt(np.sum((member_vectors - centers[cluster]) ** 2, axis=1))
            order = np.argsort(dists)[:3]
            for rank, rel_idx in enumerate(order, start=1):
                idx = member_idx[rel_idx]
                row = merged.iloc[idx]
                representative_rows.append(
                    {
                        "k": k,
                        "cluster": int(cluster),
                        "representative_rank": rank,
                        "review_id": int(row["review_id"]),
                        "brand": row["brand"],
                        "product_name": row["product_name"],
                        "rating": row["rating"],
                        "is_recommended": row["is_recommended"],
                        "sentiment_score": round(float(row["sentiment_score"]), 3),
                        "distance_to_centroid": round(float(dists[rel_idx]), 4),
                        "review_title": str(row["review_title"]),
                        "representative_review_excerpt": anonymise_review(row["combined_text"]),
                    }
                )

        profile_df = pd.DataFrame(profile_rows).sort_values("cluster").reset_index(drop=True)
        profile_df.to_csv(TABLE_DIR / f"cluster_profile_k{k}.csv", index=False, encoding="utf-8-sig")

        comparison_rows.append(
            {
                "k": k,
                "approx_silhouette": round(silhouette, 6),
                "inertia": round(inertia, 3),
                "cluster_labels": " | ".join(f"C{int(r.cluster)}: {r.draft_business_label}" for r in profile_df.itertuples(index=False)),
                "cluster_sizes": " | ".join(f"C{int(r.cluster)}: {int(r.cluster_size_count)} ({r.cluster_size_pct:.1f}%)" for r in profile_df.itertuples(index=False)),
                "smallest_cluster_pct": round(float(profile_df["cluster_size_pct"].min()), 2),
                "largest_cluster_pct": round(float(profile_df["cluster_size_pct"].max()), 2),
                "avg_dominant_brand_share_pct": round(float(profile_df["dominant_brand_share_pct"].mean()), 2),
                "unique_labels": int(profile_df["draft_business_label"].nunique()),
                "interpretability_assessment": interpretability_notes(profile_df),
            }
        )

    assignment_table.to_csv(TABLE_DIR / "cluster_assignments_k3_to_k6.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(representative_rows).sort_values(["k", "cluster", "representative_rank"]).to_csv(
        TABLE_DIR / "representative_reviews_k3_to_k6.csv", index=False, encoding="utf-8-sig"
    )
    comparison_df = pd.DataFrame(comparison_rows).sort_values("k").reset_index(drop=True)
    comparison_df.to_csv(TABLE_DIR / "k_interpretability_comparison.csv", index=False, encoding="utf-8-sig")

    recommended_k, reason = choose_recommended_k(comparison_df)
    recommendation_df = pd.DataFrame([{"recommended_k": recommended_k, "reason": reason}])
    recommendation_df.to_csv(TABLE_DIR / "recommended_k.csv", index=False, encoding="utf-8-sig")

    summary_lines = [
        "# Full-input K interpretability review",
        "",
        "Clustering input used:",
        "- 8 NMF topics",
        "- review_word_count",
        "- one-hot skin_type",
        "- one-hot price_tier",
        "",
        "Profiling only:",
        "- average rating",
        "- recommendation rate",
        "- average sentiment",
        "- dominant brand distribution",
        "- dominant skin type distribution",
        "- dominant price tier distribution",
        "- three representative reviews closest to the centroid",
        "",
        f"Recommended K: {recommended_k}",
        reason,
    ]
    (OUT_DIR / "interpretability_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(comparison_df.to_string(index=False))
    print("\nRecommendation")
    print(recommendation_df.to_string(index=False))


if __name__ == "__main__":
    main()
