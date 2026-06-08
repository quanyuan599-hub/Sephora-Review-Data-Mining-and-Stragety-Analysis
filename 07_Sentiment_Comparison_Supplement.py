from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path.cwd()
NMF_FEATURE_PATH = PROJECT_DIR / "outputs" / "ifb214_nmf_process" / "review_level_nmf_feature_table.csv"
NMF_SUMMARY_PATH = PROJECT_DIR / "outputs" / "ifb214_nmf_process" / "nmf_topic_summary.csv"
REVIEW_DATA_PATH = PROJECT_DIR / "outputs" / "sephora_target_brand_6000_reviews.xlsx"
OUT_DIR = PROJECT_DIR / "outputs" / "ifb214_sentiment_clustering_comparison"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BRANDS = ["Drunk Elephant", "Tatcha", "The Ordinary"]
K_VALUES = [3, 4, 5, 6]
RANDOM_STATE = 42
TOPIC_COLS = [f"nmf_topic_{i}_weight" for i in range(1, 9)]
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


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = current + " " + word
        if len(trial) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return clean_text(text).split()


def sentiment_score(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0

    score = 0.0
    for phrase, weight in POSITIVE_PHRASES.items():
        score += clean_text(text).count(phrase) * weight
    for phrase, weight in NEGATIVE_PHRASES.items():
        score += clean_text(text).count(phrase) * weight

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

    normaliser = math.sqrt(len(tokens)) + 1.0
    return score / normaliser


def sentiment_label(score: float) -> str:
    if score >= 0.22:
        return "positive"
    if score <= -0.22:
        return "negative"
    return "neutral"


def scale_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        values = out[col].astype(float).to_numpy()
        mean = values.mean()
        std = values.std()
        out[col] = (values - mean) / (std if std != 0 else 1.0)
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
    labels_s = labels[idx]
    dist = np.sqrt(squared_distances(xs, xs), dtype=np.float32)
    scores = []
    unique = np.unique(labels_s)
    for i in range(len(xs)):
        own = labels_s[i]
        same_mask = labels_s == own
        same_count = int(same_mask.sum())
        a = float(dist[i, same_mask].sum() / (same_count - 1)) if same_count > 1 else 0.0
        b = min(float(dist[i, labels_s == other].mean()) for other in unique if other != own)
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(scores))


def dominant_value(series: pd.Series) -> tuple[str, float]:
    counts = series.astype(str).value_counts(dropna=False)
    return counts.index[0], float(counts.iloc[0] / counts.sum())


def centroid_separation(centers: np.ndarray) -> float:
    if len(centers) < 2:
        return 0.0
    pairwise = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            pairwise.append(float(np.sqrt(np.sum((centers[i] - centers[j]) ** 2))))
    return float(np.mean(pairwise))


def dominant_themes(mean_row: pd.Series, topic_map: dict[int, str], top_n: int = 2) -> list[tuple[str, float]]:
    scored = []
    for topic_num in range(1, 9):
        col = f"nmf_topic_{topic_num}_weight"
        scored.append((topic_map[topic_num], float(mean_row[col])))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_n]


def assign_cluster_label(theme_names: list[str], avg_rating: float, rec_rate: float, avg_sentiment: float) -> str:
    joined = " ".join(theme_names)
    if "Exfoliation & Brightening" in joined and avg_sentiment > 0.08 and rec_rate >= 80:
        return "Results-driven Brightening Satisfaction"
    if "Cleansing & Makeup Removal" in joined and avg_sentiment >= 0:
        return "Gentle Cleansing Approval"
    if "Cleansing & Makeup Removal" in joined and avg_sentiment < 0:
        return "Texture or Residue Complaints"
    if "Sensitive Relief & Barrier Repair" in joined and avg_sentiment < 0:
        return "Sensitivity or Irritation Concerns"
    if "Hydration & Moisture Barrier" in joined and rec_rate >= 80:
        return "Hydration-focused Satisfaction"
    if "Value, Size & Repurchase" in joined:
        return "Value-conscious Routine Users"
    if avg_rating >= 4.15 and rec_rate >= 80:
        return "High-satisfaction Routine Users"
    return "Mixed Product Experience"


def profile_business_interpretability(
    size_table: pd.DataFrame,
    profile_table: pd.DataFrame,
    cluster_theme_labels: list[str],
) -> tuple[int, str, str]:
    tiny_clusters = int((size_table["cluster_share_pct"] < 8).sum())
    unique_theme_count = len(set(cluster_theme_labels))
    rating_range = float(profile_table["average_rating"].max() - profile_table["average_rating"].min())
    rec_range = float(profile_table["recommendation_rate"].max() - profile_table["recommendation_rate"].min())
    avg_dominant_brand_share = float(profile_table["dominant_brand_share_pct"].mean())
    pure_brand_clusters = int((profile_table["dominant_brand_share_pct"] >= 90).sum())

    score = 0
    if tiny_clusters == 0:
        score += 2
    elif tiny_clusters == 1:
        score += 1
    if unique_theme_count >= 3:
        score += 2
    elif unique_theme_count >= 2:
        score += 1
    if rating_range >= 0.18:
        score += 1
    if rec_range >= 4.0:
        score += 1
    if avg_dominant_brand_share <= 70:
        score += 1
    if pure_brand_clusters == 0:
        score += 1

    if score >= 7:
        label = "Strong"
    elif score >= 4:
        label = "Moderate"
    else:
        label = "Weak"
    note = (
        f"{unique_theme_count} distinct dominant themes; "
        f"{tiny_clusters} tiny clusters below 8%; "
        f"rating range {rating_range:.2f}; recommendation range {rec_range:.1f} pts; "
        f"{pure_brand_clusters} brand-dominant clusters >=90%."
    )
    return score, label, note


def choose_final_k(comparison_df: pd.DataFrame) -> int:
    best_sil = float(comparison_df["approx_silhouette"].max())
    candidates = comparison_df[
        (comparison_df["approx_silhouette"] >= best_sil - 0.01)
        & (comparison_df["tiny_clusters_below_8pct"] <= 1)
    ].copy()
    if candidates.empty:
        return int(comparison_df.sort_values(["interpretability_score", "approx_silhouette", "k"], ascending=[False, False, True]).iloc[0]["k"])
    return int(
        candidates.sort_values(
            ["interpretability_score", "pure_brand_clusters_90pct", "avg_dominant_brand_share_pct", "k"],
            ascending=[False, True, True, True],
        ).iloc[0]["k"]
    )


def draw_comparison_line_plot(
    diagnostics: pd.DataFrame,
    y_col: str,
    title: str,
    path: Path,
    y_label_formatter: str = ".3f",
) -> None:
    width, height = 980, 620
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((36, 24), title, font=font(28, True), fill=(28, 28, 28))
    draw.text((36, 58), "Blue = NMF topics only; Orange = NMF topics + sentiment", font=font(15), fill=(95, 95, 95))

    left, top, right, bottom = 90, 120, 80, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min = float(diagnostics[y_col].min())
    y_max = float(diagnostics[y_col].max())
    if y_min == y_max:
        y_min -= 1
        y_max += 1

    for j in range(5):
        y = top + j * plot_h / 4
        draw.line((left, y, left + plot_w, y), fill=(232, 236, 240))
        val = y_max - j * (y_max - y_min) / 4
        label = format(val, y_label_formatter) if y_label_formatter else str(val)
        draw.text((18, y - 8), label, font=font(12), fill=(90, 90, 90))

    colors = {
        "nmf_topics_only": (41, 117, 181),
        "nmf_topics_plus_sentiment": (230, 126, 34),
    }
    for feature_set, subset in diagnostics.groupby("feature_set"):
        subset = subset.sort_values("k")
        xs = subset["k"].to_numpy(dtype=float)
        ys = subset[y_col].to_numpy(dtype=float)
        pts = []
        for x, y in zip(xs, ys):
            px = left + (x - xs.min()) / (xs.max() - xs.min()) * plot_w
            py = top + plot_h - (y - y_min) / (y_max - y_min) * plot_h
            pts.append((px, py))
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=colors[feature_set])
            draw.text((px - 6, top + plot_h + 18), str(int(x)), font=font(14, True), fill=(55, 55, 55))
        draw.line(pts, fill=colors[feature_set], width=4)

    draw.rounded_rectangle((640, 34, 930, 92), radius=10, fill=(248, 250, 252), outline=(226, 230, 234))
    draw.text((658, 46), "NMF topics only", font=font(15, True), fill=colors["nmf_topics_only"])
    draw.text((658, 68), "NMF topics + sentiment", font=font(15, True), fill=colors["nmf_topics_plus_sentiment"])
    img.save(path)


def draw_feature_set_recommendation(summary_df: pd.DataFrame, path: Path) -> None:
    width, height = 1520, 440
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((36, 24), "Feature Set Comparison for Consumer Experience Segmentation", font=font(30, True), fill=(28, 28, 28))
    draw.text((36, 60), "Best K result for each feature set", font=font(16), fill=(95, 95, 95))

    headers = ["Feature Set", "Best K", "Silhouette", "Rating Range", "Recommendation Range", "Avg Dominant Brand Share", "Interpretability"]
    x_positions = [24, 320, 450, 610, 800, 1040, 1325]
    widths = [280, 100, 130, 150, 210, 250, 160]
    top_y = 110
    header_fill = (31, 165, 157)
    for x, w, header in zip(x_positions, widths, headers):
        draw.rounded_rectangle((x, top_y, x + w, top_y + 52), radius=10, fill=header_fill)
        draw.text((x + 10, top_y + 13), header, font=font(16, True), fill="white")

    row_fills = [(239, 248, 255), (255, 247, 236)]
    for idx, row in enumerate(summary_df.itertuples(index=False)):
        y = top_y + 64 + idx * 110
        fill = row_fills[idx % len(row_fills)]
        for x, w in zip(x_positions, widths):
            draw.rounded_rectangle((x, y, x + w, y + 88), radius=10, fill=fill, outline=(228, 232, 236))
        feature_name = "NMF topics only" if row.feature_set == "nmf_topics_only" else "NMF topics + sentiment"
        draw.text((x_positions[0] + 10, y + 16), feature_name, font=font(18, True), fill=(35, 35, 35))
        draw.text((x_positions[1] + 28, y + 24), str(int(row.best_k)), font=font(24, True), fill=(35, 35, 35))
        draw.text((x_positions[2] + 14, y + 28), f"{row.approx_silhouette:.3f}", font=font(20, True), fill=(35, 35, 35))
        draw.text((x_positions[3] + 16, y + 28), f"{row.rating_range:.2f}", font=font(20, True), fill=(35, 35, 35))
        draw.text((x_positions[4] + 16, y + 28), f"{row.recommendation_range:.1f} pts", font=font(20, True), fill=(35, 35, 35))
        draw.text((x_positions[5] + 16, y + 28), f"{row.avg_dominant_brand_share_pct:.1f}%", font=font(20, True), fill=(35, 35, 35))
        draw.text((x_positions[6] + 18, y + 28), str(row.business_interpretability), font=font(20, True), fill=(35, 35, 35))
    img.save(path)


def draw_consumer_map(profile: pd.DataFrame, feature_title: str, final_k: int, path: Path) -> None:
    width, height = 1700, 200 + len(profile) * 138
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((36, 24), f"Consumer Experience Map: {feature_title} (K={final_k})", font=font(30, True), fill=(28, 28, 28))
    draw.text((36, 62), "rating and is_recommended are used only after clustering for profiling.", font=font(16), fill=(95, 95, 95))

    headers = ["Cluster", "Share", "Experience Label", "Dominant Themes", "Brand Mix", "Avg Sentiment", "Avg Rating / Rec Rate"]
    x_positions = [24, 180, 320, 650, 1030, 1290, 1460]
    widths = [140, 120, 310, 360, 240, 140, 210]
    top_y = 110
    row_h = 132
    for x, w, header in zip(x_positions, widths, headers):
        draw.rounded_rectangle((x, top_y, x + w, top_y + 50), radius=10, fill=(31, 165, 157))
        draw.text((x + 10, top_y + 12), header, font=font(17, True), fill="white")

    row_fills = [(239, 248, 255), (243, 250, 244), (255, 248, 239), (248, 242, 252), (250, 246, 238)]
    for idx, row in enumerate(profile.itertuples(index=False)):
        y = top_y + 60 + idx * row_h
        fill = row_fills[idx % len(row_fills)]
        for x, w in zip(x_positions, widths):
            draw.rounded_rectangle((x, y, x + w, y + row_h - 12), radius=12, fill=fill, outline=(228, 232, 236))
        draw.text((x_positions[0] + 12, y + 18), f"Cluster {row.cluster}", font=font(18, True), fill=(35, 35, 35))
        draw.text((x_positions[0] + 12, y + 48), f"n={int(row.cluster_size)}", font=font(14), fill=(85, 85, 85))
        draw.text((x_positions[1] + 18, y + 30), f"{row.cluster_share_pct:.1f}%", font=font(24, True), fill=(35, 35, 35))
        yy = y + 16
        for line in wrap_text(row.experience_label, 26):
            draw.text((x_positions[2] + 10, yy), line, font=font(18, True), fill=(35, 35, 35))
            yy += 24
        yy = y + 14
        for line in wrap_text(row.dominant_themes, 42):
            draw.text((x_positions[3] + 10, yy), line, font=font(15), fill=(48, 48, 48))
            yy += 20
        yy = y + 14
        for line in wrap_text(row.brand_mix, 26):
            draw.text((x_positions[4] + 10, yy), line, font=font(15), fill=(48, 48, 48))
            yy += 20
        draw.text((x_positions[5] + 18, y + 34), f"{row.average_sentiment:.2f}", font=font(22, True), fill=(35, 35, 35))
        draw.text((x_positions[6] + 10, y + 20), f"Rating: {row.average_rating:.2f}", font=font(16, True), fill=(35, 35, 35))
        draw.text((x_positions[6] + 10, y + 48), f"Recommend: {row.recommendation_rate:.1f}%", font=font(16, True), fill=(35, 35, 35))
    img.save(path)


def main() -> None:
    reviews = pd.read_excel(REVIEW_DATA_PATH, sheet_name="final_6000_reviews").copy()
    reviews = reviews[reviews["brand"].isin(TARGET_BRANDS)].copy().reset_index(drop=True)
    reviews.insert(0, "review_id", np.arange(1, len(reviews) + 1))

    nmf = pd.read_csv(NMF_FEATURE_PATH)
    topic_summary = pd.read_csv(NMF_SUMMARY_PATH).sort_values("topic").reset_index(drop=True)
    topic_map = {int(row.topic): row.theme for row in topic_summary.itertuples()}

    meta_cols = ["review_id", "brand", "product_id", "product_name", "skin_type", "price_tier", "price_usd_clean", "rating", "is_recommended"]
    merged = pd.merge(
        nmf[["review_id", "brand", "product_id", "product_name", "skin_type", "price_tier", "price_usd_clean", "review_word_count", "review_title", "review_text", *TOPIC_COLS]],
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
    merged["sentiment_label"] = merged["sentiment_score"].map(sentiment_label)

    sentiment_table = merged[["review_id", "brand", "product_id", "product_name", "review_word_count", "sentiment_score", "sentiment_label", "rating", "is_recommended"]].copy()
    sentiment_table.to_csv(TABLE_DIR / "review_level_sentiment_feature_table.csv", index=False, encoding="utf-8-sig")

    feature_sets = {
        "nmf_topics_only": TOPIC_COLS,
        "nmf_topics_plus_sentiment": [*TOPIC_COLS, "sentiment_score"],
    }

    diagnostics_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    feature_set_best_rows: list[dict[str, object]] = []

    for feature_set_name, feature_cols in feature_sets.items():
        feature_dir = TABLE_DIR / feature_set_name
        feature_dir.mkdir(parents=True, exist_ok=True)

        feature_table = merged[[*meta_cols, "review_word_count", "sentiment_score", "sentiment_label", *TOPIC_COLS]].copy()
        feature_table.to_csv(feature_dir / "review_level_feature_table.csv", index=False, encoding="utf-8-sig")

        scaled = scale_columns(merged[feature_cols], feature_cols)
        x_mat = scaled.to_numpy(dtype=np.float32)
        assignments = merged[["review_id", "brand", "product_id", "product_name", "skin_type", "price_tier", "rating", "is_recommended", "sentiment_score"]].copy()

        for k in K_VALUES:
            labels, centers, inertia = kmeans(x_mat, k, RANDOM_STATE + k)
            sil = approximate_silhouette(x_mat, labels, RANDOM_STATE + k)
            separation = centroid_separation(centers)
            cluster_col = f"cluster_k{k}"
            assignments[cluster_col] = labels

            size_table = (
                pd.DataFrame({"cluster": labels})
                .value_counts()
                .reset_index(name="cluster_size")
                .sort_values("cluster")
                .reset_index(drop=True)
            )
            size_table["cluster_share_pct"] = (size_table["cluster_size"] / len(merged) * 100).round(2)
            size_table.to_csv(feature_dir / f"cluster_size_k{k}.csv", index=False, encoding="utf-8-sig")

            mean_table = (
                pd.DataFrame(scaled, columns=feature_cols)
                .assign(cluster=labels)
                .groupby("cluster", dropna=False)[feature_cols]
                .mean()
                .reset_index()
                .sort_values("cluster")
                .reset_index(drop=True)
            )
            mean_table.to_csv(feature_dir / f"cluster_mean_features_k{k}.csv", index=False, encoding="utf-8-sig")

            brand_share = (
                pd.DataFrame({"cluster": labels, "brand": merged["brand"]})
                .groupby(["cluster", "brand"])
                .size()
                .reset_index(name="review_count")
            )
            brand_share["brand_share_pct"] = (
                brand_share.groupby("cluster")["review_count"].transform(lambda s: s / s.sum() * 100)
            ).round(2)
            brand_share.to_csv(feature_dir / f"cluster_brand_share_k{k}.csv", index=False, encoding="utf-8-sig")

            profile_rows = []
            cluster_theme_labels = []
            for cluster in sorted(np.unique(labels)):
                cluster_mask = labels == cluster
                cluster_df = merged.loc[cluster_mask].copy()
                mean_topics = cluster_df[TOPIC_COLS].mean()
                top_themes = dominant_themes(mean_topics, topic_map, top_n=2)
                theme_names = [name for name, _ in top_themes]
                cluster_theme_labels.append(theme_names[0])
                dominant_brand, dominant_brand_share = dominant_value(cluster_df["brand"])
                dominant_skin, dominant_skin_share = dominant_value(cluster_df["skin_type"])
                brand_mix_df = brand_share[brand_share["cluster"] == cluster].sort_values("brand_share_pct", ascending=False)
                brand_mix = ", ".join(f"{row.brand} {row.brand_share_pct:.1f}%" for row in brand_mix_df.itertuples(index=False))
                avg_rating = float(cluster_df["rating"].mean())
                rec_rate = float(cluster_df["is_recommended"].mean() * 100)
                avg_sentiment = float(cluster_df["sentiment_score"].mean())
                label = assign_cluster_label(theme_names, avg_rating, rec_rate, avg_sentiment)
                profile_rows.append(
                    {
                        "cluster": int(cluster),
                        "cluster_size": int(cluster_mask.sum()),
                        "cluster_share_pct": round(float(cluster_mask.mean() * 100), 2),
                        "experience_label": label,
                        "dominant_themes": "; ".join(f"{theme} ({weight:.3f})" for theme, weight in top_themes),
                        "dominant_brand": dominant_brand,
                        "dominant_brand_share_pct": round(dominant_brand_share * 100, 2),
                        "dominant_skin_type": dominant_skin,
                        "dominant_skin_type_share_pct": round(dominant_skin_share * 100, 2),
                        "brand_mix": brand_mix,
                        "average_sentiment": round(avg_sentiment, 3),
                        "average_rating": round(avg_rating, 2),
                        "recommendation_rate": round(rec_rate, 1),
                    }
                )

            profile_table = pd.DataFrame(profile_rows).sort_values("cluster").reset_index(drop=True)
            profile_table.to_csv(feature_dir / f"cluster_profile_k{k}.csv", index=False, encoding="utf-8-sig")

            rating_range = float(profile_table["average_rating"].max() - profile_table["average_rating"].min())
            rec_range = float(profile_table["recommendation_rate"].max() - profile_table["recommendation_rate"].min())
            avg_dom_brand_share = float(profile_table["dominant_brand_share_pct"].mean())
            pure_brand_clusters = int((profile_table["dominant_brand_share_pct"] >= 90).sum())
            interp_score, interp_label, interp_note = profile_business_interpretability(size_table, profile_table, cluster_theme_labels)

            diagnostics_rows.append(
                {
                    "feature_set": feature_set_name,
                    "k": k,
                    "inertia": round(inertia, 3),
                    "approx_silhouette": round(sil, 6),
                    "centroid_separation": round(separation, 4),
                }
            )
            comparison_rows.append(
                {
                    "feature_set": feature_set_name,
                    "k": k,
                    "approx_silhouette": round(sil, 6),
                    "inertia": round(inertia, 3),
                    "centroid_separation": round(separation, 4),
                    "smallest_cluster_share_pct": round(float(size_table["cluster_share_pct"].min()), 2),
                    "largest_cluster_share_pct": round(float(size_table["cluster_share_pct"].max()), 2),
                    "tiny_clusters_below_8pct": int((size_table["cluster_share_pct"] < 8).sum()),
                    "rating_range": round(rating_range, 2),
                    "recommendation_range": round(rec_range, 1),
                    "avg_dominant_brand_share_pct": round(avg_dom_brand_share, 2),
                    "pure_brand_clusters_90pct": pure_brand_clusters,
                    "interpretability_score": interp_score,
                    "business_interpretability": interp_label,
                    "summary_note": interp_note,
                }
            )

        assignments.to_csv(feature_dir / "cluster_assignments_k3_to_k6.csv", index=False, encoding="utf-8-sig")
        feature_comparison = pd.DataFrame([row for row in comparison_rows if row["feature_set"] == feature_set_name]).sort_values("k").reset_index(drop=True)
        feature_comparison.to_csv(feature_dir / "k_comparison_table.csv", index=False, encoding="utf-8-sig")
        final_k = choose_final_k(feature_comparison)
        best_row = feature_comparison[feature_comparison["k"] == final_k].iloc[0].to_dict()
        best_row["best_k"] = final_k
        feature_set_best_rows.append(best_row)

        profile_for_map = pd.read_csv(feature_dir / f"cluster_profile_k{final_k}.csv").sort_values("cluster").reset_index(drop=True)
        map_title = "NMF topics only" if feature_set_name == "nmf_topics_only" else "NMF topics + sentiment"
        draw_consumer_map(profile_for_map, map_title, final_k, FIG_DIR / f"consumer_map_{feature_set_name}.png")

    diagnostics_df = pd.DataFrame(diagnostics_rows).sort_values(["feature_set", "k"]).reset_index(drop=True)
    diagnostics_df.to_csv(TABLE_DIR / "feature_set_clustering_diagnostics.csv", index=False, encoding="utf-8-sig")
    comparison_df = pd.DataFrame(comparison_rows).sort_values(["feature_set", "k"]).reset_index(drop=True)
    comparison_df.to_csv(TABLE_DIR / "feature_set_k_comparison.csv", index=False, encoding="utf-8-sig")
    best_df = pd.DataFrame(feature_set_best_rows).sort_values("feature_set").reset_index(drop=True)
    best_df.to_csv(TABLE_DIR / "feature_set_best_k_summary.csv", index=False, encoding="utf-8-sig")

    draw_comparison_line_plot(diagnostics_df, "inertia", "Elbow Comparison: NMF Topics Only vs NMF Topics + Sentiment", FIG_DIR / "figure1_elbow_comparison.png", ".0f")
    draw_comparison_line_plot(diagnostics_df, "approx_silhouette", "Silhouette Comparison: NMF Topics Only vs NMF Topics + Sentiment", FIG_DIR / "figure2_silhouette_comparison.png", ".3f")
    draw_feature_set_recommendation(best_df, FIG_DIR / "figure3_feature_set_recommendation.png")

    nmf_best = best_df[best_df["feature_set"] == "nmf_topics_only"].iloc[0]
    sent_best = best_df[best_df["feature_set"] == "nmf_topics_plus_sentiment"].iloc[0]

    def choose_feature_set(row_a: pd.Series, row_b: pd.Series) -> tuple[str, str]:
        score_a = 0
        score_b = 0
        if float(row_b["approx_silhouette"]) > float(row_a["approx_silhouette"]) + 0.005:
            score_b += 2
        elif float(row_a["approx_silhouette"]) > float(row_b["approx_silhouette"]) + 0.005:
            score_a += 2
        else:
            score_a += 1
            score_b += 1

        if float(row_b["rating_range"]) >= float(row_a["rating_range"]) + 0.05:
            score_b += 1
        elif float(row_a["rating_range"]) >= float(row_b["rating_range"]) + 0.05:
            score_a += 1

        if float(row_b["recommendation_range"]) >= float(row_a["recommendation_range"]) + 1.0:
            score_b += 1
        elif float(row_a["recommendation_range"]) >= float(row_b["recommendation_range"]) + 1.0:
            score_a += 1

        if float(row_b["avg_dominant_brand_share_pct"]) <= float(row_a["avg_dominant_brand_share_pct"]) - 3:
            score_b += 1
        elif float(row_a["avg_dominant_brand_share_pct"]) <= float(row_b["avg_dominant_brand_share_pct"]) - 3:
            score_a += 1

        if int(row_b["interpretability_score"]) > int(row_a["interpretability_score"]):
            score_b += 2
        elif int(row_a["interpretability_score"]) > int(row_b["interpretability_score"]):
            score_a += 2

        if score_b > score_a:
            return "nmf_topics_plus_sentiment", "Adding sentiment produced more meaningful consumer experience segments."
        return "nmf_topics_only", "The sentiment feature did not improve the segmentation enough to justify the extra complexity."

    recommended_feature_set, recommendation_reason = choose_feature_set(nmf_best, sent_best)
    recommendation_table = pd.DataFrame(
        [
            {
                "recommended_feature_set": recommended_feature_set,
                "recommended_k": int(best_df.loc[best_df["feature_set"] == recommended_feature_set, "best_k"].iloc[0]),
                "reason": recommendation_reason,
            }
        ]
    )
    recommendation_table.to_csv(TABLE_DIR / "recommended_feature_set.csv", index=False, encoding="utf-8-sig")

    explanation_lines = [
        "# IFB214 clustering feature-set comparison",
        "",
        "Compared feature sets:",
        "- A: 8 NMF topic scores only",
        "- B: 8 NMF topic scores + 1 sentiment_score derived from review_title + review_text",
        "",
        "Outcomes excluded from clustering input:",
        "- rating",
        "- is_recommended",
        "",
        "Evaluation dimensions:",
        "- cluster separation: inertia, approximate silhouette, centroid separation",
        "- rating differences across clusters",
        "- recommendation differences across clusters",
        "- brand mix differences across clusters",
        "- business interpretability heuristic",
        "",
        f"Recommended feature set: {recommended_feature_set}",
        recommendation_reason,
    ]
    (OUT_DIR / "comparison_explanation.md").write_text("\n".join(explanation_lines), encoding="utf-8")

    print(best_df.to_string(index=False))
    print("\nRecommendation")
    print(recommendation_table.to_string(index=False))


if __name__ == "__main__":
    main()
