from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path.cwd()
DATA_PATH = PROJECT_DIR / "outputs" / "sephora_target_brand_6000_reviews.xlsx"
OUT_DIR = PROJECT_DIR / "outputs" / "ifb214_nmf_process"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BRANDS = ["Drunk Elephant", "Tatcha", "The Ordinary"]
N_TOPICS = 8
MAX_FEATURES = 220
TOP_TERMS_PER_TOPIC = 12
MAX_ITER = 350
RANDOM_STATE = 42

BASE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "had", "has", "have",
    "he", "her", "his", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "she",
    "so", "that", "the", "their", "them", "then", "there", "these", "this", "to", "was", "we", "were",
    "with", "you", "your", "about", "after", "all", "also", "am", "because", "before", "can", "did",
    "do", "does", "don", "get", "got", "just", "like", "more", "most", "not", "now", "one", "only",
    "other", "really", "than", "they", "use", "used", "using", "very", "when", "will", "would",
    "product", "products", "skin", "face", "sephora",
}

REFINED_EXTRA_STOPWORDS = {
    "love", "great", "good", "amazing", "out", "feel", "feels", "much", "little",
    "definitely", "best", "nice", "better", "works", "work", "first", "day", "time",
    "try", "tried", "made", "make", "using", "use", "used",
}

SANITY_CHECK_STOPWORDS = {
    "doesn", "didn", "well", "even", "too", "makes", "make", "night", "recommend",
    "recommended", "recommending", "worth", "thing", "things", "still", "also",
    "how", "any", "every", "super",
}

STOPWORDS = BASE_STOPWORDS | REFINED_EXTRA_STOPWORDS | SANITY_CHECK_STOPWORDS


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    words = [w for w in clean_text(text).split() if len(w) >= 3 and w not in STOPWORDS]
    bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
    return words + bigrams


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


def one_hot(values: pd.Series, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    values = values.fillna("Unknown / Missing").astype(str)
    unique_values = sorted(values.unique())
    out = pd.DataFrame(index=values.index)
    cols = []
    for value in unique_values:
        safe = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
        col = f"{prefix}_{safe}"
        out[col] = (values == value).astype(int)
        cols.append(col)
    return out, cols


def tier_price(value: float, q1: float, q2: float) -> str:
    if value <= q1:
        return "low_price"
    if value <= q2:
        return "mid_price"
    return "high_price"


def anonymise_excerpt(text: str, brands: list[str]) -> str:
    excerpt = str(text).replace("\n", " ").strip()
    excerpt = re.sub(r"\s+", " ", excerpt)
    for brand in brands:
        excerpt = re.sub(re.escape(brand), "[brand]", excerpt, flags=re.IGNORECASE)
    return excerpt[:180]


def label_topic(top_terms: list[str]) -> tuple[str, str]:
    joined = " ".join(top_terms)
    if any(term in joined for term in ["acid", "bright", "radiance", "dark spots", "difference", "pores", "serum"]):
        return (
            "Exfoliation & Brightening",
            "Reviews mentioning resurfacing, brightening, acids, pores, visible texture change, or improvement over time.",
        )
    if any(term in joined for term in ["cleanser", "clean", "wash", "makeup", "gentle", "leaves"]):
        return (
            "Cleansing & Makeup Removal",
            "Reviews focusing on cleanser feel, makeup removal, residue, softness after washing, and whether skin feels stripped.",
        )
    if any(term in joined for term in ["cream", "water", "goes long", "long way", "way"]):
        return (
            "Lightweight Cream Texture & Finish",
            "Reviews evaluating cream weight, spreadability, finish, and how the product wears across the day.",
        )
    if any(term in joined for term in ["price", "value", "size", "worth", "repurchase"]):
        return (
            "Value, Size & Repurchase",
            "Reviews discussing price, value for money, product amount, and willingness to repurchase.",
        )
    if any(term in joined for term in ["acne", "breakout", "bumps", "congestion"]):
        return (
            "Acne, Breakouts & Congestion",
            "Reviews describing acne-prone skin, clogged pores, bumps, blemishes, or breakout-related reactions.",
        )
    if any(term in joined for term in ["texture", "smooth", "glow", "pores", "soft"]):
        return (
            "Texture, Pores & Smoothness",
            "Reviews connecting products with pore appearance, smoothness, glow, and visible skin texture.",
        )
    if any(term in joined for term in ["dry", "hydrating", "hydrated", "moisturizing", "winter", "barrier"]):
        return (
            "Hydration & Moisture Barrier",
            "Reviews about moisturising performance, dryness relief, comfort, and moisture-barrier support.",
        )
    if any(term in joined for term in ["oily", "light", "greasy", "perfect", "combination"]):
        return (
            "Balance for Oily & Combination Skin",
            "Reviews discussing lightweight hydration, greasiness, oily skin compatibility, and balance for combination skin.",
        )
    return (
        "General Product Experience",
        "Reviews reflecting a mixed product experience without one single dominant skincare function.",
    )


def refine_topic_labels(topic_summary: pd.DataFrame) -> pd.DataFrame:
    topic_summary = topic_summary.copy()
    curated = {
        1: (
            "Value, Size & Repurchase",
            "Reviews discussing price, value for money, product amount, and willingness to repurchase.",
        ),
        2: (
            "Acne, Breakouts & Congestion",
            "Reviews describing acne-prone skin, blemishes, clogged pores, sensitivity, or breakout-related reactions.",
        ),
        3: (
            "Cleansing & Makeup Removal",
            "Reviews focusing on cleanser feel, makeup removal, residue, softness after washing, and whether skin feels stripped.",
        ),
        4: (
            "Sensitive Relief & Barrier Repair",
            "Reviews about dry or reactive skin, winter dryness, comfort, calming performance, and barrier support.",
        ),
        5: (
            "Exfoliation & Brightening",
            "Reviews mentioning resurfacing, brightening, acids, pores, visible texture change, or improvement over time.",
        ),
        6: (
            "Usage Amount & Product Longevity",
            "Reviews discussing how far the product goes, how much is needed, and whether a small amount lasts a long time.",
        ),
        7: (
            "Lightweight Cream Texture & Finish",
            "Reviews evaluating cream texture, spreadability, finish, and how the product wears across the day.",
        ),
        8: (
            "Hydration & Moisture Barrier",
            "Reviews about moisturising performance, comfort, lightweight hydration, and moisture-barrier support for oily or combination skin.",
        ),
    }
    for topic, (theme, definition) in curated.items():
        topic_summary.loc[topic_summary["topic"] == topic, "theme"] = theme
        topic_summary.loc[topic_summary["topic"] == topic, "definition"] = definition
    return topic_summary


def draw_topic_overview(topic_table: pd.DataFrame, path: Path) -> None:
    w, h = 1820, 1120
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((36, 24), "NMF Topic Overview for Review-Level Clustering", font=font(30, True), fill=(30, 30, 30))
    d.text((36, 62), "Themes learned from 6,000 Sephora skincare reviews; these 8 topic weights are used as clustering input features.", font=font(16), fill=(90, 90, 90))

    headers = ["Theme", "Share", "Definition", "Representative Keywords", "Anonymised Representative Excerpt"]
    x_positions = [24, 355, 520, 930, 1385]
    widths = [320, 150, 400, 430, 390]
    row_h = 118
    top_y = 110
    header_fill = (24, 167, 163)
    row_fills = [
        (238, 250, 248), (237, 245, 252), (239, 249, 244), (255, 248, 237),
        (253, 240, 240), (244, 240, 251), (237, 245, 255), (247, 241, 236),
    ]

    for x, width, header in zip(x_positions, widths, headers):
        d.rounded_rectangle((x, top_y, x + width, top_y + 52), radius=10, fill=header_fill)
        d.text((x + 12, top_y + 13), header, font=font(18, True), fill="white")

    for idx, row in enumerate(topic_table.itertuples(index=False), start=1):
        y = top_y + 60 + (idx - 1) * row_h
        fill = row_fills[(idx - 1) % len(row_fills)]
        for x, width in zip(x_positions, widths):
            d.rounded_rectangle((x, y, x + width, y + row_h - 12), radius=12, fill=fill, outline=(228, 232, 236))

        theme_lines = wrap_text(f"{row.theme} (Topic {row.topic})", 22)
        yy = y + 12
        for line in theme_lines:
            d.text((x_positions[0] + 12, yy), line, font=font(18, True), fill=(34, 34, 34))
            yy += 24
        d.text((x_positions[0] + 12, yy + 4), f"n = {int(row.dominant_review_count):,}", font=font(14), fill=(95, 95, 95))

        d.text((x_positions[1] + 18, y + 28), f"{row.dominant_review_share_pct:.1f}%", font=font(24, True), fill=(35, 35, 35))

        yy = y + 12
        for line in wrap_text(row.definition, 42):
            d.text((x_positions[2] + 12, yy), line, font=font(15), fill=(45, 45, 45))
            yy += 20

        yy = y + 12
        kws = [kw.strip() for kw in str(row.top_terms).split(",")]
        for kw in kws[:8]:
            d.rounded_rectangle((x_positions[3] + 12, yy, x_positions[3] + 12 + min(170, 26 + 8 * len(kw)), yy + 24), radius=8, fill="white", outline=(200, 210, 220))
            d.text((x_positions[3] + 22, yy + 4), kw, font=font(13), fill=(55, 55, 55))
            yy += 30

        yy = y + 12
        for line in wrap_text(str(row.anonymised_representative_excerpt), 36):
            d.text((x_positions[4] + 12, yy), line, font=font(15), fill=(50, 50, 50))
            yy += 20

    img.save(path)


def build_tfidf(texts: pd.Series, max_features: int) -> tuple[np.ndarray, list[str], dict[str, int], dict[str, int]]:
    tokenized_docs = [tokenize(text) for text in texts]
    doc_freq: Counter[str] = Counter()
    corpus_freq: Counter[str] = Counter()

    for tokens in tokenized_docs:
        counts = Counter(tokens)
        doc_freq.update(counts.keys())
        corpus_freq.update(counts)

    n_docs = len(tokenized_docs)
    candidate_terms = [term for term, df in doc_freq.items() if df >= 5 and not term.isdigit()]
    candidate_terms.sort(key=lambda t: (doc_freq[t], corpus_freq[t], len(t)), reverse=True)
    vocab = candidate_terms[:max_features]
    vocab_index = {term: idx for idx, term in enumerate(vocab)}

    X = np.zeros((n_docs, len(vocab)), dtype=np.float32)
    idf_lookup = {term: math.log((1 + n_docs) / (1 + doc_freq[term])) + 1 for term in vocab}

    for row_idx, tokens in enumerate(tokenized_docs):
        counts = Counter(token for token in tokens if token in vocab_index)
        if not counts:
            continue
        max_tf = max(counts.values())
        for term, count in counts.items():
            X[row_idx, vocab_index[term]] = (count / max_tf) * idf_lookup[term]

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms == 0, 1, norms)
    return X, vocab, dict(doc_freq), dict(corpus_freq)


def nmf_factorize(X: np.ndarray, n_topics: int, random_state: int, max_iter: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(random_state)
    n_docs, n_terms = X.shape
    W = rng.random((n_docs, n_topics), dtype=np.float32) + 1e-4
    H = rng.random((n_topics, n_terms), dtype=np.float32) + 1e-4
    errors: list[float] = []
    eps = 1e-8

    for _ in range(max_iter):
        WH = W @ H
        H *= (W.T @ X) / np.maximum(W.T @ WH, eps)
        WH = W @ H
        W *= (X @ H.T) / np.maximum(WH @ H.T, eps)
        WH = W @ H
        errors.append(float(np.linalg.norm(X - WH)))

    return W, H, errors


def normalise_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / np.where(row_sums == 0, 1, row_sums)


def main() -> None:
    reviews = pd.read_excel(DATA_PATH, sheet_name="final_6000_reviews")
    workbook_rows = len(reviews)
    reviews = reviews[reviews["brand"].isin(TARGET_BRANDS)].copy()
    reviews["review_title"] = reviews["review_title"].fillna("").astype(str)
    reviews["review_text"] = reviews["review_text"].fillna("").astype(str)
    reviews["combined_text"] = (
        reviews["review_title"].str.strip() + " " + reviews["review_text"].str.strip()
    ).str.strip()
    reviews = reviews[reviews["combined_text"].str.len() > 0].reset_index(drop=True)
    reviews["review_word_count"] = reviews["combined_text"].map(clean_text).str.split().map(len).astype(int)
    reviews["skin_type"] = reviews["skin_type"].fillna("Unknown / Missing").astype(str).replace({"": "Unknown / Missing", "Unknown": "Unknown / Missing"})
    reviews["price_usd_clean"] = pd.to_numeric(reviews["price_usd_clean"], errors="coerce")
    reviews["price_usd_clean"] = reviews["price_usd_clean"].fillna(reviews["price_usd_clean"].median())
    q1, q2 = reviews["price_usd_clean"].quantile([0.33, 0.67]).to_list()
    reviews["price_tier"] = reviews["price_usd_clean"].map(lambda x: tier_price(float(x), float(q1), float(q2)))

    X, vocab, doc_freq, corpus_freq = build_tfidf(reviews["combined_text"], MAX_FEATURES)
    W, H, errors = nmf_factorize(X, N_TOPICS, RANDOM_STATE, MAX_ITER)
    W_norm = normalise_rows(W)
    dominant_topics = np.argmax(W_norm, axis=1) + 1

    # Review-level NMF feature table: one review per row, one topic-weight column per topic.
    review_feature_table = reviews[
        [
            "brand",
            "product_id",
            "product_name",
            "skin_type",
            "price_tier",
            "price_usd_clean",
            "review_word_count",
            "review_title",
            "review_text",
        ]
    ].copy()
    review_feature_table.insert(0, "review_id", np.arange(1, len(review_feature_table) + 1))
    for topic_idx in range(N_TOPICS):
        review_feature_table[f"nmf_topic_{topic_idx + 1}_weight"] = W_norm[:, topic_idx]
    review_feature_table["dominant_topic"] = dominant_topics
    review_feature_table.to_csv(OUT_DIR / "review_level_nmf_feature_table.csv", index=False, encoding="utf-8-sig")

    # Topic-term feature table: each topic's strongest vocabulary terms and loadings.
    topic_rows = []
    for topic_idx in range(N_TOPICS):
        term_indices = np.argsort(H[topic_idx])[::-1][:TOP_TERMS_PER_TOPIC]
        for rank, term_idx in enumerate(term_indices, start=1):
            term = vocab[term_idx]
            topic_rows.append(
                {
                    "topic": topic_idx + 1,
                    "term_rank_within_topic": rank,
                    "term": term,
                    "term_loading": round(float(H[topic_idx, term_idx]), 6),
                    "document_frequency": doc_freq[term],
                    "document_coverage_pct": round(doc_freq[term] / len(reviews) * 100, 2),
                    "corpus_term_count": corpus_freq[term],
                }
            )
    topic_feature_table = pd.DataFrame(topic_rows)
    topic_feature_table.to_csv(OUT_DIR / "nmf_topic_term_feature_table.csv", index=False, encoding="utf-8-sig")

    # Topic summary with theme label, explanation, and representative excerpt.
    summary_rows = []
    for topic_idx in range(1, N_TOPICS + 1):
        mask = dominant_topics == topic_idx
        topic_reviews = reviews.loc[mask]
        top_terms = topic_feature_table[topic_feature_table["topic"] == topic_idx]["term"].head(8).tolist()
        theme, definition = label_topic(top_terms)
        excerpt = ""
        if not topic_reviews.empty:
            best_idx = int(np.argmax(W_norm[mask, topic_idx - 1]))
            excerpt = anonymise_excerpt(topic_reviews.iloc[best_idx]["review_text"], TARGET_BRANDS)
        summary_rows.append(
            {
                "topic": topic_idx,
                "theme": theme,
                "dominant_review_count": int(mask.sum()),
                "dominant_review_share_pct": round(mask.mean() * 100, 2),
                "average_topic_weight_among_dominant_reviews": round(float(W_norm[mask, topic_idx - 1].mean()) if mask.any() else 0.0, 6),
                "dominant_brand": topic_reviews["brand"].value_counts().index[0] if not topic_reviews.empty else "",
                "dominant_skin_type": topic_reviews["skin_type"].fillna("Unknown / Missing").astype(str).value_counts().index[0] if not topic_reviews.empty else "",
                "definition": definition,
                "top_terms": ", ".join(top_terms),
                "anonymised_representative_excerpt": excerpt,
            }
        )
    topic_summary = pd.DataFrame(summary_rows)
    topic_summary = refine_topic_labels(topic_summary)
    topic_summary.to_csv(OUT_DIR / "nmf_topic_summary.csv", index=False, encoding="utf-8-sig")

    # Review-level clustering input feature table.
    skin_onehot, skin_cols = one_hot(reviews["skin_type"], "skin_type")
    brand_onehot, brand_cols = one_hot(reviews["brand"], "brand_name")
    tier_onehot, tier_cols = one_hot(reviews["price_tier"], "price_tier")
    clustering_input = pd.DataFrame(
        {
            "review_id": np.arange(1, len(reviews) + 1),
            "brand": reviews["brand"].to_numpy(),
            "product_id": reviews["product_id"].to_numpy(),
            "product_name": reviews["product_name"].to_numpy(),
            "skin_type": reviews["skin_type"].to_numpy(),
            "price_tier": reviews["price_tier"].to_numpy(),
            "review_word_count": reviews["review_word_count"].to_numpy(),
        }
    )
    for topic_idx in range(N_TOPICS):
        clustering_input[f"nmf_topic_{topic_idx + 1}_weight"] = W_norm[:, topic_idx]
    clustering_input = pd.concat([clustering_input, skin_onehot, brand_onehot, tier_onehot], axis=1)
    clustering_input.to_csv(OUT_DIR / "review_level_clustering_input_feature_table.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "iteration": np.arange(1, len(errors) + 1),
            "reconstruction_error": errors,
        }
    ).to_csv(OUT_DIR / "nmf_reconstruction_history.csv", index=False, encoding="utf-8-sig")

    draw_topic_overview(topic_summary, OUT_DIR / "nmf_topic_overview.png")

    explanation = [
        "# IFB214 NMF Process",
        "",
        f"Dataset: `{DATA_PATH.name}`.",
        f"Rows in workbook: {workbook_rows:,}.",
        f"Rows used after combining review title and review text: {len(reviews):,}.",
        f"Vocabulary size used for NMF input: {len(vocab)} terms.",
        f"Number of latent topics: {N_TOPICS}.",
        "",
        "## What NMF is doing",
        "NMF factorises the non-negative TF-IDF matrix `X` into two non-negative matrices: `W` and `H`, so that `X ≈ W x H`.",
        "`W` is the review-level feature matrix. Each row is one review and each topic weight shows how strongly that review is associated with each latent topic.",
        "`H` is the topic-term feature matrix. Each row is one latent topic and each term loading shows how strongly that term helps define the topic.",
        "",
        "## Why these numbers appear",
        "A topic weight in `W` is larger when the review contains a pattern of words that the model repeatedly sees as part of the same latent theme.",
        "A term loading in `H` is larger when that term consistently helps reconstruct many reviews assigned to the same latent topic.",
        "Because all values must stay non-negative, NMF tends to produce more interpretable additive themes than methods that allow positive and negative term weights.",
        "",
        "## Clustering input features",
        "Included for clustering input: 8 NMF topic weights, `review_word_count`, one-hot encoded `skin_type`, one-hot encoded `brand_name`, and one-hot encoded `price_tier`.",
        "Explicitly excluded from clustering input: `rating` and `is_recommended`.",
        "",
        "## Main outputs",
        "- `review_level_nmf_feature_table.csv`: one review per row with the 8 NMF topic weights.",
        "- `review_level_clustering_input_feature_table.csv`: the final clustering input feature table.",
        "- `nmf_topic_term_feature_table.csv`: the topic-term feature table showing the strongest terms for each topic.",
        "- `nmf_topic_summary.csv`: theme label, share, definition, representative keywords, and anonymised excerpt.",
        "- `nmf_topic_overview.png`: visual topic summary inspired by your reference layout.",
        "- `nmf_reconstruction_history.csv`: reconstruction error over iterations.",
    ]
    (OUT_DIR / "nmf_explanation.md").write_text("\n".join(explanation), encoding="utf-8")

    print("NMF topic summary")
    print(topic_summary.to_string(index=False))
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
