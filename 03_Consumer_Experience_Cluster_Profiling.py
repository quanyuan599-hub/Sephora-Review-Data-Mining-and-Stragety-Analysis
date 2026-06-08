from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path.cwd()
BASE_DIR = PROJECT_DIR / "outputs" / "【保留】ifb214_full_feature_k_interpretability" / "tables"
OUT_DIR = PROJECT_DIR / "outputs" / "consumer_experience_cluster_profiling"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_PATH = BASE_DIR / "cluster_profile_k5.csv"
ASSIGNMENTS_PATH = BASE_DIR / "cluster_assignments_k3_to_k6.csv"
FEATURE_PATH = BASE_DIR / "review_level_feature_table_full_input.csv"

COLORS = {
    0: (52, 101, 164),
    1: (243, 156, 18),
    2: (231, 76, 60),
    3: (39, 174, 96),
    4: (142, 68, 173),
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
    lines = []
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


def compact_theme_string(row: pd.Series) -> str:
    return " | ".join(
        [
            f"{row['top_theme_1']}",
            f"{row['top_theme_2']}",
            f"{row['top_theme_3']}",
        ]
    )


def interpret_cluster(row: pd.Series) -> str:
    label = str(row["draft_business_label"])
    dominant_brand = str(row["dominant_brand"])
    dominant_skin = str(row["dominant_skin_type"])
    rec = float(row["recommendation_rate"])
    rating = float(row["average_rating"])
    dominant_themes = [str(row["top_theme_1"]), str(row["top_theme_2"]), str(row["top_theme_3"])]
    first_theme = dominant_themes[0]

    if label == "Hydration-focused Satisfaction":
        return f"A strongly positive hydration-oriented segment, led by {first_theme.lower()} needs, with high satisfaction among mainly {dominant_skin} skin reviewers and a slight tilt toward {dominant_brand}."
    if label == "Gentle Cleansing Satisfaction":
        return f"A high-approval cleansing segment where users emphasise gentle but effective cleansing performance; it is most associated with {dominant_brand} and mainly {dominant_skin} skin."
    if label == "Breakout Management with Caution":
        return f"A results-seeking but more cautious acne-management segment. Reviews centre on breakout control and brightening, but the lower recommendation rate suggests uneven outcomes."
    if label == "Lightweight Finish Appreciation":
        return f"A texture-led segment focused on lightweight finish and wearability. Users often value how the product feels on skin, with {dominant_brand} clearly over-represented."
    if label == "Accessible Results Seekers":
        return f"A large performance-led segment combining visible results with value cues. It leans toward lower-price products and is disproportionately shaped by {dominant_brand}."
    return f"A mixed experience segment defined by {first_theme.lower()} and {dominant_themes[1].lower()} themes, with average rating {rating:.2f} and recommendation rate {rec:.1f}%."


def pca_2d(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return centered @ components


def min_max_scale(values: np.ndarray, low: float, high: float) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        return np.full_like(values, (low + high) / 2.0, dtype=float)
    return low + (values - vmin) * (high - low) / (vmax - vmin)


def draw_pca_plot(coords: np.ndarray, labels: np.ndarray, label_names: dict[int, str], path: Path) -> None:
    width, height = 1500, 980
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    draw.text((36, 24), "PCA Cluster Visualization", font=font(30, True), fill=(28, 28, 28))
    draw.text((36, 62), "K=5 consumer experience clusters based on 8 NMF topics, review length, skin type and price tier", font=font(16), fill=(95, 95, 95))

    left, top, right, bottom = 110, 120, 260, 110
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_vals = coords[:, 0]
    y_vals = coords[:, 1]
    xs = min_max_scale(x_vals, left, left + plot_w)
    ys = min_max_scale(-y_vals, top, top + plot_h)

    draw.rectangle((left, top, left + plot_w, top + plot_h), outline=(210, 214, 220), width=2)
    x_mid = float(min_max_scale(np.array([0.0]), left, left + plot_w)[0]) if (x_vals.min() <= 0 <= x_vals.max()) else None
    y_mid = float(min_max_scale(np.array([0.0]), top + plot_h, top)[0]) if (y_vals.min() <= 0 <= y_vals.max()) else None
    if x_mid is not None:
        draw.line((x_mid, top, x_mid, top + plot_h), fill=(232, 236, 240), width=1)
    if y_mid is not None:
        draw.line((left, y_mid, left + plot_w, y_mid), fill=(232, 236, 240), width=1)

    for cluster in sorted(label_names):
        mask = labels == cluster
        color = COLORS[cluster]
        for x, y in zip(xs[mask], ys[mask]):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

    for cluster in sorted(label_names):
        mask = labels == cluster
        cx = float(xs[mask].mean())
        cy = float(ys[mask].mean())
        color = COLORS[cluster]
        draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=color, outline="white", width=2)
        tag = f"C{cluster}"
        tw = draw.textlength(tag, font=font(14, True))
        draw.rounded_rectangle((cx + 10, cy - 14, cx + 22 + tw, cy + 14), radius=8, fill=(255, 255, 255), outline=color, width=2)
        draw.text((cx + 16, cy - 9), tag, font=font(14, True), fill=color)

    legend_x = width - 230
    legend_y = 150
    draw.rounded_rectangle((legend_x, legend_y, width - 36, 150 + 40 + 5 * 110), radius=14, fill=(248, 250, 252), outline=(224, 228, 232))
    draw.text((legend_x + 16, legend_y + 14), "Cluster Legend", font=font(18, True), fill=(35, 35, 35))
    for i, cluster in enumerate(sorted(label_names)):
        yy = legend_y + 52 + i * 110
        color = COLORS[cluster]
        draw.ellipse((legend_x + 16, yy + 4, legend_x + 36, yy + 24), fill=color)
        draw.text((legend_x + 46, yy), f"C{cluster}: {label_names[cluster]}", font=font(15, True), fill=(35, 35, 35))
        for j, line in enumerate(wrap_text(label_names[cluster], 22)[:2]):
            if j == 0:
                continue
            draw.text((legend_x + 46, yy + 20 + j * 18), line, font=font(13), fill=(95, 95, 95))

    draw.text((left, height - 70), "PC1", font=font(16, True), fill=(80, 80, 80))
    draw.text((36, top - 8), "PC2", font=font(16, True), fill=(80, 80, 80))
    img.save(path)


def main() -> None:
    profile = pd.read_csv(PROFILE_PATH)
    assignments = pd.read_csv(ASSIGNMENTS_PATH)
    features = pd.read_csv(FEATURE_PATH)

    merged = pd.merge(
        features,
        assignments[["review_id", "cluster_k5"]],
        on="review_id",
        how="left",
    )

    nmf_cols = [c for c in merged.columns if re.fullmatch(r"nmf_topic_\d+_weight", c)]
    review_length_col = "review_word_count.1" if "review_word_count.1" in merged.columns else "review_word_count"
    skin_cols = [c for c in merged.columns if c.startswith("skin_type_")]
    price_cols = [c for c in merged.columns if c.startswith("price_tier_")]
    feature_cols = [*nmf_cols, review_length_col, *skin_cols, *price_cols]

    profiling_rows = []
    label_names = {}
    for row in profile.itertuples(index=False):
        cluster_name = str(row.draft_business_label)
        label_names[int(row.cluster)] = cluster_name
        profiling_rows.append(
            {
                "Cluster Name": cluster_name,
                "Size": f"{int(row.cluster_size_count)} ({row.cluster_size_pct:.1f}%)",
                "Dominant Themes": f"{row.top_theme_1}; {row.top_theme_2}; {row.top_theme_3}",
                "Rating": round(float(row.average_rating), 2),
                "Rec Rate": f"{float(row.recommendation_rate):.1f}%",
                "Dominant Skin Type": f"{row.dominant_skin_type} ({float(row.dominant_skin_type_share_pct):.1f}%)",
                "Dominant Brand": f"{row.dominant_brand} ({float(row.dominant_brand_share_pct):.1f}%)",
                "Interpretation": interpret_cluster(pd.Series(row._asdict())),
            }
        )

    profiling_table = pd.DataFrame(profiling_rows)
    profiling_csv = TABLE_DIR / "consumer_experience_cluster_profiling.csv"
    profiling_table.to_csv(profiling_csv, index=False, encoding="utf-8-sig")

    profiling_xlsx = TABLE_DIR / "consumer_experience_cluster_profiling.xlsx"
    try:
        profiling_table.to_excel(profiling_xlsx, index=False)
    except Exception:
        pass

    x = merged[feature_cols].to_numpy(dtype=np.float32)
    coords = pca_2d(x)
    labels = merged["cluster_k5"].to_numpy(dtype=np.int32)

    pca_coords = pd.DataFrame(
        {
            "review_id": merged["review_id"],
            "cluster_k5": labels,
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
        }
    )
    pca_coords.to_csv(TABLE_DIR / "pca_cluster_coordinates_k5.csv", index=False, encoding="utf-8-sig")

    figure_path = FIG_DIR / "pca_cluster_visualization_k5.png"
    draw_pca_plot(coords, labels, label_names, figure_path)

    description_lines = [
        "# Consumer Experience Cluster Profiling",
        "",
        "This profiling table summarises the final K=5 consumer experience segmentation based on 8 NMF topics, review length, skin type and price tier.",
        "",
        "## Table reading guide",
        "- `Cluster Name`: draft business-facing label for the segment.",
        "- `Size`: number of reviews and share of the 6,000-review analytical sample.",
        "- `Dominant Themes`: top three NMF themes with the highest mean presence in the cluster.",
        "- `Rating` and `Rec Rate`: outcome variables used only after clustering to interpret satisfaction levels.",
        "- `Dominant Skin Type` and `Dominant Brand`: the most common profile within the cluster.",
        "",
        "## Cluster summary",
    ]

    for row in profiling_table.itertuples(index=False):
        description_lines.append(
            f"- `{row[0]}`: {row[7]}"
        )

    description_lines += [
        "",
        "## PCA figure interpretation",
        "The PCA visualisation projects the multi-feature clustering input into two dimensions so the broad spatial separation of the five clusters can be inspected visually.",
        "Clusters that form denser point clouds indicate more internally consistent review patterns, while overlap suggests adjacent experience types rather than fully isolated segments.",
        "This figure should be treated as a visual summary rather than the basis for choosing K, because PCA compresses the original feature space into only two components.",
    ]

    description_path = OUT_DIR / "consumer_experience_cluster_profiling_description.md"
    description_path.write_text("\n".join(description_lines), encoding="utf-8")

    print(profiling_table.to_string(index=False))
    print(f"\nSaved figure to {figure_path}")


if __name__ == "__main__":
    main()
