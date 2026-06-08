from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path.cwd()
INPUT_PATH = PROJECT_DIR / "outputs" / "【保留】ifb214_full_feature_k_interpretability" / "tables" / "k_interpretability_comparison.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "consumer_experience_cluster_profiling" / "figures"
TABLE_DIR = PROJECT_DIR / "outputs" / "consumer_experience_cluster_profiling" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


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


def scale(values, low, high):
    vmin = float(min(values))
    vmax = float(max(values))
    if vmax == vmin:
        return [(low + high) / 2.0 for _ in values]
    return [low + (v - vmin) * (high - low) / (vmax - vmin) for v in values]


def main() -> None:
    df = pd.read_csv(INPUT_PATH).sort_values("k").reset_index(drop=True)
    df[["k", "inertia", "approx_silhouette"]].to_csv(
        TABLE_DIR / "k3_to_k6_elbow_silhouette_values.csv", index=False, encoding="utf-8-sig"
    )

    ks = df["k"].astype(int).tolist()
    inertias = df["inertia"].astype(float).tolist()
    sils = df["approx_silhouette"].astype(float).tolist()

    width, height = 1500, 900
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    draw.text((36, 24), "K=3 to K=6 Elbow and Silhouette Diagnostics", font=font(30, True), fill=(28, 28, 28))
    draw.text((36, 62), "Based on the full clustering input: 8 NMF topics, review length, skin type and price tier", font=font(16), fill=(95, 95, 95))

    left, top, right, bottom = 110, 130, 110, 120
    plot_w = width - left - right
    plot_h = height - top - bottom

    # Grid
    for j in range(5):
        y = top + j * plot_h / 4
        draw.line((left, y, left + plot_w, y), fill=(232, 236, 240))

    # Left-axis labels for inertia
    inertia_min = min(inertias)
    inertia_max = max(inertias)
    for j in range(5):
        y = top + j * plot_h / 4
        val = inertia_max - j * (inertia_max - inertia_min) / 4
        draw.text((18, y - 8), f"{val:,.0f}", font=font(12), fill=(52, 101, 164))

    # Right-axis labels for silhouette
    sil_min = min(sils)
    sil_max = max(sils)
    for j in range(5):
        y = top + j * plot_h / 4
        val = sil_max - j * (sil_max - sil_min) / 4
        txt = f"{val:.3f}"
        tw = draw.textlength(txt, font=font(12))
        draw.text((width - 20 - tw, y - 8), txt, font=font(12), fill=(228, 111, 81))

    xs = scale(ks, left + 40, left + plot_w - 40)
    ys_inertia = scale(inertias, top + plot_h, top)
    ys_sil = scale(sils, top + plot_h, top)

    # vertical guides + x labels
    for x, k in zip(xs, ks):
        draw.line((x, top, x, top + plot_h), fill=(246, 247, 249))
        draw.text((x - 8, top + plot_h + 16), str(k), font=font(14, True), fill=(55, 55, 55))

    # Elbow line
    elbow_pts = list(zip(xs, ys_inertia))
    draw.line(elbow_pts, fill=(52, 101, 164), width=4)
    for x, y, val in zip(xs, ys_inertia, inertias):
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(52, 101, 164))
        draw.text((x - 24, y - 28), f"{val:,.0f}", font=font(11, True), fill=(52, 101, 164))

    # Silhouette line
    sil_pts = list(zip(xs, ys_sil))
    draw.line(sil_pts, fill=(228, 111, 81), width=4)
    for x, y, val in zip(xs, ys_sil, sils):
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(228, 111, 81))
        draw.text((x - 16, y + 10), f"{val:.3f}", font=font(11, True), fill=(228, 111, 81))

    # Mark recommended K=5
    k5_x = xs[ks.index(5)]
    draw.line((k5_x, top, k5_x, top + plot_h), fill=(120, 120, 120), width=2)
    draw.text((k5_x + 8, top + 8), "Recommended K=5", font=font(13, True), fill=(70, 70, 70))

    # Legend
    legend_x = width - 360
    legend_y = 92
    draw.rounded_rectangle((legend_x, legend_y, width - 36, legend_y + 72), radius=12, fill=(248, 250, 252), outline=(226, 230, 234))
    draw.line((legend_x + 18, legend_y + 24, legend_x + 52, legend_y + 24), fill=(52, 101, 164), width=4)
    draw.ellipse((legend_x + 30, legend_y + 18, legend_x + 42, legend_y + 30), fill=(52, 101, 164))
    draw.text((legend_x + 62, legend_y + 14), "Elbow / Inertia", font=font(15, True), fill=(52, 101, 164))
    draw.line((legend_x + 18, legend_y + 50, legend_x + 52, legend_y + 50), fill=(228, 111, 81), width=4)
    draw.ellipse((legend_x + 30, legend_y + 44, legend_x + 42, legend_y + 56), fill=(228, 111, 81))
    draw.text((legend_x + 62, legend_y + 40), "Silhouette", font=font(15, True), fill=(228, 111, 81))

    draw.text((left, height - 58), "Number of Clusters (K)", font=font(16, True), fill=(60, 60, 60))

    out_path = OUT_DIR / "k3_to_k6_elbow_silhouette_combined.png"
    img.save(out_path)

    notes = [
        "# K=3 to K=6 Elbow and Silhouette Diagnostics",
        "",
        "This combined chart uses the final full-input clustering workflow:",
        "- 8 NMF topics",
        "- review length",
        "- skin type",
        "- price tier",
        "",
        "The blue line shows inertia for the Elbow method. Lower values indicate lower within-cluster distortion.",
        "The orange line shows the approximate silhouette score. Higher values indicate stronger separation between clusters.",
        "",
        "Across K=3 to K=6, inertia decreases steadily as K increases, while silhouette improves from 0.135 to 0.215.",
        "Although K=6 has the best statistical separation, K=5 remains the recommended solution for report use because it offers stronger readability and a cleaner consumer-experience story without adding an extra layer of segmentation complexity.",
    ]
    (PROJECT_DIR / "outputs" / "consumer_experience_cluster_profiling" / "k3_to_k6_elbow_silhouette_notes.md").write_text(
        "\n".join(notes), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
