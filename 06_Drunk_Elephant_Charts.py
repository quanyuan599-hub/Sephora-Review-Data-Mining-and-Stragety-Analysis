from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path.cwd()
INPUT_DIR = PROJECT_DIR / "outputs" / "drunk_elephant_cluster_diagnosis"
OUT_DIR = INPUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = INPUT_DIR / "drunk_elephant_cluster_comparison_summary.csv"
BRAND_CLUSTER_PATH = INPUT_DIR / "brand_cluster_profile_k5.csv"

EQUAL_SHARE = 100 / 3
BRANDS = ["Drunk Elephant", "Tatcha", "The Ordinary"]
BRAND_COLORS = {
    "Drunk Elephant": (228, 111, 81),
    "Tatcha": (74, 144, 226),
    "The Ordinary": (88, 88, 88),
}
HEAT_COLORS = {
    "low": (245, 245, 245),
    "high": (53, 132, 228),
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


def blend_color(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    low = HEAT_COLORS["low"]
    high = HEAT_COLORS["high"]
    if vmax == vmin:
        ratio = 0.5
    else:
        ratio = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    return tuple(int(low[i] + (high[i] - low[i]) * ratio) for i in range(3))


def shorten_cluster(name: str) -> str:
    mapping = {
        "Accessible Results Seekers": "Results",
        "Lightweight Finish Appreciation": "Lightweight Finish",
        "Breakout Management with Caution": "Breakout Caution",
        "Hydration-focused Satisfaction": "Hydration",
        "Gentle Cleansing Satisfaction": "Gentle Cleansing",
    }
    return mapping.get(name, name)


def draw_grouped_bar_chart(summary: pd.DataFrame, path: Path) -> None:
    width, height = 1500, 900
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((36, 24), "Figure A. Brand Share by Consumer Experience Cluster", font=font(30, True), fill=(28, 28, 28))
    draw.text((36, 60), "Each cluster shows the internal brand mix of reviews in the final K=5 segmentation.", font=font(16), fill=(95, 95, 95))

    left, top, right, bottom = 120, 140, 70, 180
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_max = 80.0

    for j in range(5):
        y = top + j * plot_h / 4
        draw.line((left, y, left + plot_w, y), fill=(232, 236, 240))
        val = y_max - j * y_max / 4
        draw.text((52, y - 10), f"{val:.0f}%", font=font(13), fill=(95, 95, 95))

    cluster_gap = plot_w / len(summary)
    bar_w = cluster_gap * 0.18
    for idx, row in enumerate(summary.itertuples(index=False)):
        cx = left + idx * cluster_gap + cluster_gap / 2
        shares = {
            "Drunk Elephant": float(row.drunk_elephant_share_pct),
            "Tatcha": float(row.tatcha_share_pct),
            "The Ordinary": float(row.the_ordinary_share_pct),
        }
        for offset_i, brand in enumerate(BRANDS):
            x1 = cx + (offset_i - 1) * bar_w * 1.35 - bar_w / 2
            x2 = x1 + bar_w
            y2 = top + plot_h
            y1 = y2 - shares[brand] / y_max * plot_h
            draw.rounded_rectangle((x1, y1, x2, y2), radius=6, fill=BRAND_COLORS[brand])
            draw.text((x1 + 2, y1 - 22), f"{shares[brand]:.1f}%", font=font(12, True), fill=BRAND_COLORS[brand])
        label_lines = wrap_text(shorten_cluster(str(row.cluster_name)), 16)
        yy = top + plot_h + 18
        for line in label_lines[:2]:
            tw = draw.textlength(line, font=font(14, True))
            draw.text((cx - tw / 2, yy), line, font=font(14, True), fill=(45, 45, 45))
            yy += 18

    legend_x = width - 350
    legend_y = 26
    for i, brand in enumerate(BRANDS):
        x = legend_x + i * 105
        draw.rounded_rectangle((x, legend_y, x + 20, legend_y + 20), radius=4, fill=BRAND_COLORS[brand])
        draw.text((x + 26, legend_y + 1), brand.replace("Drunk Elephant", "DE").replace("The Ordinary", "TO"), font=font(13, True), fill=(55, 55, 55))

    img.save(path)


def draw_overindex_chart(summary: pd.DataFrame, path: Path) -> None:
    width, height = 1450, 820
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((36, 24), "Figure B. Drunk Elephant Over-index by Cluster", font=font(30, True), fill=(28, 28, 28))
    draw.text((36, 60), "Positive values mean Drunk Elephant is more concentrated than an equal three-brand baseline of 33.3%.", font=font(16), fill=(95, 95, 95))

    left, top, right, bottom = 300, 130, 100, 100
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = summary["drunk_elephant_over_index_pct"].astype(float).tolist()
    vmin = min(-10.0, min(values) - 1.0)
    vmax = max(10.0, max(values) + 1.0)
    zero_x = left + (-vmin) / (vmax - vmin) * plot_w

    draw.line((zero_x, top, zero_x, top + plot_h), fill=(120, 120, 120), width=2)
    draw.text((zero_x - 10, top - 26), "0", font=font(13, True), fill=(90, 90, 90))

    row_h = plot_h / len(summary)
    for idx, row in enumerate(summary.itertuples(index=False)):
        y = top + idx * row_h + row_h * 0.2
        bar_h = row_h * 0.6
        val = float(row.drunk_elephant_over_index_pct)
        x_end = left + (val - vmin) / (vmax - vmin) * plot_w
        fill = (228, 111, 81) if val >= 0 else (120, 120, 120)
        x1, x2 = sorted([zero_x, x_end])
        draw.rounded_rectangle((x1, y, x2, y + bar_h), radius=8, fill=fill)
        label = shorten_cluster(str(row.cluster_name))
        draw.text((40, y + 6), label, font=font(15, True), fill=(45, 45, 45))
        value_text = f"{val:+.1f} pts"
        tx = x2 + 8 if val >= 0 else x1 - draw.textlength(value_text, font=font(14, True)) - 8
        draw.text((tx, y + 7), value_text, font=font(14, True), fill=fill)

    img.save(path)


def draw_heatmap(brand_cluster: pd.DataFrame, path: Path) -> None:
    rating_map = {}
    rec_map = {}
    cluster_order = []
    for cluster_name, sub in brand_cluster.groupby("cluster_name", sort=False):
        cluster_order.append(cluster_name)
        for row in sub.itertuples(index=False):
            rating_map[(cluster_name, row.brand)] = float(row.avg_rating)
            rec_map[(cluster_name, row.brand)] = float(row.recommendation_rate)

    rating_vals = list(rating_map.values())
    rec_vals = list(rec_map.values())
    width, height = 1600, 980
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((36, 24), "Figure C. Brand Performance Heatmap by Cluster", font=font(30, True), fill=(28, 28, 28))
    draw.text((36, 60), "Each cell shows average rating and recommendation rate for a brand within a consumer experience cluster.", font=font(16), fill=(95, 95, 95))

    left = 290
    top = 150
    cell_w = 360
    cell_h = 120
    header_h = 54

    for j, brand in enumerate(BRANDS):
        x = left + j * cell_w
        draw.rounded_rectangle((x, top, x + cell_w - 16, top + header_h), radius=10, fill=BRAND_COLORS[brand])
        tw = draw.textlength(brand, font=font(18, True))
        draw.text((x + (cell_w - 16 - tw) / 2, top + 14), brand, font=font(18, True), fill="white")

    for i, cluster_name in enumerate(cluster_order):
        y = top + header_h + 12 + i * cell_h
        cluster_short = shorten_cluster(cluster_name)
        lines = wrap_text(cluster_short, 20)
        yy = y + 18
        for line in lines[:2]:
            draw.text((36, yy), line, font=font(17, True), fill=(45, 45, 45))
            yy += 22

        for j, brand in enumerate(BRANDS):
            x = left + j * cell_w
            rating = rating_map[(cluster_name, brand)]
            rec = rec_map[(cluster_name, brand)]
            fill = blend_color(rec, min(rec_vals), max(rec_vals))
            draw.rounded_rectangle((x, y, x + cell_w - 16, y + cell_h - 18), radius=12, fill=fill, outline=(228, 232, 236))
            draw.text((x + 18, y + 20), f"Rating: {rating:.2f}", font=font(18, True), fill=(35, 35, 35))
            draw.text((x + 18, y + 54), f"Rec rate: {rec:.1f}%", font=font(18, True), fill=(35, 35, 35))

    legend_x = 1220
    legend_y = 30
    for i in range(120):
        color = blend_color(i, 0, 119)
        draw.line((legend_x + i, legend_y, legend_x + i, legend_y + 18), fill=color, width=1)
    draw.text((legend_x, legend_y + 24), "Lower approval", font=font(12), fill=(95, 95, 95))
    draw.text((legend_x + 70, legend_y + 24), "Higher approval", font=font(12), fill=(95, 95, 95))

    img.save(path)


def main() -> None:
    summary = pd.read_csv(SUMMARY_PATH)
    brand_cluster = pd.read_csv(BRAND_CLUSTER_PATH)

    fig1 = OUT_DIR / "figureA_brand_share_by_cluster.png"
    fig2 = OUT_DIR / "figureB_drunk_elephant_overindex.png"
    fig3 = OUT_DIR / "figureC_brand_performance_heatmap.png"

    draw_grouped_bar_chart(summary, fig1)
    draw_overindex_chart(summary, fig2)
    draw_heatmap(brand_cluster, fig3)

    lines = [
        "# Drunk Elephant cluster chart notes",
        "",
        "## Figure A. Brand share by cluster",
        "This chart compares the internal brand mix of each consumer experience cluster.",
        "It is useful for showing which experience spaces are led by Tatcha, The Ordinary, or Drunk Elephant.",
        "The figure shows that Tatcha dominates the cleansing and lightweight finish clusters, while The Ordinary dominates the results-led and breakout-related clusters.",
        "",
        "## Figure B. Drunk Elephant over-index by cluster",
        "This figure focuses only on Drunk Elephant and compares its share in each cluster against an equal three-brand baseline of 33.3%.",
        "Positive bars indicate that Drunk Elephant is more concentrated than expected in that cluster; negative bars indicate relative under-representation.",
        "The strongest positive concentration appears in Hydration-focused Satisfaction, while the strongest under-index appears in Lightweight Finish Appreciation.",
        "",
        "## Figure C. Brand performance heatmap",
        "This heatmap compares average rating and recommendation rate by brand within each consumer experience cluster.",
        "It is especially useful for identifying relative weaknesses: for example, Drunk Elephant trails Tatcha and The Ordinary in Gentle Cleansing Satisfaction and in Breakout Management with Caution.",
        "Together, the three figures show where Drunk Elephant is present, where it is over- or under-indexed, and whether it performs better or worse than competing brands within the same experience space.",
    ]
    (INPUT_DIR / "drunk_elephant_cluster_chart_notes.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
