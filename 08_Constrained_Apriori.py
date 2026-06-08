from __future__ import annotations

from pathlib import Path
from itertools import combinations

import pandas as pd


PROJECT_DIR = Path.cwd()
DATA_PATH = PROJECT_DIR / "outputs" / "sephora_target_brand_6000_reviews.xlsx"
NMF_FEATURE_PATH = PROJECT_DIR / "outputs" / "【保留】ifb214_nmf_process" / "review_level_nmf_feature_table.csv"
NMF_SUMMARY_PATH = PROJECT_DIR / "outputs" / "【保留】ifb214_nmf_process" / "nmf_topic_summary.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "task3_constrained_apriori"
TABLE_DIR = OUT_DIR / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.60
MIN_LIFT = 1.05
SENSITIVITY_SUPPORT = 0.05
SENSITIVITY_CONFIDENCE = 0.80

TOPIC_COLS = [f"nmf_topic_{i}_weight" for i in range(1, 9)]


def make_price_tier(values: pd.Series) -> pd.Series:
    q1 = float(values.quantile(1 / 3))
    q2 = float(values.quantile(2 / 3))

    def bucket(v: float) -> str:
        if v <= q1:
            return "low_price"
        if v <= q2:
            return "mid_price"
        return "high_price"

    return values.map(bucket)


def clean_skin_type(values: pd.Series) -> pd.Series:
    out = values.fillna("Unknown / Missing").astype(str).str.strip()
    out = out.replace({"Unknown": "Unknown / Missing", "unknown": "Unknown / Missing", "": "Unknown / Missing"})
    return out


def parse_item(item: str) -> str:
    prefix, value = item.split("=", 1)
    mapping = {
        "theme": f"NMF theme = {value}",
        "nmf_theme": f"NMF theme = {value}",
        "skin_type": f"skin type = {value}",
        "skin_type_clean": f"skin type = {value}",
        "price_tier": f"price tier = {value}",
        "recommendation": f"recommendation = {value}",
        "recommendation_label": f"recommendation = {value}",
        "rating": f"rating = {value}",
        "rating_label": f"rating = {value}",
        "brand": f"brand = {value}",
    }
    return mapping.get(prefix, item)


def support(mask: pd.Series) -> float:
    return float(mask.mean())


def build_rule_rows(df: pd.DataFrame, antecedent_cols: tuple[str, str], consequent_col: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total_n = len(df)
    grouped = df.groupby(list(antecedent_cols) + [consequent_col]).size().reset_index(name="count")
    antecedent_counts = df.groupby(list(antecedent_cols)).size().reset_index(name="antecedent_count")
    consequent_counts = df.groupby(consequent_col).size().reset_index(name="consequent_count")

    merged = grouped.merge(antecedent_counts, on=list(antecedent_cols), how="left")
    merged = merged.merge(consequent_counts, on=consequent_col, how="left")

    for row in merged.itertuples(index=False):
        antecedent_items = tuple(
            f"{col}={getattr(row, col)}"
            for col in antecedent_cols
        )
        consequent_item = f"{consequent_col}={getattr(row, consequent_col)}"
        supp = float(row.count / total_n)
        conf = float(row.count / row.antecedent_count)
        consequent_support = float(row.consequent_count / total_n)
        lift = float(conf / consequent_support) if consequent_support > 0 else 0.0
        rows.append(
            {
                "rule_family": f"{antecedent_cols[0]} + {antecedent_cols[1]} => {consequent_col}",
                "antecedent": " | ".join(antecedent_items),
                "consequent": consequent_item,
                "support": supp,
                "confidence": conf,
                "lift": lift,
                "antecedent_count": int(row.antecedent_count),
                "rule_count": int(row.count),
                "consequent_support": consequent_support,
            }
        )
    return rows


def business_meaning(antecedent: str, consequent: str, supp: float, conf: float, lift: float) -> str:
    ant = ", ".join(parse_item(x) for x in antecedent.split(" | "))
    con = parse_item(consequent)
    return (
        f"When reviews are characterised by {ant}, they are associated with {con}. "
        f"This rule appears in {supp * 100:.1f}% of all reviews, with confidence {conf * 100:.1f}% and lift {lift:.2f}."
    )


def main() -> None:
    reviews = pd.read_excel(DATA_PATH, sheet_name="final_6000_reviews").copy().reset_index(drop=True)
    reviews.insert(0, "review_id", range(1, len(reviews) + 1))

    nmf = pd.read_csv(NMF_FEATURE_PATH)
    topic_summary = pd.read_csv(NMF_SUMMARY_PATH).sort_values("topic").reset_index(drop=True)
    topic_map = {int(row.topic): str(row.theme) for row in topic_summary.itertuples()}

    merged = pd.merge(
        reviews[["review_id", "brand", "rating", "is_recommended", "skin_type", "price_usd_clean"]],
        nmf[["review_id", *TOPIC_COLS]],
        on="review_id",
        how="left",
    )
    merged["skin_type_clean"] = clean_skin_type(merged["skin_type"])
    merged["price_tier"] = make_price_tier(merged["price_usd_clean"])
    merged["recommendation_label"] = merged["is_recommended"].map({1: "yes", 0: "no"}).fillna("unknown")
    merged["rating_label"] = merged["rating"].astype(int).astype(str)
    merged["dominant_topic_num"] = merged[TOPIC_COLS].idxmax(axis=1).str.extract(r"(\d+)").astype(int)
    merged["nmf_theme"] = merged["dominant_topic_num"].map(topic_map)

    transaction_representation = pd.DataFrame(
        {
            "transaction_id": merged["review_id"],
            "skin_type": "skin_type=" + merged["skin_type_clean"].astype(str),
            "price_tier": "price_tier=" + merged["price_tier"].astype(str),
            "nmf_theme": "theme=" + merged["nmf_theme"].astype(str),
            "recommendation": "recommendation=" + merged["recommendation_label"].astype(str),
            "rating": "rating=" + merged["rating_label"].astype(str),
            "brand": "brand=" + merged["brand"].astype(str),
        }
    )
    transaction_representation.to_csv(TABLE_DIR / "01_transaction_representation_constrained.csv", index=False, encoding="utf-8-sig")

    one_hot = pd.get_dummies(transaction_representation.drop(columns="transaction_id"))
    one_hot.insert(0, "transaction_id", transaction_representation["transaction_id"])
    one_hot.to_csv(TABLE_DIR / "02_transaction_matrix_constrained.csv", index=False, encoding="utf-8-sig")

    antecedent_pairs = [
        ("nmf_theme", "price_tier"),
        ("nmf_theme", "skin_type_clean"),
        ("skin_type_clean", "price_tier"),
    ]
    consequent_targets = [
        ("recommendation_label", "recommendation"),
        ("brand", "brand"),
        ("rating_label", "rating"),
    ]

    all_rule_rows: list[dict[str, object]] = []
    for left, right in antecedent_pairs:
        for raw_consequent, _ in consequent_targets:
            all_rule_rows.extend(build_rule_rows(merged, (left, right), raw_consequent))

    rules = pd.DataFrame(all_rule_rows)
    rules["business_meaning"] = rules.apply(
        lambda row: business_meaning(
            str(row["antecedent"]),
            str(row["consequent"]),
            float(row["support"]),
            float(row["confidence"]),
            float(row["lift"]),
        ),
        axis=1,
    )
    rules = rules.sort_values(["rule_family", "lift", "confidence", "support"], ascending=[True, False, False, False]).reset_index(drop=True)
    rules.to_csv(TABLE_DIR / "03_all_constrained_rules.csv", index=False, encoding="utf-8-sig")

    filtered_rules = rules[
        (rules["support"] >= MIN_SUPPORT)
        & (rules["confidence"] >= MIN_CONFIDENCE)
        & (rules["lift"] >= MIN_LIFT)
    ].copy()
    filtered_rules = filtered_rules.sort_values(["rule_family", "lift", "confidence", "support"], ascending=[True, False, False, False]).reset_index(drop=True)
    filtered_rules.to_csv(TABLE_DIR / "04_filtered_constrained_rules.csv", index=False, encoding="utf-8-sig")

    family_summary = (
        filtered_rules.groupby("rule_family")
        .agg(
            rule_count=("antecedent", "size"),
            max_lift=("lift", "max"),
            max_confidence=("confidence", "max"),
            max_support=("support", "max"),
        )
        .reset_index()
        .sort_values("rule_family")
        .reset_index(drop=True)
    )
    family_summary.to_csv(TABLE_DIR / "05_rule_family_summary.csv", index=False, encoding="utf-8-sig")

    # Curated report-ready table with 8 business-relevant rules.
    preferred_pairs = [
        ("nmf_theme=Exfoliation & Brightening | price_tier=low_price", "brand=The Ordinary"),
        ("nmf_theme=Lightweight Cream Texture & Finish | price_tier=high_price", "brand=Tatcha"),
        ("nmf_theme=Value, Size & Repurchase | price_tier=low_price", "recommendation_label=yes"),
        ("nmf_theme=Usage Amount & Product Longevity | price_tier=mid_price", "rating_label=5"),
        ("nmf_theme=Lightweight Cream Texture & Finish | skin_type_clean=combination", "brand=Tatcha"),
        ("nmf_theme=Usage Amount & Product Longevity | skin_type_clean=combination", "recommendation_label=yes"),
        ("skin_type_clean=combination | price_tier=low_price", "brand=The Ordinary"),
        ("skin_type_clean=dry | price_tier=high_price", "brand=Tatcha"),
    ]
    selected_rows = []
    for antecedent, consequent in preferred_pairs:
        match = filtered_rules[
            (filtered_rules["antecedent"] == antecedent)
            & (filtered_rules["consequent"] == consequent)
        ]
        if not match.empty:
            selected_rows.append(match.iloc[0])

    report_rules = pd.DataFrame(selected_rows).copy()
    concise_rows = []
    for row in report_rules.itertuples(index=False):
        concise_rows.append(
            {
                "rule_family": row.rule_family,
                "antecedent": row.antecedent,
                "consequent": row.consequent,
                "support_pct": round(float(row.support) * 100, 2),
                "confidence_pct": round(float(row.confidence) * 100, 2),
                "lift": round(float(row.lift), 3),
                "business_meaning": row.business_meaning,
            }
        )
    concise_table = pd.DataFrame(concise_rows)
    concise_table.to_csv(TABLE_DIR / "06_report_ready_rules.csv", index=False, encoding="utf-8-sig")

    def pretty_rule(antecedent: str, consequent: str) -> str:
        ant = " + ".join(parse_item(x) for x in antecedent.split(" | "))
        con = parse_item(consequent)
        return f"{ant} -> {con}"

    def concise_business_interpretation(antecedent: str, consequent: str) -> str:
        text = pretty_rule(antecedent, consequent)
        if "brand = The Ordinary" in text and "low_price" in antecedent:
            return "Low-price, results-led needs are strongly aligned with The Ordinary's value positioning."
        if "brand = Tatcha" in text and "high_price" in antecedent:
            return "Premium price tiers reinforce Tatcha's sensory and prestige positioning."
        if "recommendation = yes" in text and "Value, Size & Repurchase" in text:
            return "Value-oriented claims at low price levels strengthen positive recommendation intent."
        if "rating = 5" in text and "Usage Amount & Product Longevity" in text:
            return "Consumers reward long-lasting, efficient products with very high satisfaction."
        if "Lightweight Cream Texture & Finish" in text and "brand = Tatcha" in text:
            return "Combination-skin consumers link lightweight finish benefits with Tatcha's texture-led positioning."
        if "recommendation = yes" in text and "combination" in text:
            return "Combination-skin shoppers respond positively when product longevity and amount feel worthwhile."
        if "skin type = combination" in text and "brand = The Ordinary" in text:
            return "Low-price merchandising for combination skin is closely associated with The Ordinary."
        if "skin type = dry" in text and "brand = Tatcha" in text:
            return "Dry-skin consumers at higher price tiers are more strongly aligned with Tatcha's premium care positioning."
        return "This rule highlights a meaningful link between consumer profile, need state, and brand or satisfaction outcome."

    top8_table = concise_table.copy()
    top8_table["Rule"] = top8_table.apply(lambda row: pretty_rule(str(row["antecedent"]), str(row["consequent"])), axis=1)
    top8_table["Business interpretation"] = top8_table.apply(
        lambda row: concise_business_interpretation(str(row["antecedent"]), str(row["consequent"])),
        axis=1,
    )
    top8_table = top8_table[["Rule", "support_pct", "confidence_pct", "lift", "Business interpretation"]].rename(
        columns={
            "support_pct": "Support",
            "confidence_pct": "Confidence",
        }
    )
    top8_table.to_csv(TABLE_DIR / "09_top8_rules_table.csv", index=False, encoding="utf-8-sig")

    # Sensitivity check using stricter thresholds.
    sensitivity_rules = rules[
        (rules["support"] >= SENSITIVITY_SUPPORT)
        & (rules["confidence"] >= SENSITIVITY_CONFIDENCE)
        & (rules["lift"] >= MIN_LIFT)
    ].copy()
    sensitivity_rules = sensitivity_rules.sort_values(["rule_family", "lift", "confidence", "support"], ascending=[True, False, False, False]).reset_index(drop=True)
    sensitivity_rules.to_csv(TABLE_DIR / "07_sensitivity_filtered_rules.csv", index=False, encoding="utf-8-sig")

    report_rule_keys = set(zip(concise_table["antecedent"], concise_table["consequent"]))
    sensitivity_keys = set(zip(sensitivity_rules["antecedent"], sensitivity_rules["consequent"]))
    stable_keys = report_rule_keys.intersection(sensitivity_keys)
    sensitivity_summary = pd.DataFrame(
        [
            {
                "base_min_support": MIN_SUPPORT,
                "base_min_confidence": MIN_CONFIDENCE,
                "sensitivity_min_support": SENSITIVITY_SUPPORT,
                "sensitivity_min_confidence": SENSITIVITY_CONFIDENCE,
                "base_rule_count": len(filtered_rules),
                "sensitivity_rule_count": len(sensitivity_rules),
                "report_rule_count": len(concise_table),
                "stable_report_rule_count": len(stable_keys),
                "stable_report_rule_share_pct": round(len(stable_keys) / len(concise_table) * 100, 2) if len(concise_table) else 0.0,
            }
        ]
    )
    sensitivity_summary.to_csv(TABLE_DIR / "08_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    workbook_path = OUT_DIR / "task3_constrained_apriori_outputs.xlsx"
    with pd.ExcelWriter(workbook_path) as writer:
        transaction_representation.to_excel(writer, sheet_name="transaction_repr", index=False)
        one_hot.to_excel(writer, sheet_name="transaction_matrix", index=False)
        rules.to_excel(writer, sheet_name="all_rules", index=False)
        filtered_rules.to_excel(writer, sheet_name="filtered_rules", index=False)
        family_summary.to_excel(writer, sheet_name="rule_family_summary", index=False)
        concise_table.to_excel(writer, sheet_name="report_ready_rules", index=False)
        sensitivity_rules.to_excel(writer, sheet_name="sensitivity_rules", index=False)
        sensitivity_summary.to_excel(writer, sheet_name="sensitivity_summary", index=False)
        top8_table.to_excel(writer, sheet_name="top8_rules_table", index=False)

    notes = [
        "# Task 3 Constrained Apriori Analysis",
        "",
        "## Rule structure",
        "The analysis uses a constrained association-rule design:",
        "",
        "Antecedents:",
        "- NMF Theme + Price Tier",
        "- NMF Theme + Skin Type",
        "- Skin Type + Price Tier",
        "",
        "Consequents:",
        "- Recommendation",
        "- Brand",
        "- Rating",
        "",
        "This produces nine targeted rule families:",
        "- Theme + Price Tier -> Recommendation",
        "- Theme + Price Tier -> Brand",
        "- Theme + Price Tier -> Rating",
        "- Theme + Skin Type -> Recommendation",
        "- Theme + Skin Type -> Brand",
        "- Theme + Skin Type -> Rating",
        "- Skin Type + Price Tier -> Recommendation",
        "- Skin Type + Price Tier -> Brand",
        "- Skin Type + Price Tier -> Rating",
        "",
        "## NMF theme representation",
        "Each review is assigned to its dominant NMF theme, defined as the theme with the highest topic score among the eight NMF topics.",
        "",
        "## Rule evaluation",
        f"Rules are retained for interpretation when support >= {MIN_SUPPORT:.2f}, confidence >= {MIN_CONFIDENCE:.2f}, and lift >= {MIN_LIFT:.2f}.",
        "- Support shows how common the rule is in the full dataset.",
        "- Confidence shows how strongly the antecedent predicts the consequent.",
        "- Lift shows whether the rule is stronger than random co-occurrence.",
        "",
        "## Sensitivity check",
        f"A stricter sensitivity check is also reported using support >= {SENSITIVITY_SUPPORT:.2f} and confidence >= {SENSITIVITY_CONFIDENCE:.2f}.",
        "This helps assess whether the main business-relevant rules remain visible under a tougher filtering standard.",
    ]
    (OUT_DIR / "task3_constrained_apriori_explanation.md").write_text("\n".join(notes), encoding="utf-8")

    wording = [
        "# Task 3 Constrained Report Wording",
        "",
        "For Task 3, each review was transformed into a transaction representation containing skin type, price tier, dominant NMF theme, recommendation status, rating, and brand. Rather than mining unrestricted rules, the association analysis used a constrained structure in which the antecedents were always built from consumer-profile and experience variables, while the consequents were limited to recommendation, brand, or rating outcomes.",
        "",
        "Specifically, nine rule families were evaluated: Theme plus Price Tier, Theme plus Skin Type, and Skin Type plus Price Tier were each tested against Recommendation, Brand, and Rating. This constrained structure was retained because it improves business interpretability and keeps the rules aligned with the strategic question of how consumer profile cues and experience themes relate to outcomes.",
        "",
        "The strength of each association was evaluated using support, confidence, and lift. Support measures how often the full pattern appears in the dataset, confidence captures the conditional probability of the consequent given the antecedent, and lift indicates whether the rule is stronger than chance. Stronger lift values point to more distinctive positioning and more meaningful market structure.",
        "",
        "A brief sensitivity check was also performed using stricter thresholds. This helps show whether the strongest report-level rules are stable rather than threshold-dependent.",
    ]
    (OUT_DIR / "task3_constrained_report_wording.md").write_text("\n".join(wording), encoding="utf-8")

    print("Saved constrained outputs to", OUT_DIR)
    print("Filtered constrained rules:", len(filtered_rules))
    if not concise_table.empty:
        print(concise_table.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
