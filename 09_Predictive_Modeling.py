from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd


PROJECT_DIR = Path.cwd()
DATA_PATH = PROJECT_DIR / "outputs" / "sephora_target_brand_6000_reviews.xlsx"
NMF_FEATURE_PATH = PROJECT_DIR / "outputs" / "【保留】ifb214_nmf_process" / "review_level_nmf_feature_table.csv"
NMF_SUMMARY_PATH = PROJECT_DIR / "outputs" / "【保留】ifb214_nmf_process" / "nmf_topic_summary.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "task4_predictive_modeling"
TABLE_DIR = OUT_DIR / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
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
        if token_score == 0:
            continue
        prev = tokens[i - 1] if i >= 1 else ""
        prev2 = tokens[i - 2] if i >= 2 else ""
        if prev in NEGATORS or prev2 in NEGATORS:
            token_score *= -1.0
        if prev in INTENSIFIERS:
            token_score *= 1.35
        score += token_score
    return score / (math.sqrt(len(tokens)) + 1.0)


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


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def roc_auc_score_manual(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    pos = y_true == 1
    n_pos = pos.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def f1_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def accuracy_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def precision_recall_manual(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return float(precision), float(recall)


class LogisticRegressionGD:
    def __init__(self, lr: float = 0.05, epochs: int = 3000, l2: float = 0.001):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        n, p = x.shape
        w = np.zeros(p, dtype=float)
        b = 0.0
        for _ in range(self.epochs):
            z = x @ w + b
            p_hat = sigmoid(z)
            error = p_hat - y
            grad_w = (x.T @ error) / n + self.l2 * w
            grad_b = float(error.mean())
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.coef_ = w
        self.intercept_ = b

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(x @ self.coef_ + self.intercept_)  # type: ignore[operator]

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)


@dataclass
class TreeNode:
    feature: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None
    value: float | None = None


class DecisionTreeClassifierSimple:
    def __init__(self, max_depth: int = 6, min_samples_split: int = 25, max_features: int | None = None, random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root: TreeNode | None = None
        self.feature_importances_: np.ndarray | None = None
        self._rng = np.random.default_rng(random_state)

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p = float(y.mean())
        return 1.0 - p ** 2 - (1 - p) ** 2

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None, float]:
        n, p = x.shape
        parent_gini = self._gini(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        feat_indices = np.arange(p)
        if self.max_features is not None and self.max_features < p:
            feat_indices = self._rng.choice(feat_indices, size=self.max_features, replace=False)

        for feat in feat_indices:
            values = np.unique(x[:, feat])
            if len(values) <= 1:
                continue
            if len(values) > 12:
                qs = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
                thresholds = np.unique(qs)
            else:
                thresholds = (values[:-1] + values[1:]) / 2
            for thr in thresholds:
                left_mask = x[:, feat] <= thr
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                weighted = (left_mask.sum() / n) * left_gini + (right_mask.sum() / n) * right_gini
                gain = parent_gini - weighted
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feat)
                    best_threshold = float(thr)
        return best_feature, best_threshold, best_gain

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        node = TreeNode(value=float(y.mean()))
        if depth >= self.max_depth or len(y) < self.min_samples_split or self._gini(y) == 0.0:
            return node
        feat, thr, gain = self._best_split(x, y)
        if feat is None or thr is None or gain <= 1e-8:
            return node
        left_mask = x[:, feat] <= thr
        right_mask = ~left_mask
        node.feature = feat
        node.threshold = thr
        node.left = self._build(x[left_mask], y[left_mask], depth + 1)
        node.right = self._build(x[right_mask], y[right_mask], depth + 1)
        self.feature_importances_[feat] += gain * len(y)  # type: ignore[index]
        return node

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.feature_importances_ = np.zeros(x.shape[1], dtype=float)
        self.root = self._build(x, y, 0)

    def _predict_row(self, row: np.ndarray, node: TreeNode) -> float:
        if node.feature is None or node.threshold is None or node.left is None or node.right is None:
            return float(node.value)
        if row[node.feature] <= node.threshold:
            return self._predict_row(row, node.left)
        return self._predict_row(row, node.right)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(row, self.root) for row in x], dtype=float)  # type: ignore[arg-type]


class RandomForestSimple:
    def __init__(self, n_trees: int = 60, max_depth: int = 6, min_samples_split: int = 25, max_features: int | None = None, random_state: int = 42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees: list[DecisionTreeClassifierSimple] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        n, p = x.shape
        max_features = self.max_features or max(1, int(np.sqrt(p)))
        self.trees = []
        importances = np.zeros(p, dtype=float)
        for i in range(self.n_trees):
            idx = rng.choice(np.arange(n), size=n, replace=True)
            tree = DecisionTreeClassifierSimple(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                random_state=self.random_state + i + 1,
            )
            tree.fit(x[idx], y[idx])
            self.trees.append(tree)
            importances += tree.feature_importances_
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        preds = np.vstack([tree.predict_proba(x) for tree in self.trees])
        return preds.mean(axis=0)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)


def permutation_importance(model, x_test: np.ndarray, y_test: np.ndarray, rng_seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    baseline = roc_auc_score_manual(y_test, model.predict_proba(x_test))
    importances = []
    for j in range(x_test.shape[1]):
        x_perm = x_test.copy()
        rng.shuffle(x_perm[:, j])
        score = roc_auc_score_manual(y_test, model.predict_proba(x_perm))
        importances.append(baseline - score)
    return np.array(importances, dtype=float)


def train_test_split_manual(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(len(df) * (1 - test_size))
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def build_feature_set(exclude_top1_price: bool = False) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    reviews = pd.read_excel(DATA_PATH, sheet_name="final_6000_reviews").copy().reset_index(drop=True)
    reviews.insert(0, "review_id", np.arange(1, len(reviews) + 1))
    nmf = pd.read_csv(NMF_FEATURE_PATH)
    topic_summary = pd.read_csv(NMF_SUMMARY_PATH).sort_values("topic").reset_index(drop=True)
    topic_map = {int(row.topic): str(row.theme) for row in topic_summary.itertuples()}

    df = pd.merge(
        reviews[["review_id", "brand", "skin_type", "price_usd_clean", "review_text", "review_title", "rating", "is_recommended"]],
        nmf[["review_id", "review_word_count", *TOPIC_COLS]],
        on="review_id",
        how="left",
    )
    if exclude_top1_price:
        n_drop = max(1, int(np.ceil(len(df) * 0.01)))
        drop_idx = df["price_usd_clean"].sort_values(ascending=False).head(n_drop).index
        df = df.drop(index=drop_idx).reset_index(drop=True)

    df["skin_type_clean"] = clean_skin_type(df["skin_type"])
    df["price_tier"] = make_price_tier(df["price_usd_clean"])
    df["combined_text"] = (
        df["review_title"].fillna("").astype(str).str.strip()
        + " "
        + df["review_text"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["sentiment_score"] = df["combined_text"].map(sentiment_score)
    df["dominant_topic_num"] = df[TOPIC_COLS].idxmax(axis=1).str.extract(r"(\d+)").astype(int)
    df["nmf_theme"] = df["dominant_topic_num"].map(topic_map)
    df["target"] = df["is_recommended"].astype(int)

    feature_design = [
        {"feature_type": "Structured features", "feature": "brand", "explanation": "Captures broad brand positioning and brand-level expectation effects."},
        {"feature_type": "Structured features", "feature": "skin_type", "explanation": "Represents consumer profile heterogeneity across dry, oily, combination, normal, or unknown skin."},
        {"feature_type": "Structured features", "feature": "price_tier", "explanation": "Captures perceived affordability and value band rather than raw price alone."},
        {"feature_type": "Structured features", "feature": "review_length", "explanation": "Acts as a proxy for engagement depth and review intensity."},
        {"feature_type": "Text-derived features", "feature": "NMF theme", "explanation": "Captures the dominant product experience or need expressed in the review."},
        {"feature_type": "Text-derived features", "feature": "sentiment", "explanation": "Captures the overall emotional tone of the review text."},
    ]
    return df, feature_design


def prepare_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    modeling = pd.DataFrame(
        {
            "brand": df["brand"].astype(str),
            "skin_type": df["skin_type_clean"].astype(str),
            "price_tier": df["price_tier"].astype(str),
            "review_length": df["review_word_count"].astype(float),
            "nmf_theme": df["nmf_theme"].astype(str),
            "sentiment": df["sentiment_score"].astype(float),
        }
    )
    x_cat = pd.get_dummies(
        modeling[["brand", "skin_type", "price_tier", "nmf_theme"]],
        drop_first=True,
        dtype=float,
    )
    review_len = modeling["review_length"].to_numpy()
    sentiment = modeling["sentiment"].to_numpy()
    modeling["review_length_scaled"] = (review_len - review_len.mean()) / (review_len.std() if review_len.std() else 1.0)
    modeling["sentiment_scaled"] = (sentiment - sentiment.mean()) / (sentiment.std() if sentiment.std() else 1.0)
    x = pd.concat([x_cat, modeling[["review_length_scaled", "sentiment_scaled"]]], axis=1)
    y = df["target"].to_numpy(dtype=int)
    return modeling, x, y


def evaluate_binary_model(model_name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float | str]:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall = precision_recall_manual(y_true, y_pred)
    return {
        "model": model_name,
        "auc": round(roc_auc_score_manual(y_true, y_prob), 4),
        "f1": round(f1_score_manual(y_true, y_pred), 4),
        "accuracy": round(accuracy_score_manual(y_true, y_pred), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def run_pipeline(exclude_top1_price: bool = False, suffix: str = "base") -> dict[str, pd.DataFrame]:
    df, feature_design = build_feature_set(exclude_top1_price=exclude_top1_price)
    modeling, x_df, y = prepare_model_matrix(df)
    master = pd.concat([df[["review_id", "brand", "skin_type_clean", "price_tier", "rating", "is_recommended", "nmf_theme", "sentiment_score"]], x_df], axis=1)
    master.to_csv(TABLE_DIR / f"01_feature_table_{suffix}.csv", index=False, encoding="utf-8-sig")

    train_df, test_df = train_test_split_manual(pd.concat([x_df, pd.Series(y, name="target")], axis=1), test_size=TEST_SIZE, random_state=RANDOM_STATE)
    x_train = train_df.drop(columns="target").to_numpy(dtype=float)
    y_train = train_df["target"].to_numpy(dtype=int)
    x_test = test_df.drop(columns="target").to_numpy(dtype=float)
    y_test = test_df["target"].to_numpy(dtype=int)

    logit = LogisticRegressionGD(lr=0.08, epochs=3500, l2=0.002)
    logit.fit(x_train, y_train)
    logit_prob = logit.predict_proba(x_test)
    logit_metrics = evaluate_binary_model("Logistic Regression", y_test, logit_prob)
    coef_df = pd.DataFrame(
        {
            "feature": x_df.columns,
            "coefficient": logit.coef_,  # type: ignore[arg-type]
            "odds_ratio": np.exp(logit.coef_),  # type: ignore[arg-type]
            "effect_direction": np.where(np.array(logit.coef_) >= 0, "positive", "negative"),  # type: ignore[arg-type]
        }
    ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    coef_df.to_csv(TABLE_DIR / f"02_logistic_coefficients_{suffix}.csv", index=False, encoding="utf-8-sig")

    rf = RandomForestSimple(n_trees=60, max_depth=6, min_samples_split=30, random_state=RANDOM_STATE)
    rf.fit(x_train, y_train)
    rf_prob = rf.predict_proba(x_test)
    rf_metrics = evaluate_binary_model("Random Forest", y_test, rf_prob)
    perm_imp = permutation_importance(rf, x_test, y_test, rng_seed=RANDOM_STATE)
    importance_df = pd.DataFrame(
        {
            "feature": x_df.columns,
            "permutation_importance_auc_drop": perm_imp,
            "forest_split_importance": rf.feature_importances_,
        }
    ).sort_values("permutation_importance_auc_drop", ascending=False).reset_index(drop=True)
    importance_df.to_csv(TABLE_DIR / f"03_random_forest_importance_{suffix}.csv", index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame([logit_metrics, rf_metrics])
    metrics_df.to_csv(TABLE_DIR / f"04_model_metrics_{suffix}.csv", index=False, encoding="utf-8-sig")

    class_balance = pd.DataFrame(
        [
            {
                "dataset": suffix,
                "n_reviews": len(df),
                "recommended_yes_pct": round(float(df["target"].mean() * 100), 2),
                "recommended_no_pct": round(float((1 - df["target"].mean()) * 100), 2),
                "train_size": len(train_df),
                "test_size": len(test_df),
            }
        ]
    )
    class_balance.to_csv(TABLE_DIR / f"00_dataset_balance_{suffix}.csv", index=False, encoding="utf-8-sig")

    feature_design_df = pd.DataFrame(feature_design)
    feature_design_df.to_csv(TABLE_DIR / "00_feature_design_table.csv", index=False, encoding="utf-8-sig")

    return {
        "class_balance": class_balance,
        "feature_design": feature_design_df,
        "metrics": metrics_df,
        "coef": coef_df,
        "importance": importance_df,
    }


def main() -> None:
    base = run_pipeline(exclude_top1_price=False, suffix="base")
    robust = run_pipeline(exclude_top1_price=True, suffix="excluding_top1pct_price")

    robustness = (
        base["metrics"]
        .merge(robust["metrics"], on="model", suffixes=("_base", "_robust"))
        .assign(
            auc_change=lambda d: d["auc_robust"] - d["auc_base"],
            f1_change=lambda d: d["f1_robust"] - d["f1_base"],
        )
    )
    robustness.to_csv(TABLE_DIR / "05_robustness_comparison.csv", index=False, encoding="utf-8-sig")

    explanation = [
        "# Task 4 Predictive Analysis",
        "",
        "Primary KPI: `is_recommended`",
        "",
        "## Feature set",
        "Structured features:",
        "- brand",
        "- skin_type",
        "- price_tier",
        "- review_length",
        "",
        "Text-derived features:",
        "- NMF theme",
        "- sentiment",
        "",
        "## Modelling workflow",
        "- 80/20 train-test split",
        "- logistic regression for interpretable coefficients",
        "- random forest for non-linear predictive modelling",
        "- AUC and F1 used as the primary model comparison metrics",
        "- robustness check excluding the top 1% of price outliers",
    ]
    (OUT_DIR / "task4_predictive_modeling_explanation.md").write_text("\n".join(explanation), encoding="utf-8")

    base_metrics = base["metrics"]
    robust_metrics = robust["metrics"]
    base_coef = base["coef"].head(10)
    base_importance = base["importance"].head(10)

    wording = [
        "# Task 4 Report Wording",
        "",
        "Task 4 models recommendation behaviour using `is_recommended` as the primary KPI. The modelling design combines structured variables (brand, skin type, price tier, and review length) with text-derived variables (dominant NMF theme and sentiment). This design allows the analysis to test whether recommendation behaviour is explained more strongly by brand and consumer profile, or by the experience themes expressed in the reviews.",
        "",
        "Two predictive models were estimated using an 80/20 train-test split. Logistic regression was used first to provide interpretable coefficients, showing the direction and magnitude of each feature's association with recommendation behaviour. Random forest was then used to capture non-linear relationships and to compare predictive performance using feature importance.",
        "",
        f"In the base sample, Logistic Regression achieved AUC {base_metrics.loc[base_metrics['model']=='Logistic Regression','auc'].iloc[0]:.3f} and F1 {base_metrics.loc[base_metrics['model']=='Logistic Regression','f1'].iloc[0]:.3f}, while Random Forest achieved AUC {base_metrics.loc[base_metrics['model']=='Random Forest','auc'].iloc[0]:.3f} and F1 {base_metrics.loc[base_metrics['model']=='Random Forest','f1'].iloc[0]:.3f}.",
        "",
        "The logistic regression coefficients and random forest importance rankings should be interpreted together: coefficients indicate directional drivers of recommendation, while feature importance indicates which variables contribute most to overall predictive discrimination.",
        "",
        "A robustness check was also carried out by excluding the top 1% of price outliers. This allows the analysis to assess whether extreme premium-price observations materially affect model conclusions.",
    ]
    (OUT_DIR / "task4_report_wording.md").write_text("\n".join(wording), encoding="utf-8")

    workbook_path = OUT_DIR / "task4_predictive_modeling_outputs.xlsx"
    with pd.ExcelWriter(workbook_path) as writer:
        base["feature_design"].to_excel(writer, sheet_name="feature_design", index=False)
        base["class_balance"].to_excel(writer, sheet_name="dataset_balance_base", index=False)
        robust["class_balance"].to_excel(writer, sheet_name="dataset_balance_robust", index=False)
        base["metrics"].to_excel(writer, sheet_name="metrics_base", index=False)
        robust["metrics"].to_excel(writer, sheet_name="metrics_robust", index=False)
        robustness.to_excel(writer, sheet_name="robustness_compare", index=False)
        base["coef"].to_excel(writer, sheet_name="logit_coef_base", index=False)
        robust["coef"].to_excel(writer, sheet_name="logit_coef_robust", index=False)
        base["importance"].to_excel(writer, sheet_name="rf_importance_base", index=False)
        robust["importance"].to_excel(writer, sheet_name="rf_importance_robust", index=False)

    print("Base metrics")
    print(base["metrics"].to_string(index=False))
    print("\nRobustness comparison")
    print(robustness.to_string(index=False))
    print("\nTop logistic coefficients")
    print(base_coef.to_string(index=False))
    print("\nTop random forest features")
    print(base_importance.to_string(index=False))


if __name__ == "__main__":
    main()
