"""Seller-oriented Streamlit dashboard for customer review analysis."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.complaint_summary import get_top_complaints
from src.fuzzy_system import compute_reliability_score
from src.preprocessing import preprocess_text


CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_reviews.csv"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best_sentiment_model.joblib"
SAMPLE_CATALOG_PATH = PROJECT_ROOT / "data" / "sample_seller_catalog.csv"
LABEL_ORDER = ["negative", "neutral", "positive"]
DATASET_SOURCE_DEFAULT = "Default Project Dataset"
DATASET_SOURCE_UPLOADED = "Uploaded Test Dataset"
UPLOADED_REQUIRED_COLUMNS = {"product_id", "product_title", "product_category"}
UPLOADED_OPTIONAL_COLUMNS = [
    "star_rating",
    "review_headline",
    "review_body",
    "review_date",
    "helpful_votes",
    "total_votes",
    "verified_purchase",
]


TEXT = {
    "tr": {
        "page_title": "Satıcı Yorum Paneli",
        "language": "Dil",
        "sidebar_title": "Panel Ayarları",
        "dataset_source": "Veri Kaynağı",
        "active_source_default": "Currently using: Default Project Dataset",
        "active_source_uploaded": "Currently using: Uploaded Test Dataset",
        "upload_test_dataset": "Satıcı test dataset CSV dosyası yükleyin",
        "upload_test_needed": "Please upload a seller test dataset CSV, or switch back to Default Project Dataset.",
        "uploaded_missing": "Yüklenen test datasetinde eksik zorunlu kolonlar var:",
        "uploaded_reviewed_rows": "Yorumlu satır",
        "uploaded_without_review_rows": "Yorumsuz satır",
        "download_uploaded_analysis": "Analiz edilmiş uploaded dataset CSV indir",
        "no_reviewed_rows": "Seçili veri kaynağında analiz edilecek yorum bulunamadı.",
        "uploaded_products_note": "Yüklenen test datasetinde review_body boş olan satırlar yorumsuz ürün olarak ayrılır; proje dataset dosyası değiştirilmez.",
        "uploaded_source_help": "Yüklenen CSV bu oturumda geçici olarak analiz edilir; clean_reviews.csv, model ve rapor dosyaları değiştirilmez.",
        "data_status": "Veri Durumu",
        "ready": "Hazır",
        "rows": "yorum",
        "products": "ürün",
        "hero_title": "Satıcı Yorum Zekası",
        "hero_subtitle": "Yorum alan ürünleri izleyin, riskli ürünleri belirleyin, şikayet temalarını özetleyin ve NLP modelini iş kararlarına bağlayın.",
        "missing_title": "Gerekli dosyalar eksik.",
        "missing_clean": "Temizlenmiş veri bulunamadı. Lütfen çalıştırın:",
        "missing_model": "Model dosyası bulunamadı. Lütfen çalıştırın:",
        "overview": "Genel Bakış",
        "reviewed_products": "Yorum Alan Ürünler",
        "product_detail": "Ürün Detay Analizi",
        "top_complaints": "Öne Çıkan Şikayetler",
        "without_reviews": "Yorumsuz Ürünler",
        "single_test": "Tekil Yorum Testi",
        "overview_title": "Satıcı Genel Görünümü",
        "overview_note": "Temizlenmiş Amazon Electronics yorum örneklemi üzerinden ürün, duygu ve güvenilirlik özeti.",
        "total_reviews": "Toplam Yorum",
        "reviewed_product_count": "Yorum Alan Ürün",
        "avg_rating": "Ortalama Puan",
        "avg_reliability": "Ortalama Güvenilirlik",
        "positive_reviews": "Olumlu Yorum",
        "neutral_reviews": "Nötr Yorum",
        "negative_reviews": "Olumsuz Yorum",
        "sentiment_dist": "Duygu Dağılımı",
        "rating_dist": "Puan Dağılımı",
        "length_dist": "Yorum Uzunluğu Dağılımı",
        "review_count": "Yorum Sayısı",
        "rating": "Puan",
        "category": "Kategori",
        "reviewed_products_note": "Ürünler product_id ve ürün adına göre gruplanır; risk etiketi olumsuz yorum oranına göre hesaplanır.",
        "product_category": "Ürün kategorisi",
        "all": "Tümü",
        "min_review_count": "Minimum yorum sayısı",
        "risk_filter": "Risk etiketi",
        "products_after_filters": "Filtre sonrası ürün",
        "high_risk": "Yüksek Risk",
        "medium_risk": "Orta Risk",
        "low_risk": "Düşük Risk",
        "risk": "Risk",
        "product_detail_note": "Seçilen ürün için duygu dağılımı, fuzzy güvenilirlik, şikayet temaları ve yorum tablosu.",
        "select_product": "Ürün seçin",
        "negative_ratio": "Olumsuz Oranı",
        "complaint_terms": "Şikayet kelimeleri / bigramlar",
        "no_complaints_product": "Bu ürün için olumsuz şikayet terimi bulunamadı.",
        "reviews_for_product": "Seçilen ürünün yorumları",
        "top_complaints_note": "Bu veri seti tarihsel olduğu için panel tekrar kullanılabilir bir şikayet izleme akışını gösterir. Canlı haftalık veriyle aynı mantık bu haftanın öne çıkan şikayetlerini üretebilir.",
        "negative_reviews_scope": "Seçilen kapsamda olumsuz yorum",
        "no_complaints_scope": "Bu kapsamda şikayet terimi bulunamadı.",
        "example_negative_reviews": "Örnek olumsuz yorumlar",
        "without_reviews_note": "Amazon Reviews veri seti yalnızca yorum almış ürünleri içerir. Bu nedenle yorumsuz ürünleri yalnızca yorum verisinden tespit edemeyiz. Yorumsuz ürünleri bulmak için satıcı ürün kataloğu yükleyin.",
        "expected_catalog": "Beklenen katalog kolonları: product_id, product_title, product_category",
        "upload_catalog": "Satıcı katalog CSV dosyası yükleyin",
        "upload_needed": "Tüm satıcı ürünlerini yorum verisiyle karşılaştırmak için katalog yükleyin.",
        "sample_format": "Örnek CSV formatı",
        "sample_available": "Demo katalog dosyası hazır:",
        "download_sample": "Demo katalog CSV indir",
        "catalog_missing": "Katalogda eksik kolonlar var:",
        "catalog_products": "Katalog Ürünü",
        "with_reviews": "Yorumlu Ürün",
        "without_reviews_count": "Yorumsuz Ürün",
        "coverage": "Yorum Kapsamı",
        "products_without_reviews": "Yorumsuz Ürünler",
        "products_with_reviews": "Yorumlu Ürünler",
        "single_note": "Bu sekme yalnızca hızlı teknik test içindir; ana iş akışı satıcı dashboard sekmeleridir.",
        "review_text": "Yorum metni",
        "review_age": "Yorum yaşı (gün)",
        "analyze": "Yorumu Analiz Et",
        "enter_review": "Lütfen bir yorum metni girin.",
        "predicted_sentiment": "Tahmin Edilen Duygu",
        "model_confidence": "Model Güveni",
        "fuzzy_reliability": "Fuzzy Güvenilirlik",
        "weighted_confidence": "Ağırlıklı Güven",
        "positive": "Olumlu",
        "neutral": "Nötr",
        "negative": "Olumsuz",
    },
    "en": {
        "page_title": "Seller Review Dashboard",
        "language": "Language",
        "sidebar_title": "Dashboard Settings",
        "dataset_source": "Dataset Source",
        "active_source_default": "Currently using: Default Project Dataset",
        "active_source_uploaded": "Currently using: Uploaded Test Dataset",
        "upload_test_dataset": "Upload seller test dataset CSV",
        "upload_test_needed": "Please upload a seller test dataset CSV, or switch back to Default Project Dataset.",
        "uploaded_missing": "Uploaded test dataset is missing required columns:",
        "uploaded_reviewed_rows": "Rows with reviews",
        "uploaded_without_review_rows": "Rows without reviews",
        "download_uploaded_analysis": "Download analyzed uploaded dataset CSV",
        "no_reviewed_rows": "The selected dataset source has no reviews to analyze.",
        "uploaded_products_note": "For the uploaded test dataset, rows with an empty review_body are separated as products without reviews; project dataset files are not changed.",
        "uploaded_source_help": "The uploaded CSV is analyzed temporarily in this session; clean_reviews.csv, model files, and report files are not modified.",
        "data_status": "Data Status",
        "ready": "Ready",
        "rows": "reviews",
        "products": "products",
        "hero_title": "Seller Review Intelligence",
        "hero_subtitle": "Monitor reviewed products, detect risky items, summarize complaint themes, and connect the NLP model to seller decisions.",
        "missing_title": "Required files are missing.",
        "missing_clean": "Cleaned dataset not found. Please run:",
        "missing_model": "Model file not found. Please run:",
        "overview": "Overview",
        "reviewed_products": "Reviewed Products",
        "product_detail": "Product Detail Analysis",
        "top_complaints": "Top Complaints",
        "without_reviews": "Products Without Reviews",
        "single_test": "Single Review Test",
        "overview_title": "Seller Review Overview",
        "overview_note": "Product, sentiment, and reliability summary from the cleaned Amazon Electronics review sample.",
        "total_reviews": "Total Reviews",
        "reviewed_product_count": "Reviewed Products",
        "avg_rating": "Average Rating",
        "avg_reliability": "Average Reliability",
        "positive_reviews": "Positive Reviews",
        "neutral_reviews": "Neutral Reviews",
        "negative_reviews": "Negative Reviews",
        "sentiment_dist": "Sentiment Distribution",
        "rating_dist": "Rating Distribution",
        "length_dist": "Review Length Distribution",
        "review_count": "Review Count",
        "rating": "Rating",
        "category": "Category",
        "reviewed_products_note": "Products are grouped by product_id and title; risk labels are calculated from negative review ratio.",
        "product_category": "Product category",
        "all": "All",
        "min_review_count": "Minimum review count",
        "risk_filter": "Risk label",
        "products_after_filters": "Products after filters",
        "high_risk": "High Risk",
        "medium_risk": "Medium Risk",
        "low_risk": "Low Risk",
        "risk": "Risk",
        "product_detail_note": "Sentiment distribution, fuzzy reliability, complaint themes, and review table for one selected product.",
        "select_product": "Select reviewed product",
        "negative_ratio": "Negative Ratio",
        "complaint_terms": "Complaint keywords / bigrams",
        "no_complaints_product": "No negative complaint terms found for this product.",
        "reviews_for_product": "Reviews for selected product",
        "top_complaints_note": "This dataset is historical, so the dashboard demonstrates a reusable complaint-monitoring pipeline. With live weekly data, the same logic can generate this week's top complaints.",
        "negative_reviews_scope": "Negative reviews in selected scope",
        "no_complaints_scope": "No complaint terms found for this scope.",
        "example_negative_reviews": "Example negative reviews",
        "without_reviews_note": "The Amazon Reviews dataset contains only products that already have reviews. Therefore, products without reviews cannot be detected from review data alone. To identify products without reviews, upload a seller product catalog.",
        "expected_catalog": "Expected catalog columns: product_id, product_title, product_category",
        "upload_catalog": "Upload seller catalog CSV",
        "upload_needed": "Upload a seller catalog to compare all seller products against reviewed products.",
        "sample_format": "Sample CSV format",
        "sample_available": "Demo catalog file available:",
        "download_sample": "Download demo catalog CSV",
        "catalog_missing": "Catalog is missing required columns:",
        "catalog_products": "Catalog Products",
        "with_reviews": "With Reviews",
        "without_reviews_count": "Without Reviews",
        "coverage": "Review Coverage",
        "products_without_reviews": "Products Without Reviews",
        "products_with_reviews": "Products With Reviews",
        "single_note": "This tab is only for quick technical testing; the main workflow is the seller dashboard.",
        "review_text": "Review text",
        "review_age": "Review age in days",
        "analyze": "Analyze review",
        "enter_review": "Please enter a review text.",
        "predicted_sentiment": "Predicted Sentiment",
        "model_confidence": "Model Confidence",
        "fuzzy_reliability": "Fuzzy Reliability",
        "weighted_confidence": "Weighted Confidence",
        "positive": "Positive",
        "neutral": "Neutral",
        "negative": "Negative",
    },
}

RISK_LABELS = {
    "High Risk": {"tr": "Yüksek Risk", "en": "High Risk"},
    "Medium Risk": {"tr": "Orta Risk", "en": "Medium Risk"},
    "Low Risk": {"tr": "Düşük Risk", "en": "Low Risk"},
}

SENTIMENT_COLORS = {
    "positive": "#10b981",
    "neutral": "#f59e0b",
    "negative": "#ef4444",
}


st.set_page_config(page_title="Seller Review Dashboard", layout="wide")


def get_language() -> str:
    with st.sidebar:
        language_label = st.selectbox("Dil / Language", ["Türkçe", "English"])
        selected_language = "tr" if language_label == "Türkçe" else "en"
        st.markdown(f"### {TEXT[selected_language]['sidebar_title']}")
    return selected_language


LANG = get_language()


def t(key: str) -> str:
    return TEXT[LANG].get(key, key)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --brand: #2563eb;
            --brand-dark: #1d4ed8;
            --ink: #172033;
            --muted: #64748b;
            --panel: #ffffff;
            --line: #d8e1ef;
            --soft: #eef5ff;
            --good: #10b981;
            --warn: #f59e0b;
            --bad: #ef4444;
        }
        .stApp {
            background:
                linear-gradient(180deg, #f6f9ff 0%, #eef4fb 42%, #f8fafc 100%);
            color: var(--ink);
        }
        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2.5rem;
            max-width: 1400px;
        }
        header[data-testid="stHeader"] {
            background: #f6f9ff;
            color: var(--ink);
            box-shadow: none;
        }
        div[data-testid="stToolbar"] {
            color: var(--ink);
        }
        div[data-testid="stDecoration"] {
            background: #2563eb;
        }
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
            color: var(--ink) !important;
        }
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
            color: var(--muted) !important;
        }
        section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
            background: #1f2433;
            color: #ffffff;
            border: 1px solid #1f2433;
            border-radius: 8px;
        }
        section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button *,
        section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button p {
            color: #ffffff !important;
        }
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stMetric"] label {
            color: var(--muted);
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink);
        }
        .hero {
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.95), rgba(20, 184, 166, 0.88)),
                linear-gradient(45deg, #1e293b, #0f766e);
            border-radius: 8px;
            padding: 28px 32px;
            color: white;
            margin-bottom: 18px;
            box-shadow: 0 14px 36px rgba(37, 99, 235, 0.20);
        }
        .hero h1 {
            color: white;
            margin: 0 0 8px 0;
            font-size: 2.25rem;
            letter-spacing: 0;
        }
        .hero p {
            color: rgba(255, 255, 255, 0.92);
            font-size: 1rem;
            max-width: 920px;
            margin: 0;
        }
        .section-note {
            color: var(--muted);
            font-size: 0.95rem;
            margin-top: -4px;
            margin-bottom: 12px;
        }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid transparent;
            white-space: nowrap;
        }
        .badge-high {
            background: #fee2e2;
            color: #991b1b;
            border-color: #fecaca;
        }
        .badge-medium {
            background: #fef3c7;
            color: #92400e;
            border-color: #fde68a;
        }
        .badge-low {
            background: #dcfce7;
            color: #166534;
            border-color: #bbf7d0;
        }
        .insight-strip {
            background: #ffffff;
            border: 1px solid var(--line);
            border-left: 4px solid var(--brand);
            border-radius: 8px;
            padding: 12px 14px;
            color: var(--muted);
            margin-bottom: 14px;
        }
        .source-strip {
            background: #ecfeff;
            border: 1px solid #a5f3fc;
            border-left: 4px solid #0891b2;
            border-radius: 8px;
            padding: 12px 14px;
            color: #164e63;
            font-weight: 700;
            margin: 0 0 16px 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid var(--line);
        }
        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-bottom: none;
            border-radius: 8px 8px 0 0;
            padding: 9px 13px;
            opacity: 1;
        }
        .stTabs [data-baseweb="tab"] *,
        .stTabs [data-baseweb="tab"] p {
            color: var(--ink) !important;
            opacity: 1;
        }
        .stTabs [aria-selected="true"] {
            background: #eff6ff;
            color: var(--brand-dark);
        }
        .stTabs [aria-selected="true"] *,
        .stTabs [aria-selected="true"] p {
            color: var(--brand-dark) !important;
            font-weight: 700;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme()


def sentiment_label(value: str) -> str:
    value = str(value)
    return t(value) if value in LABEL_ORDER else value


def risk_label(value: str) -> str:
    return RISK_LABELS.get(str(value), {"tr": str(value), "en": str(value)})[LANG]


def risk_badge(value: str) -> str:
    css = {
        "High Risk": "badge-high",
        "Medium Risk": "badge-medium",
        "Low Risk": "badge-low",
    }.get(str(value), "badge-low")
    return f'<span class="badge {css}">{risk_label(value)}</span>'


def info_note(text: str) -> None:
    st.markdown(f'<div class="section-note">{text}</div>', unsafe_allow_html=True)


def show_hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{t("hero_title")}</h1>
            <p>{t("hero_subtitle")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def select_dataset_source() -> tuple[str, object | None]:
    with st.sidebar:
        source = st.selectbox(
            t("dataset_source"),
            [DATASET_SOURCE_DEFAULT, DATASET_SOURCE_UPLOADED],
        )
        uploaded = None
        if source == DATASET_SOURCE_UPLOADED:
            st.caption(t("uploaded_source_help"))
            uploaded = st.file_uploader(
                t("upload_test_dataset"),
                type=["csv"],
                key="uploaded_test_dataset",
            )
    return source, uploaded


def show_active_source_notice(source: str) -> None:
    message_key = "active_source_uploaded" if source == DATASET_SOURCE_UPLOADED else "active_source_default"
    st.markdown(f'<div class="source-strip">{t(message_key)}</div>', unsafe_allow_html=True)


def show_missing_file_message(missing: list[Path]) -> None:
    st.error(t("missing_title"))
    for path in missing:
        st.write(f"- `{path.relative_to(PROJECT_ROOT)}`")
    if CLEAN_DATA_PATH in missing:
        st.info(t("missing_clean"))
        st.code("python -m src.data_loader", language="bash")
    if MODEL_PATH in missing:
        st.info(t("missing_model"))
        st.code("python -m src.train", language="bash")


@st.cache_data(show_spinner=False)
def load_clean_reviews(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["review_date"])
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["review_headline"] = df["review_headline"].fillna("").astype(str)
    df["review_body"] = df["review_body"].fillna("").astype(str)
    df["product_id"] = df.get("product_id", pd.Series(["unknown"] * len(df))).fillna("unknown").astype(str)
    df["product_title"] = df.get("product_title", pd.Series(["Unknown product"] * len(df))).fillna("Unknown product").astype(str)
    df["product_category"] = df.get("product_category", pd.Series(["Unknown"] * len(df))).fillna("Unknown").astype(str)
    return df


def safe_float(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@st.cache_data(show_spinner=False)
def load_uploaded_dataset(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))


def prepare_uploaded_dataset(uploaded_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    missing = sorted(UPLOADED_REQUIRED_COLUMNS.difference(uploaded_df.columns))
    if missing:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), missing

    normalized = uploaded_df.copy()
    normalized["_source_row_id"] = range(len(normalized))

    for column in UPLOADED_REQUIRED_COLUMNS:
        normalized[column] = normalized[column].fillna("").astype(str)

    for column in UPLOADED_OPTIONAL_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""

    normalized["review_headline"] = normalized["review_headline"].fillna("").astype(str)
    normalized["review_body"] = normalized["review_body"].fillna("").astype(str)
    normalized["verified_purchase"] = normalized["verified_purchase"].fillna("").astype(str)
    normalized["star_rating"] = pd.to_numeric(normalized["star_rating"], errors="coerce")
    normalized["helpful_votes"] = pd.to_numeric(normalized["helpful_votes"], errors="coerce").fillna(0)
    normalized["total_votes"] = pd.to_numeric(normalized["total_votes"], errors="coerce").fillna(0)

    has_review = normalized["review_body"].str.strip().ne("")
    no_review_rows = normalized[~has_review].copy()
    reviewed = normalized[has_review].copy()

    reviewed["review_text"] = (
        reviewed["review_headline"].fillna("").astype(str)
        + " "
        + reviewed["review_body"].fillna("").astype(str)
    ).str.strip()
    reviewed["review_length"] = reviewed["review_text"].str.split().map(len)
    reviewed["review_date"] = pd.to_datetime(reviewed["review_date"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    reviewed["review_age_days"] = (today - reviewed["review_date"]).dt.days.clip(lower=0).fillna(0)

    return reviewed, no_review_rows, normalized, []


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def add_model_predictions(df: pd.DataFrame, model_path: str, model_mtime: float) -> pd.DataFrame:
    scored = df.copy()
    if scored.empty:
        for column in ["predicted_sentiment", "sentiment_confidence", "reliability_score", "weighted_confidence"]:
            if column not in scored.columns:
                scored[column] = pd.Series(dtype="float64" if column != "predicted_sentiment" else "object")
        return scored

    model = joblib.load(model_path)
    processed_text = None

    if "predicted_sentiment" not in scored.columns:
        processed_text = scored["review_text"].map(preprocess_text)
        scored["predicted_sentiment"] = model.predict(processed_text)

    if "sentiment_confidence" not in scored.columns:
        if hasattr(model, "predict_proba"):
            if processed_text is None:
                processed_text = scored["review_text"].map(preprocess_text)
            probabilities = model.predict_proba(processed_text)
            scored["sentiment_confidence"] = probabilities.max(axis=1)
        else:
            scored["sentiment_confidence"] = np.nan

    if "reliability_score" not in scored.columns:
        scored["reliability_score"] = scored.apply(
            lambda row: compute_reliability_score(
                rating=safe_float(row.get("star_rating", 0)),
                review_length=safe_float(row.get("review_length", 0)),
                review_age_days=safe_float(row.get("review_age_days", 0)),
            ),
            axis=1,
        )

    if "weighted_confidence" not in scored.columns:
        scored["weighted_confidence"] = scored["sentiment_confidence"] * (
            scored["reliability_score"] / 100
        )

    return scored


@st.cache_data
def make_review_length_histogram(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "review_length" not in df.columns:
        return pd.DataFrame({"count": []})
    clipped = df["review_length"].clip(upper=df["review_length"].quantile(0.99))
    counts, edges = np.histogram(clipped, bins=28)
    labels = [f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(len(edges) - 1)]
    return pd.DataFrame({"count": counts}, index=labels)


@st.cache_data
def build_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_columns = [
        "product_id",
        "product_title",
        "product_category",
        "review_count",
        "average_rating",
        "positive_count",
        "neutral_count",
        "negative_count",
        "average_reliability_score",
        "positive_ratio",
        "neutral_ratio",
        "negative_ratio",
        "risk_label",
    ]
    if df.empty:
        return pd.DataFrame(columns=summary_columns)

    grouped = (
        df.groupby(["product_id", "product_title", "product_category"], dropna=False)
        .agg(
            review_count=("review_text", "size"),
            average_rating=("star_rating", "mean"),
            positive_count=("predicted_sentiment", lambda values: (values == "positive").sum()),
            neutral_count=("predicted_sentiment", lambda values: (values == "neutral").sum()),
            negative_count=("predicted_sentiment", lambda values: (values == "negative").sum()),
            average_reliability_score=("reliability_score", "mean"),
        )
        .reset_index()
    )

    grouped["positive_ratio"] = grouped["positive_count"] / grouped["review_count"]
    grouped["neutral_ratio"] = grouped["neutral_count"] / grouped["review_count"]
    grouped["negative_ratio"] = grouped["negative_count"] / grouped["review_count"]
    grouped["risk_label"] = np.select(
        [
            (grouped["negative_ratio"] >= 0.40) & (grouped["review_count"] >= 5),
            (grouped["negative_ratio"] >= 0.20) & (grouped["review_count"] >= 5),
        ],
        ["High Risk", "Medium Risk"],
        default="Low Risk",
    )
    return grouped.sort_values(["negative_ratio", "review_count"], ascending=[False, False])


def format_percent(value: float | int | None) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.1%}"


def localized_sentiment_counts(df: pd.DataFrame) -> pd.Series:
    counts = df["predicted_sentiment"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    counts.index = [sentiment_label(item) for item in counts.index]
    return counts


def localized_rating_counts(df: pd.DataFrame) -> pd.Series:
    counts = df["star_rating"].value_counts().sort_index()
    counts.index = [str(int(item)) for item in counts.index]
    return counts


def complaint_summary_for_scope(df: pd.DataFrame, category: str | None = None, top_n: int = 10) -> pd.DataFrame:
    scope = df.copy()
    scope["sentiment"] = scope["predicted_sentiment"]
    return get_top_complaints(scope, category=category, top_n=top_n)


def display_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    display["risk_label"] = display["risk_label"].map(risk_label)
    columns = {
        "product_id": "product_id",
        "product_title": "product_title" if LANG == "en" else "ürün_adı",
        "product_category": t("category"),
        "review_count": t("review_count"),
        "average_rating": t("avg_rating"),
        "positive_count": t("positive_reviews"),
        "neutral_count": t("neutral_reviews"),
        "negative_count": t("negative_reviews"),
        "positive_ratio": "positive_ratio" if LANG == "en" else "olumlu_oranı",
        "neutral_ratio": "neutral_ratio" if LANG == "en" else "nötr_oranı",
        "negative_ratio": "negative_ratio" if LANG == "en" else "olumsuz_oranı",
        "average_reliability_score": t("avg_reliability"),
        "risk_label": t("risk"),
    }
    return display.rename(columns=columns)


def display_reviews(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    if "predicted_sentiment" in display.columns:
        display["predicted_sentiment"] = display["predicted_sentiment"].map(sentiment_label)
    columns = {
        "review_date": "review_date" if LANG == "en" else "yorum_tarihi",
        "star_rating": t("rating"),
        "review_headline": "review_headline" if LANG == "en" else "yorum_başlığı",
        "review_body": "review_body" if LANG == "en" else "yorum_metni",
        "predicted_sentiment": t("predicted_sentiment"),
        "sentiment_confidence": t("model_confidence"),
        "reliability_score": t("fuzzy_reliability"),
        "weighted_confidence": t("weighted_confidence"),
        "product_title": "product_title" if LANG == "en" else "ürün_adı",
    }
    return display.rename(columns=columns)


def build_uploaded_export(all_rows: pd.DataFrame, scored_reviews: pd.DataFrame) -> pd.DataFrame:
    export = all_rows.copy()
    analysis_columns = [
        "_source_row_id",
        "review_text",
        "review_length",
        "review_age_days",
        "predicted_sentiment",
        "sentiment_confidence",
        "reliability_score",
        "weighted_confidence",
    ]
    available = [column for column in analysis_columns if column in scored_reviews.columns]
    if "_source_row_id" in export.columns and available:
        export = export.merge(
            scored_reviews[available],
            on="_source_row_id",
            how="left",
            suffixes=("", "_analysis"),
        )
    return export.drop(columns=["_source_row_id"], errors="ignore")


def show_sidebar_status(df: pd.DataFrame, no_review_rows: pd.DataFrame | None = None) -> None:
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"### {t('data_status')}")
        st.write(f"**{t('ready')}**")
        st.caption(f"{len(df):,} {t('rows')}")
        st.caption(f"{df['product_id'].nunique():,} {t('products')}")
        if no_review_rows is not None:
            st.caption(f"{len(df):,} {t('uploaded_reviewed_rows')}")
            st.caption(f"{len(no_review_rows):,} {t('uploaded_without_review_rows')}")
        if SAMPLE_CATALOG_PATH.exists():
            st.download_button(
                t("download_sample"),
                data=SAMPLE_CATALOG_PATH.read_bytes(),
                file_name="sample_seller_catalog.csv",
                mime="text/csv",
                use_container_width=True,
            )


def show_overview(df: pd.DataFrame, product_summary: pd.DataFrame) -> None:
    st.subheader(t("overview_title"))
    info_note(t("overview_note"))

    if df.empty:
        st.info(t("no_reviewed_rows"))
        return

    sentiment_counts = df["predicted_sentiment"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    metric_row_1 = st.columns(4)
    metric_row_1[0].metric(t("total_reviews"), f"{len(df):,}")
    metric_row_1[1].metric(t("reviewed_product_count"), f"{df['product_id'].nunique():,}")
    metric_row_1[2].metric(t("avg_rating"), f"{df['star_rating'].mean():.2f}")
    metric_row_1[3].metric(t("avg_reliability"), f"{df['reliability_score'].mean():.1f}/100")

    metric_row_2 = st.columns(4)
    metric_row_2[0].metric(t("positive_reviews"), f"{int(sentiment_counts['positive']):,}")
    metric_row_2[1].metric(t("neutral_reviews"), f"{int(sentiment_counts['neutral']):,}")
    metric_row_2[2].metric(t("negative_reviews"), f"{int(sentiment_counts['negative']):,}")
    high_risk_count = int((product_summary["risk_label"] == "High Risk").sum())
    metric_row_2[3].metric(t("high_risk"), f"{high_risk_count:,}")

    chart_cols = st.columns(3)
    with chart_cols[0]:
        st.caption(t("sentiment_dist"))
        st.bar_chart(localized_sentiment_counts(df), color="#2563eb")
    with chart_cols[1]:
        st.caption(t("rating_dist"))
        st.bar_chart(localized_rating_counts(df), color="#10b981")
    with chart_cols[2]:
        st.caption(t("length_dist"))
        st.bar_chart(
            make_review_length_histogram(df).rename(columns={"count": t("review_count")}),
            color="#f59e0b",
        )


def show_reviewed_products(product_summary: pd.DataFrame) -> None:
    st.subheader(t("reviewed_products"))
    info_note(t("reviewed_products_note"))

    if product_summary.empty:
        st.info(t("no_reviewed_rows"))
        return

    categories = [t("all")] + sorted(product_summary["product_category"].dropna().astype(str).unique().tolist())
    selected_category = st.selectbox(t("product_category"), categories, key="reviewed_category")
    max_count = int(product_summary["review_count"].max()) if not product_summary.empty else 1
    min_count = st.slider(t("min_review_count"), min_value=1, max_value=max_count, value=1)
    risk_options = [t("all"), t("high_risk"), t("medium_risk"), t("low_risk")]
    risk_filter = st.selectbox(t("risk_filter"), risk_options)

    reverse_risk = {risk_label(key): key for key in RISK_LABELS}
    filtered = product_summary[product_summary["review_count"] >= min_count].copy()
    if selected_category != t("all"):
        filtered = filtered[filtered["product_category"] == selected_category]
    if risk_filter != t("all"):
        filtered = filtered[filtered["risk_label"] == reverse_risk[risk_filter]]

    risk_counts = filtered["risk_label"].value_counts()
    risk_cols = st.columns(4)
    risk_cols[0].metric(t("products_after_filters"), f"{len(filtered):,}")
    risk_cols[1].metric(t("high_risk"), f"{int(risk_counts.get('High Risk', 0)):,}")
    risk_cols[2].metric(t("medium_risk"), f"{int(risk_counts.get('Medium Risk', 0)):,}")
    risk_cols[3].metric(t("low_risk"), f"{int(risk_counts.get('Low Risk', 0)):,}")

    st.dataframe(
        display_product_summary(
            filtered[
                [
                    "product_id",
                    "product_title",
                    "product_category",
                    "review_count",
                    "average_rating",
                    "positive_count",
                    "neutral_count",
                    "negative_count",
                    "positive_ratio",
                    "neutral_ratio",
                    "negative_ratio",
                    "average_reliability_score",
                    "risk_label",
                ]
            ]
        ),
        use_container_width=True,
        hide_index=True,
        height=560,
    )


def show_product_detail(df: pd.DataFrame, product_summary: pd.DataFrame) -> None:
    st.subheader(t("product_detail"))
    info_note(t("product_detail_note"))

    if product_summary.empty:
        st.info("No reviewed products are available." if LANG == "en" else "Yorum alan ürün bulunamadı.")
        return

    options = product_summary.sort_values("review_count", ascending=False).copy()
    options["label"] = (
        options["product_title"].str.slice(0, 90)
        + " | "
        + options["product_id"].astype(str)
        + " | "
        + options["review_count"].astype(str)
        + (" reviews" if LANG == "en" else " yorum")
    )
    selected_label = st.selectbox(t("select_product"), options["label"].tolist())
    selected = options[options["label"] == selected_label].iloc[0]
    product_reviews = df[df["product_id"] == selected["product_id"]].copy()

    st.markdown(
        f"""
        <div class="insight-strip">
            <strong>{selected['product_title']}</strong><br>
            product_id: {selected['product_id']} &nbsp; {risk_badge(selected['risk_label'])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    detail_cols = st.columns(5)
    detail_cols[0].metric(t("category"), selected["product_category"])
    detail_cols[1].metric(t("review_count"), f"{int(selected['review_count']):,}")
    detail_cols[2].metric(t("avg_rating"), f"{selected['average_rating']:.2f}")
    detail_cols[3].metric(t("avg_reliability"), f"{selected['average_reliability_score']:.1f}/100")
    detail_cols[4].metric(t("negative_ratio"), format_percent(selected["negative_ratio"]))

    left, right = st.columns([1.05, 1])
    with left:
        st.caption(t("sentiment_dist"))
        st.bar_chart(localized_sentiment_counts(product_reviews), color="#2563eb")
    with right:
        st.caption(t("complaint_terms"))
        complaints = complaint_summary_for_scope(product_reviews, top_n=10)
        if complaints.empty:
            st.info(t("no_complaints_product"))
        else:
            st.dataframe(complaints, use_container_width=True, hide_index=True)

    review_columns = [
        "review_date",
        "star_rating",
        "review_headline",
        "review_body",
        "predicted_sentiment",
        "sentiment_confidence",
        "reliability_score",
        "weighted_confidence",
    ]
    st.caption(t("reviews_for_product"))
    st.dataframe(
        display_reviews(product_reviews[review_columns].sort_values("review_date", ascending=False)),
        use_container_width=True,
        hide_index=True,
        height=520,
    )


def show_top_complaints(df: pd.DataFrame) -> None:
    st.subheader(t("top_complaints"))
    st.markdown(f'<div class="insight-strip">{t("top_complaints_note")}</div>', unsafe_allow_html=True)

    if df.empty:
        st.info(t("no_reviewed_rows"))
        return

    categories = [t("all")] + sorted(df["product_category"].dropna().astype(str).unique().tolist())
    selected_category = st.selectbox(t("product_category"), categories, key="complaint_category")
    category = None if selected_category == t("all") else selected_category
    scope = df if category is None else df[df["product_category"] == category]
    negative_scope = scope[scope["predicted_sentiment"] == "negative"].copy()

    st.metric(t("negative_reviews_scope"), f"{len(negative_scope):,}")
    complaints = complaint_summary_for_scope(scope, category=category, top_n=10)
    if complaints.empty:
        st.info(t("no_complaints_scope"))
    else:
        st.dataframe(complaints, use_container_width=True, hide_index=True)

    st.caption(t("example_negative_reviews"))
    examples = negative_scope[
        ["product_title", "review_date", "star_rating", "review_headline", "review_body", "sentiment_confidence"]
    ].sort_values("sentiment_confidence", ascending=False).head(10)
    st.dataframe(display_reviews(examples), use_container_width=True, hide_index=True)


def show_products_without_reviews(
    df: pd.DataFrame,
    dataset_source: str,
    uploaded_without_reviews: pd.DataFrame | None = None,
) -> None:
    st.subheader(t("without_reviews"))
    st.markdown(f'<div class="insight-strip">{t("without_reviews_note")}</div>', unsafe_allow_html=True)

    if dataset_source == DATASET_SOURCE_UPLOADED:
        st.markdown(f'<div class="insight-strip">{t("uploaded_products_note")}</div>', unsafe_allow_html=True)
        no_review_rows = uploaded_without_reviews if uploaded_without_reviews is not None else pd.DataFrame()
        reviewed_products = df[["product_id", "product_title", "product_category"]].drop_duplicates()

        if no_review_rows.empty:
            without_review_products = pd.DataFrame(columns=["product_id", "product_title", "product_category"])
        else:
            no_review_rows = no_review_rows.copy()
            no_review_rows["product_id"] = no_review_rows["product_id"].astype(str)
            reviewed_ids = set(reviewed_products["product_id"].astype(str))
            without_review_products = (
                no_review_rows[~no_review_rows["product_id"].isin(reviewed_ids)]
                [["product_id", "product_title", "product_category"]]
                .drop_duplicates()
            )

        all_ids = set(reviewed_products["product_id"].astype(str))
        all_ids.update(without_review_products["product_id"].astype(str))
        total_products = len(all_ids)
        with_reviews = len(reviewed_products)
        without_reviews = len(without_review_products)
        coverage = with_reviews / total_products if total_products else 0

        cols = st.columns(4)
        cols[0].metric(t("catalog_products"), f"{total_products:,}")
        cols[1].metric(t("with_reviews"), f"{with_reviews:,}")
        cols[2].metric(t("without_reviews_count"), f"{without_reviews:,}")
        cols[3].metric(t("coverage"), f"{coverage:.1%}")

        left, right = st.columns(2)
        with left:
            st.caption(t("products_without_reviews"))
            st.dataframe(without_review_products, use_container_width=True, hide_index=True)
        with right:
            st.caption(t("products_with_reviews"))
            st.dataframe(reviewed_products, use_container_width=True, hide_index=True)
        return

    st.caption(t("expected_catalog"))
    uploaded = st.file_uploader(t("upload_catalog"), type=["csv"], key="seller_catalog_upload")

    if uploaded is None:
        st.info(t("upload_needed"))
        sample = pd.DataFrame(
            {
                "product_id": ["B00428R89M", "DEMO_NO_REVIEW_001"],
                "product_title": [
                    "Known reviewed product",
                    "Demo product without reviews",
                ],
                "product_category": ["Electronics", "Electronics"],
            }
        )
        st.caption(t("sample_format"))
        st.dataframe(sample, use_container_width=True, hide_index=True)
        if SAMPLE_CATALOG_PATH.exists():
            st.write(f"{t('sample_available')} `{SAMPLE_CATALOG_PATH.relative_to(PROJECT_ROOT)}`")
            st.download_button(
                t("download_sample"),
                data=SAMPLE_CATALOG_PATH.read_bytes(),
                file_name="sample_seller_catalog.csv",
                mime="text/csv",
            )
        return

    catalog = pd.read_csv(uploaded)
    required = {"product_id", "product_title", "product_category"}
    missing = required.difference(catalog.columns)
    if missing:
        st.error(f"{t('catalog_missing')} {sorted(missing)}")
        return

    catalog["product_id"] = catalog["product_id"].astype(str)
    reviewed_ids = set(df["product_id"].astype(str))
    catalog["has_reviews"] = catalog["product_id"].isin(reviewed_ids)

    total_catalog = len(catalog)
    with_reviews = int(catalog["has_reviews"].sum())
    without_reviews = total_catalog - with_reviews
    coverage = with_reviews / total_catalog if total_catalog else 0

    cols = st.columns(4)
    cols[0].metric(t("catalog_products"), f"{total_catalog:,}")
    cols[1].metric(t("with_reviews"), f"{with_reviews:,}")
    cols[2].metric(t("without_reviews_count"), f"{without_reviews:,}")
    cols[3].metric(t("coverage"), f"{coverage:.1%}")

    left, right = st.columns(2)
    with left:
        st.caption(t("products_without_reviews"))
        st.dataframe(catalog[~catalog["has_reviews"]], use_container_width=True, hide_index=True)
    with right:
        st.caption(t("products_with_reviews"))
        st.dataframe(catalog[catalog["has_reviews"]], use_container_width=True, hide_index=True)


def show_single_review_test(model) -> None:
    st.subheader(t("single_test"))
    info_note(t("single_note"))

    review_text = st.text_area(t("review_text"), height=180)
    rating = st.slider(t("rating"), min_value=1, max_value=5, value=4)
    review_age_days = st.number_input(t("review_age"), min_value=0, value=30, step=1)

    if st.button(t("analyze"), type="primary"):
        if not review_text.strip():
            st.warning(t("enter_review"))
            return

        review_length = len(review_text.split())
        processed_text = preprocess_text(review_text)
        prediction = model.predict([processed_text])[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([processed_text])[0]
            confidence = float(probabilities.max())

        reliability_score = compute_reliability_score(
            rating=float(rating),
            review_length=float(review_length),
            review_age_days=float(review_age_days),
        )
        weighted_confidence = confidence * (reliability_score / 100) if confidence is not None else None

        result_cols = st.columns(4)
        result_cols[0].metric(t("predicted_sentiment"), sentiment_label(prediction))
        if confidence is not None:
            result_cols[1].metric(t("model_confidence"), f"{confidence:.2%}")
        result_cols[2].metric(t("fuzzy_reliability"), f"{reliability_score:.1f}/100")
        if weighted_confidence is not None:
            result_cols[3].metric(t("weighted_confidence"), f"{weighted_confidence:.2%}")


def main() -> None:
    show_hero()
    dataset_source, uploaded_file = select_dataset_source()
    show_active_source_notice(dataset_source)

    if not MODEL_PATH.exists():
        show_missing_file_message([MODEL_PATH])
        return

    model = load_model(str(MODEL_PATH))
    uploaded_without_reviews = None
    uploaded_export = None

    if dataset_source == DATASET_SOURCE_DEFAULT:
        if not CLEAN_DATA_PATH.exists():
            show_missing_file_message([CLEAN_DATA_PATH])
            return
        raw_reviews = load_clean_reviews(str(CLEAN_DATA_PATH))
    else:
        if uploaded_file is None:
            st.info(t("upload_test_needed"))
            return
        try:
            uploaded_raw = load_uploaded_dataset(uploaded_file.getvalue())
        except Exception as exc:
            st.error(f"CSV could not be read: {exc}")
            return

        raw_reviews, uploaded_without_reviews, uploaded_all_rows, missing_columns = prepare_uploaded_dataset(uploaded_raw)
        if missing_columns:
            st.error(f"{t('uploaded_missing')} {missing_columns}")
            return

    reviews = add_model_predictions(
        raw_reviews,
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )
    if dataset_source == DATASET_SOURCE_UPLOADED:
        uploaded_export = build_uploaded_export(uploaded_all_rows, reviews)

    product_summary = build_product_summary(reviews)
    show_sidebar_status(
        reviews,
        uploaded_without_reviews if dataset_source == DATASET_SOURCE_UPLOADED else None,
    )
    if uploaded_export is not None:
        st.sidebar.download_button(
            t("download_uploaded_analysis"),
            data=uploaded_export.to_csv(index=False).encode("utf-8-sig"),
            file_name="analyzed_uploaded_test_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    tabs = st.tabs(
        [
            t("overview"),
            t("reviewed_products"),
            t("product_detail"),
            t("top_complaints"),
            t("without_reviews"),
            t("single_test"),
        ]
    )

    with tabs[0]:
        show_overview(reviews, product_summary)
    with tabs[1]:
        show_reviewed_products(product_summary)
    with tabs[2]:
        show_product_detail(reviews, product_summary)
    with tabs[3]:
        show_top_complaints(reviews)
    with tabs[4]:
        show_products_without_reviews(reviews, dataset_source, uploaded_without_reviews)
    with tabs[5]:
        show_single_review_test(model)


if __name__ == "__main__":
    main()
