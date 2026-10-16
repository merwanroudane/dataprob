"""بيانات تدريبية مُصنّعة (synthetic) تُستخدم في كل الدروس.

لماذا بيانات مُصنّعة؟ لأننا نعرف «الحقيقة» سلفًا: نعرف أيّ قيمة حُذفت وأيّ
نقطة شاذة زُرعت، فنستطيع قياس أثر كل معالجة بدقّة — وهذا مستحيل على بيانات
حقيقية. كل الدوال مُخزّنة بـ `st.cache_data` حتى لا تُعاد الحسابات مع كل
تفاعل.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

SEED = 7


# ---------------------------------------------------------------- بيانات عامة
@st.cache_data(ttl=3600, max_entries=16)
def clean_customers(n: int = 500, seed: int = SEED) -> pd.DataFrame:
    """بيانات عملاء «نظيفة» تمامًا — تمثّل الحقيقة الكاملة (ground truth)."""
    rng = np.random.default_rng(seed)

    age = rng.normal(38, 11, n).clip(18, 80).round(0)
    # الخبرة مرتبطة بالعمر (علاقة سنستغلّها في التعويض بالانحدار)
    experience = ((age - 20) * rng.uniform(0.25, 0.55, n)).clip(0, 45).round(1)
    education = rng.choice(["Secondary", "Bachelor", "Master", "PhD"], n, p=[0.3, 0.42, 0.22, 0.06])
    edu_bonus = pd.Series(education).map(
        {"Secondary": 0, "Bachelor": 1800, "Master": 3600, "PhD": 6200}
    ).to_numpy()
    city = rng.choice(["Algiers", "Oran", "Constantine", "Annaba", "Setif"], n, p=[0.34, 0.24, 0.18, 0.12, 0.12])
    city_bonus = pd.Series(city).map(
        {"Algiers": 2400, "Oran": 1500, "Constantine": 900, "Annaba": 500, "Setif": 300}
    ).to_numpy()

    income = (
        18000
        + 620 * experience
        + 130 * age
        + edu_bonus
        + city_bonus
        + rng.normal(0, 3000, n)
    ).clip(12000, None)

    # الدخل موزّع بانحراف يمين (right-skewed) كما في الواقع
    income = income * rng.lognormal(0, 0.18, n)

    spending = (0.22 * income + rng.normal(0, 1200, n)).clip(500, None)
    satisfaction = np.clip(rng.normal(6.8, 1.6, n) + (income - income.mean()) / income.std() * 0.5, 1, 10)

    churn_logit = (
        -1.1
        - 0.045 * (satisfaction - 6.8) * 8
        - 0.00004 * (income - income.mean())
        + 0.02 * (age - 38)
    )
    churn = rng.binomial(1, 1 / (1 + np.exp(-churn_logit)))

    df = pd.DataFrame(
        {
            "customer_id": [f"C{1000 + i}" for i in range(n)],
            "age": age,
            "experience_years": experience,
            "education": education,
            "city": city,
            "income": income.round(0),
            "spending": spending.round(0),
            "satisfaction": satisfaction.round(1),
            "churn": churn,
        }
    )
    return df


# ------------------------------------------------- زرع القيم المفقودة بالأنواع
def _inject_mcar(s: pd.Series, rate: float, rng: np.random.Generator) -> pd.Series:
    """MCAR: الفقد عشوائي تمامًا — لا يعتمد على أي شيء."""
    out = s.copy()
    mask = rng.random(len(s)) < rate
    out[mask] = np.nan
    return out


def _inject_mar(s: pd.Series, driver: pd.Series, rate: float, rng: np.random.Generator) -> pd.Series:
    """MAR: احتمال الفقد يعتمد على متغيّر آخر **مرصود** (driver)."""
    out = s.copy()
    z = (driver - driver.mean()) / (driver.std() + 1e-9)
    p = 1 / (1 + np.exp(-(z * 2.2 - 0.6)))
    p = p / p.mean() * rate
    mask = rng.random(len(s)) < p.clip(0, 0.95)
    out[mask] = np.nan
    return out


def _inject_mnar(s: pd.Series, rate: float, rng: np.random.Generator) -> pd.Series:
    """MNAR: احتمال الفقد يعتمد على قيمة المتغيّر نفسه (القيم العالية تختفي)."""
    out = s.copy()
    z = (s - s.mean()) / (s.std() + 1e-9)
    p = 1 / (1 + np.exp(-(z * 2.6 - 0.8)))
    p = p / p.mean() * rate
    mask = rng.random(len(s)) < p.clip(0, 0.95)
    out[mask] = np.nan
    return out


@st.cache_data(ttl=3600, max_entries=32)
def missing_demo(
    mechanism: str = "MCAR", rate: float = 0.25, n: int = 500, seed: int = SEED
) -> pd.DataFrame:
    """نسخة من بيانات العملاء بعد زرع فقد في `income` بآلية محدّدة.

    العمود `income_true` يحفظ القيمة الأصلية — نستخدمه لقياس خطأ التعويض.
    """
    rng = np.random.default_rng(seed + 11)
    df = clean_customers(n, seed).copy()
    df["income_true"] = df["income"]

    if mechanism == "MCAR":
        df["income"] = _inject_mcar(df["income"], rate, rng)
    elif mechanism == "MAR":
        df["income"] = _inject_mar(df["income"], df["age"], rate, rng)
    elif mechanism == "MNAR":
        df["income"] = _inject_mnar(df["income"], rate, rng)
    else:  # pragma: no cover
        raise ValueError(f"mechanism غير معروف: {mechanism}")

    return df


# ---------------------------------------------------------- بيانات «وسخة» شاملة
@st.cache_data(ttl=3600, max_entries=8)
def messy_customers(n: int = 420, seed: int = SEED) -> pd.DataFrame:
    """بيانات تحتوي كل مشاكل الواقع في جدول واحد.

    المشاكل المزروعة:
    1. فقد بثلاث آليات مختلفة (MCAR / MAR / MNAR)
    2. قيم شاذة أحادية (univariate) وثنائية (multivariate) وأخطاء إدخال
    3. تكرار تام (exact) وتكرار جزئي (fuzzy / near-duplicate)
    4. تناقض في كتابة الفئات (Algiers / algiers / ALGIERS / Alger)
    5. أعمدة بصيغة نصّية يجب أن تكون رقمية أو تاريخية
    6. عمود ثابت (zero variance) وعمود شبه-معرّف (high cardinality)
    7. عمود مُسرّب (leakage) يحتوي معلومة من المستقبل
    """
    rng = np.random.default_rng(seed + 3)
    df = clean_customers(n, seed).copy()

    # (5) أعمدة بصيغة خاطئة
    df["income_text"] = df["income"].map(lambda v: f"{v:,.0f} DA")
    df["signup_date"] = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 1200, len(df)), unit="D"
    )
    df["signup_date"] = df["signup_date"].dt.strftime("%d/%m/%Y")

    # (4) تناقض كتابة الفئات
    variants = {"Algiers": ["Algiers", "algiers", "ALGIERS", " Algiers ", "Alger"]}
    city = df["city"].to_numpy(dtype=object).copy()
    for i, v in enumerate(city):
        if v == "Algiers":
            city[i] = rng.choice(variants["Algiers"])
    df["city"] = city

    # (6) عمود ثابت + عمود شبه معرّف
    df["country"] = "DZ"
    df["email"] = [f"user{i}_{rng.integers(1000, 9999)}@mail.dz" for i in range(len(df))]

    # (7) تسريب: عمود يُحسب بعد معرفة الهدف
    df["retention_call_done"] = np.where(
        df["churn"] == 1, rng.binomial(1, 0.92, len(df)), rng.binomial(1, 0.05, len(df))
    )

    # (2) قيم شاذة
    out_idx = rng.choice(len(df), 8, replace=False)
    df.loc[out_idx[:3], "income"] = df["income"].max() * rng.uniform(6, 12, 3)  # شاذ أحادي
    df.loc[out_idx[3:5], "age"] = [199, 0]  # خطأ إدخال (data entry error)
    df.loc[out_idx[5:], "spending"] = df.loc[out_idx[5:], "income"] * 3.5  # شاذ ثنائي
    df["age"] = df["age"].astype(float)
    df.loc[rng.choice(len(df), 4, replace=False), "satisfaction"] = -99  # رمز قيمة مفقودة

    # (1) فقد بثلاث آليات
    rng2 = np.random.default_rng(seed + 5)
    df["income"] = _inject_mcar(df["income"], 0.08, rng2)
    df["satisfaction"] = _inject_mar(df["satisfaction"], df["age"], 0.12, rng2)
    df["spending"] = _inject_mnar(df["spending"], 0.10, rng2)
    df.loc[rng2.choice(len(df), 14, replace=False), "education"] = None

    # (3) تكرارات
    exact = df.sample(10, random_state=1)
    near = df.sample(8, random_state=2).copy()
    near["customer_id"] = [f"C{9000 + i}" for i in range(len(near))]
    near["income"] = near["income"] * 1.0001
    df = pd.concat([df, exact, near], ignore_index=True)

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ------------------------------------------------------- بيانات القيم الشاذة
@st.cache_data(ttl=3600, max_entries=16)
def outlier_demo(kind: str = "global", n: int = 300, seed: int = SEED) -> pd.DataFrame:
    """بيانات ثنائية الأبعاد لشرح أنواع الشواذ.

    kind: global | contextual | collective | multivariate
    """
    rng = np.random.default_rng(seed + 17)
    x = rng.normal(50, 10, n)
    y = 2.0 * x + rng.normal(0, 6, n)
    df = pd.DataFrame({"x": x, "y": y, "type": "normal"})

    if kind == "global":
        extra = pd.DataFrame(
            {"x": [120, 125, -15], "y": [240, 250, -30], "type": "global outlier"}
        )
    elif kind == "contextual":
        # قيم طبيعية منفردة، لكنها غير منطقية في سياقها
        extra = pd.DataFrame({"x": [52, 48, 55], "y": [20, 180, 25], "type": "contextual outlier"})
    elif kind == "collective":
        extra = pd.DataFrame(
            {
                "x": rng.normal(75, 1.5, 18),
                "y": rng.normal(95, 1.5, 18),
                "type": "collective outlier",
            }
        )
    else:  # multivariate
        extra = pd.DataFrame(
            {"x": [62, 38, 58, 42], "y": [60, 150, 70, 140], "type": "multivariate outlier"}
        )

    return pd.concat([df, extra], ignore_index=True)


# ------------------------------------------------------ بيانات عدم التوازن
@st.cache_data(ttl=3600, max_entries=16)
def imbalanced_fraud(n: int = 3000, pos_rate: float = 0.02, seed: int = SEED) -> pd.DataFrame:
    """بيانات احتيال غير متوازنة: الفئة الموجبة نادرة."""
    rng = np.random.default_rng(seed + 23)
    n_pos = max(int(n * pos_rate), 5)
    n_neg = n - n_pos

    neg = pd.DataFrame(
        {
            "amount": rng.lognormal(4.0, 0.7, n_neg),
            "night_ratio": rng.beta(2, 6, n_neg),
            "n_devices": rng.poisson(1.2, n_neg),
            "fraud": 0,
        }
    )
    pos = pd.DataFrame(
        {
            "amount": rng.lognormal(5.1, 0.8, n_pos),
            "night_ratio": rng.beta(5, 3, n_pos),
            "n_devices": rng.poisson(3.1, n_pos),
            "fraud": 1,
        }
    )
    return pd.concat([neg, pos], ignore_index=True).sample(frac=1, random_state=5).reset_index(drop=True)


# ------------------------------------------------------------- أدوات مساعدة
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """تقرير فقد لكل عمود: العدد، النسبة، النوع."""
    rep = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "n_missing": df.isna().sum().to_numpy(),
            "pct_missing": (df.isna().mean() * 100).round(2).to_numpy(),
            "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
        }
    )
    return rep.sort_values("pct_missing", ascending=False).reset_index(drop=True)
