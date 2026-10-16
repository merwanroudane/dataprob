"""بناء Pipeline كامل — الدرس الختامي."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

from core import charts, ui
from core.anim import play_animation
from core.datasets import messy_customers

ui.page_header(
    "بناء Pipeline كامل",
    "كل ما سبق في مكان واحد: من جدول «وسخ» إلى نموذج مُقيَّم بنزاهة وقابل "
    "للحفظ والنشر — دون سطر واحد يسرّب معلومة.",
    icon=":material/account_tree:",
    badges=[
        ":blue-badge[ColumnTransformer]",
        ":green-badge[Custom transformer]",
        ":orange-badge[Cross-validation]",
        ":violet-badge[joblib]",
        ":red-badge[Zero leakage]",
    ],
)

LEAKY_COLS = ["retention_call_done"]
ID_COLS = ["customer_id", "email"]
CONST_COLS = ["country"]
RAW_DROP = ["income_text", "signup_date", "churn"]


@st.cache_data(ttl=1800, max_entries=4)
def structural_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """تنظيف حتمي يسبق التقسيم — لا يتعلّم أي معامل من البيانات."""
    log: dict[str, int | str] = {"rows_in": len(df)}
    d = df.copy()

    n0 = len(d)
    d = d.drop_duplicates()
    log["exact_duplicates_removed"] = n0 - len(d)

    # توحيد الفئات
    d["city"] = (
        d["city"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip().str.title()
    ).replace({"Alger": "Algiers"})

    # تاريخ ← أقدمية
    d["signup_date"] = pd.to_datetime(d["signup_date"], format="%d/%m/%Y", errors="coerce")
    d["tenure_days"] = (pd.Timestamp("2025-06-30") - d["signup_date"]).dt.days
    d["signup_month_sin"] = np.sin(2 * np.pi * d["signup_date"].dt.month / 12)
    d["signup_month_cos"] = np.cos(2 * np.pi * d["signup_date"].dt.month / 12)

    # رموز فقد مقنّعة
    n_sentinel = int((d["satisfaction"] < 0).sum())
    d.loc[d["satisfaction"] < 0, "satisfaction"] = np.nan
    log["sentinel_values_fixed"] = n_sentinel

    # قيم مستحيلة ⇒ مفقودة
    n_bad_age = int((~d["age"].between(16, 110)).sum())
    d.loc[~d["age"].between(16, 110), "age"] = np.nan
    log["impossible_ages_to_nan"] = n_bad_age

    # متغيّرات مشتقّة من بنية البيانات
    d["n_missing_row"] = d.isna().sum(axis=1)

    # أعمدة تُحذف ولماذا
    d = d.drop(columns=ID_COLS + CONST_COLS + LEAKY_COLS, errors="ignore")
    log["dropped_columns"] = ", ".join(ID_COLS + CONST_COLS + LEAKY_COLS)
    log["rows_out"] = len(d)
    return d, log


NUM_COLS = ["age", "experience_years", "income", "spending", "satisfaction",
            "tenure_days", "signup_month_sin", "signup_month_cos", "n_missing_row"]
CAT_COLS = ["education", "city"]


def build_pipeline(model: str = "HistGB") -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01,
                    sparse_output=False,
                ),
            ),
        ]
    )
    pre = ColumnTransformer(
        [("num", numeric, NUM_COLS), ("cat", categorical, CAT_COLS)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    clf = (
        HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=0)
        if model == "HistGB"
        else LogisticRegression(max_iter=2000, class_weight="balanced")
    )
    return Pipeline([("pre", pre), ("clf", clf)])


@st.cache_data(ttl=1800, max_entries=8, show_spinner="جارٍ تشغيل الـ Pipeline كاملًا…")
def run_everything(model: str = "HistGB") -> dict:
    raw = messy_customers()
    clean, log = structural_clean(raw)

    y = clean["churn"]
    X = clean.drop(columns=[c for c in RAW_DROP if c in clean.columns])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = build_pipeline(model)
    cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc")
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    feat_names = pipe.named_steps["pre"].get_feature_names_out()
    X_tr_t = pipe.named_steps["pre"].transform(X_tr)

    return {
        "log": log,
        "raw_shape": raw.shape,
        "clean_shape": clean.shape,
        "train_shape": X_tr.shape,
        "test_shape": X_te.shape,
        "matrix_shape": X_tr_t.shape,
        "n_missing_raw": int(raw.isna().sum().sum()),
        "n_missing_matrix": int(np.isnan(X_tr_t).sum()),
        "cv_mean": float(cv.mean()),
        "cv_std": float(cv.std()),
        "test_auc": float(roc_auc_score(y_te, proba)),
        "report": classification_report(y_te, pred, digits=3, zero_division=0),
        "feature_names": list(feat_names),
        "sample_row": X_te.iloc[[0]].to_dict(orient="records")[0],
        "sample_proba": float(proba[0]),
    }


t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. لماذا Pipeline وليس سطورًا متتابعة؟", divider="blue")
    st.markdown(
        """
| بدون Pipeline | مع Pipeline |
|---|---|
| تتذكّر يدويًا أن تطبّق نفس الخطوات على الاختبار | تُطبَّق آليًا وبنفس الترتيب |
| خطر `fit` على الاختبار قائم دائمًا | **مستحيل بنيويًا**: `fit` يُستدعى على التدريب فقط |
| التحقّق المتقاطع يسرّب عبر المعالجة | كل طيّة تعيد المعالجة من بياناتها وحدها |
| ضبط معاملات المعالجة صعب | `GridSearchCV` يضبط `pre__num__impute__strategy` مباشرة |
| النشر يحتاج إعادة كتابة الكود | `joblib.dump(pipe)` وانتهى الأمر |

الفكرة الجوهرية: `Pipeline` **كائن واحد يتصرّف كنموذج**: له `fit` و
`predict` و `score`. كل ما يدخله يمرّ بنفس المسار دائمًا.
"""
    )

    st.subheader("٢. تشريح البنية", divider="green")
    st.markdown(
        """
```mermaid
flowchart TD
    A[Raw DataFrame] --> B[Structural cleaning - a plain function, before the split]
    B --> C[train_test_split]
    C --> D[Pipeline]
    D --> E[ColumnTransformer]
    E --> F[numeric branch: impute + indicator + power transform]
    E --> G[categorical branch: impute Missing + one-hot with min_frequency]
    F --> H[Estimator]
    G --> H
    H --> I[cross_val_score on train]
    H --> J[fit on full train]
    J --> K[predict on test - once]
    J --> L[joblib.dump for production]
```

**ما الذي يبقى خارج الـ Pipeline؟** التنظيف الحتمي فقط: حذف التكرار،
تصحيح الأنواع، توحيد الفئات، استبدال رموز الفقد، حذف أعمدة المعرّفات
والأعمدة المسرّبة. كلها عمليات لا تقرأ أي إحصاء من البيانات.
"""
    )

    st.subheader("٣. الأخطاء الشائعة في بناء Pipelines", divider="red")
    st.markdown(
        """
1. **ترتيب خاطئ داخل الفرع:** التقييس قبل التعويض يعني حساب المتوسط على
   أعمدة فيها `NaN`. الترتيب الصحيح: تعويض ← تحويل ← تقييس.
2. **`remainder="passthrough"` بلا انتباه:** يمرّر أعمدة لم تعالَج (ومنها
   قد يتسلّل عمود نصّي أو معرّف).
3. **إعادة المعاينة داخل `sklearn.pipeline.Pipeline`:** لن تعمل كما تظن؛
   استعمل `imblearn.pipeline.Pipeline`.
4. **نسيان `handle_unknown`:** الكود يعمل في الـ notebook وينهار في
   الإنتاج مع أول فئة جديدة.
5. **فقدان أسماء الأعمدة:** استعمل `set_output(transform="pandas")` أو
   `get_feature_names_out()` لتبقى قادرًا على تفسير النموذج.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("الكود الكامل — انسخه كنقطة بداية لأي مشروع", divider="blue")
    ui.code(
        '''
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

# ==================== 1) تنظيف حتمي (قبل التقسيم) ====================
def structural_clean(df: pd.DataFrame) -> pd.DataFrame:
    d = df.drop_duplicates().copy()

    d["city"] = (d["city"].astype("string")
                   .str.replace(r"\\s+", " ", regex=True).str.strip().str.title()
                   .replace({"Alger": "Algiers"}))

    d["signup_date"] = pd.to_datetime(d["signup_date"], format="%d/%m/%Y", errors="coerce")
    d["tenure_days"] = (pd.Timestamp("2025-06-30") - d["signup_date"]).dt.days
    d["signup_month_sin"] = np.sin(2 * np.pi * d["signup_date"].dt.month / 12)
    d["signup_month_cos"] = np.cos(2 * np.pi * d["signup_date"].dt.month / 12)

    d.loc[d["satisfaction"] < 0, "satisfaction"] = np.nan        # رمز فقد مقنّع
    d.loc[~d["age"].between(16, 110), "age"] = np.nan            # قيم مستحيلة
    d["n_missing_row"] = d.isna().sum(axis=1)                    # متغيّر بنيوي

    return d.drop(columns=["customer_id", "email", "country",
                           "retention_call_done"],              # معرّفات/ثابت/مُسرَّب
                  errors="ignore")

df = structural_clean(pd.read_csv("customers.csv"))

# ==================== 2) التقسيم (مبكّرًا) ====================
y = df["churn"]
X = df.drop(columns=["churn", "signup_date", "income_text"], errors="ignore")
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ==================== 3) كل ما يُتعلَّم داخل Pipeline ====================
NUM = ["age", "experience_years", "income", "spending", "satisfaction",
       "tenure_days", "signup_month_sin", "signup_month_cos", "n_missing_row"]
CAT = ["education", "city"]

numeric = Pipeline([
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("power",  PowerTransformer(method="yeo-johnson", standardize=True)),
])

categorical = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                             min_frequency=0.01, sparse_output=False)),
])

pre = ColumnTransformer(
    [("num", numeric, NUM), ("cat", categorical, CAT)],
    remainder="drop", verbose_feature_names_out=False,
)

pipe = Pipeline([
    ("pre", pre),
    ("clf", HistGradientBoostingClassifier(max_iter=200, random_state=0)),
])

# ==================== 4) تقييم نزيه ====================
cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc")
print(f"CV ROC-AUC = {cv.mean():.3f} ± {cv.std():.3f}")

# ==================== 5) ضبط معاملات المعالجة والنموذج معًا ====================
grid = GridSearchCV(
    pipe,
    {
        "pre__num__impute__strategy": ["median", "mean"],
        "pre__cat__encode__min_frequency": [0.01, 0.05],
        "clf__max_iter": [120, 250],
        "clf__learning_rate": [0.05, 0.1],
    },
    cv=5, scoring="roc_auc", n_jobs=-1,
)
grid.fit(X_tr, y_tr)
print(grid.best_params_, grid.best_score_)

# ==================== 6) الاختبار يُفتح مرّة واحدة ====================
best = grid.best_estimator_
print("Test ROC-AUC:", roc_auc_score(y_te, best.predict_proba(X_te)[:, 1]))

# ==================== 7) الحفظ والنشر ====================
joblib.dump(best, "churn_pipeline.joblib")

model = joblib.load("churn_pipeline.joblib")
new_row = pd.DataFrame([{...}])                 # صفّ خام واحد
model.predict_proba(structural_clean(new_row))  # نفس التنظيف ثم نفس المعالجة
''',
        caption="الخطوة ٧ هي الاختبار الحقيقي للتصميم: إن استطعت التنبّؤ بصفّ "
        "خام واحد بسطرين، فالـ Pipeline سليم.",
    )

    st.subheader("محوّل مخصّص حين لا تكفي الأدوات الجاهزة", divider="green")
    ui.code(
        '''
from sklearn.base import BaseEstimator, TransformerMixin

class RatioFeatures(BaseEstimator, TransformerMixin):
    """يضيف نسبًا مشتقّة — بلا أي معامل مُتعلَّم، لكنه يندمج في Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["burn_rate"] = X["spending"] / X["income"].replace(0, np.nan)
        X["income_per_year_exp"] = X["income"] / (X["experience_years"] + 1)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(list(input_features) + ["burn_rate", "income_per_year_exp"])
''',
        caption="ثلاث دوال فقط: `fit` و `transform` و (اختياريًا) "
        "`get_feature_names_out`. بعدها يصبح جزءًا كاملًا من المنظومة.",
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("رحلة البيانات من الخام إلى المصفوفة", divider="violet")

    model_choice = st.segmented_control(
        "النموذج النهائي",
        ["HistGB", "LogisticRegression"],
        default="HistGB",
        key="pipe_model",
    ) or "HistGB"

    r = run_everything(model_choice)

    STAGES = [
        (
            "١. الجدول الخام",
            f"**{r['raw_shape'][0]} صفًا × {r['raw_shape'][1]} عمودًا** — "
            f"فيه {r['n_missing_raw']:,} خلية مفقودة، تكرارات، أعمدة نصّية، "
            "رموز فقد مقنّعة، وعمود مُسرَّب.",
            r["raw_shape"][0],
            r["raw_shape"][1],
        ),
        (
            "٢. بعد التنظيف الحتمي",
            f"حُذف **{r['log']['exact_duplicates_removed']}** صفًا مكرّرًا، "
            f"وصُحِّح **{r['log']['sentinel_values_fixed']}** رمز فقد مقنّع، "
            f"وحُوِّل **{r['log']['impossible_ages_to_nan']}** عمرًا مستحيلًا إلى `NaN`، "
            f"وحُذفت الأعمدة: `{r['log']['dropped_columns']}`. "
            "وأُضيفت سمات التاريخ و `n_missing_row`.",
            r["clean_shape"][0],
            r["clean_shape"][1],
        ),
        (
            "٣. بعد التقسيم",
            f"تدريب **{r['train_shape'][0]}** صفًا، اختبار **{r['test_shape'][0]}** صفًا "
            "(مع `stratify=y`). من الآن فصاعدًا الاختبار مُقفل.",
            r["train_shape"][0],
            r["train_shape"][1],
        ),
        (
            "٤. بعد المعالجة داخل الـ Pipeline",
            f"المصفوفة النهائية **{r['matrix_shape'][0]} × {r['matrix_shape'][1]}** — "
            f"بها **{r['n_missing_matrix']}** قيمة مفقودة. "
            "الزيادة في الأعمدة سببها أعمدة مؤشّر الفقد وترميز One-Hot.",
            r["matrix_shape"][0],
            r["matrix_shape"][1],
        ),
        (
            "٥. التقييم النزيه",
            f"ROC-AUC بتحقّق متقاطع خماسي على التدريب: "
            f"**{r['cv_mean']:.3f} ± {r['cv_std']:.3f}**. "
            f"وعلى الاختبار (فُتح مرّة واحدة): **{r['test_auc']:.3f}**. "
            "تقارب الرقمين علامة صحّة.",
            r["train_shape"][0],
            r["matrix_shape"][1],
        ),
    ]

    def _render_stage(frame: int) -> None:
        title, note, rows, cols = STAGES[frame]
        k1, k2, k3 = st.columns(3)
        k1.metric("الصفوف", f"{rows:,}")
        k2.metric("الأعمدة", f"{cols:,}")
        k3.metric("المرحلة", f"{frame + 1}/{len(STAGES)}")

        shape_df = pd.DataFrame(
            {
                "stage": [s[0] for s in STAGES[: frame + 1]],
                "rows": [s[2] for s in STAGES[: frame + 1]],
                "columns": [s[3] for s in STAGES[: frame + 1]],
            }
        )
        ch = (
            alt.Chart(shape_df.melt(id_vars="stage", var_name="dim", value_name="n"))
            .mark_bar()
            .encode(
                y=alt.Y("stage:N", sort=None, title=None),
                x=alt.X("n:Q", title=None),
                color=alt.Color("dim:N", legend=alt.Legend(orient="top", title=None)),
                yOffset="dim:N",
                tooltip=["stage", "dim", "n"],
            )
            .properties(height=max(160, 55 * (frame + 1)))
        )
        st.altair_chart(ch, width="stretch")
        with st.container(border=True):
            st.markdown(f"##### :violet-badge[{title}]")
            st.markdown(note)

    play_animation(
        "pipeline_journey",
        len(STAGES),
        _render_stage,
        fps=0.5,
        label="المرحلة",
        frame_labels=[s[0] for s in STAGES],
    )

    st.subheader("النتائج التفصيلية", divider="orange")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.metric("CV ROC-AUC (تدريب)", f"{r['cv_mean']:.3f}", delta=f"± {r['cv_std']:.3f}", delta_color="off")
        st.metric("Test ROC-AUC", f"{r['test_auc']:.3f}")
        st.metric("عدد المتغيّرات النهائية", len(r["feature_names"]))
    with c2:
        st.markdown("###### تقرير التصنيف على الاختبار")
        st.code(r["report"], language="text")

    with st.expander("أسماء المتغيّرات بعد المعالجة", icon=":material/list:"):
        st.dataframe(
            pd.DataFrame({"feature": r["feature_names"]}),
            width="stretch",
            hide_index=True,
            height=260,
        )

    with st.expander("تنبّؤ على صفّ واحد (كما في الإنتاج)", icon=":material/bolt:"):
        st.json(r["sample_row"])
        st.metric("احتمال المغادرة المتوقّع", f"{r['sample_proba']:.3f}")
        st.caption(
            "لاحظ أن الصف مرّ بنفس التنظيف ونفس المعاملات المُتعلَّمة من "
            "التدريب — وهذا بالضبط ما يضمنه الـ Pipeline."
        )

# ============================================================== القرار
with t_decide:
    st.subheader("قائمة فحص الـ Pipeline", divider="orange")
    ui.decision_table(
        [
            ("حذف تكرار، تصحيح أنواع، توحيد فئات", "دالة تنظيف **قبل** التقسيم", "حتمية، لا تتعلّم معاملات"),
            ("حذف المعرّفات والأعمدة الثابتة والمسرّبة", "قبل التقسيم", "قرار معرفي لا إحصائي"),
            ("تعويض الفقد", "داخل Pipeline", "الوسيط/المتوسط معامل مُتعلَّم"),
            ("التقييس وتحويل التوزيع", "داخل Pipeline، **بعد** التعويض", "لا يمكن حساب إحصاء على NaN"),
            ("الترميز الفئوي", "داخل Pipeline مع `handle_unknown`", "قائمة الفئات مُتعلَّمة"),
            ("اختيار المتغيّرات", "داخل Pipeline", "يستعمل `y` ⇒ مُتعلَّم"),
            ("إعادة المعاينة", "`imblearn.pipeline` فقط", "يجب أن تُطبَّق في `fit` دون `transform`"),
            ("ضبط المعاملات", "`GridSearchCV` على الـ Pipeline كاملًا", "يضبط المعالجة والنموذج معًا بنزاهة"),
            ("الحفظ", "`joblib.dump(pipeline)`", "النموذج والمعالجة كائن واحد"),
            ("الإنتاج", "نفس دالة التنظيف + `pipeline.predict`", "ضمان تطابق التدريب والتنفيذ"),
        ],
        headers=("الخطوة", "أين؟", "لماذا؟"),
    )

    ui.takeaways(
        [
            "التنظيف الحتمي خارج الـ Pipeline، وكل ما يُتعلَّم داخله.",
            "`ColumnTransformer` يسمح بمسار مختلف لكل نوع من الأعمدة.",
            "`GridSearchCV` على الـ Pipeline يضبط المعالجة والنموذج معًا.",
            "`joblib.dump` يحفظ كل شيء — لا تحفظ النموذج وحده أبدًا.",
            "إن لم تستطع التنبّؤ بصفّ خام واحد، فالتصميم غير مكتمل.",
        ]
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("scikit-learn — Pipelines and composite estimators", "https://scikit-learn.org/stable/modules/compose.html"),
            ("scikit-learn — ColumnTransformer with Mixed Types", "https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html"),
            ("scikit-learn — Common pitfalls and recommended practices", "https://scikit-learn.org/stable/common_pitfalls.html"),
            ("scikit-learn — Model persistence", "https://scikit-learn.org/stable/model_persistence.html"),
            ("scikit-learn — Developing custom estimators", "https://scikit-learn.org/stable/developers/develop.html"),
            ("scikit-learn — set_output API", "https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html"),
            ("imbalanced-learn — Pipeline", "https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html"),
        ]
    )
    ui.terms(
        [
            ("Pipeline", "سلسلة خطوات تُعامَل كنموذج واحد"),
            ("ColumnTransformer", "معالجات مختلفة لمجموعات أعمدة مختلفة"),
            ("Custom transformer", "محوّل مخصّص"),
            ("get_feature_names_out", "أسماء المتغيّرات بعد المعالجة"),
            ("Model persistence", "حفظ النموذج للنشر"),
            ("GridSearchCV", "بحث شبكي بتحقّق متقاطع"),
        ]
    )
