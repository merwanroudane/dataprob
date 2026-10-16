"""أثر الفقد وطريقة معالجته على الخوارزميات."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core import charts, ui
from core.anim import play_animation
from core.datasets import missing_demo

ui.page_header(
    "أثر الفقد على الخوارزميات",
    "ليست كل الخوارزميات متساوية أمام الفقد: بعضها **يرفض** العمل، وبعضها "
    "يتعامل معه أصلًا، وبعضها ينهار بصمت. وهنا نقيس ذلك تجريبيًا على مهمّة "
    "تصنيف حقيقية.",
    icon=":material/psychology:",
    badges=[":blue-badge[Native NaN]", ":green-badge[HistGB]", ":orange-badge[Linear]", ":red-badge[Distance-based]"],
)

NUM = ["age", "experience_years", "satisfaction", "spending", "income"]
CAT = ["education", "city"]


@st.cache_data(ttl=1800, max_entries=16, show_spinner="جارٍ تشغيل التجربة…")
def experiment(mechanism: str, rate: float, n: int = 600) -> pd.DataFrame:
    d = missing_demo(mechanism, rate, n)
    X = d[NUM + CAT]
    y = d["churn"]

    def prep(num_steps, cat_steps=None):
        cat_steps = cat_steps or [
            ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False)),
        ]
        return ColumnTransformer(
            [("num", Pipeline(num_steps), NUM), ("cat", Pipeline(cat_steps), CAT)]
        )

    median = ("imp", SimpleImputer(strategy="median"))
    median_ind = ("imp", SimpleImputer(strategy="median", add_indicator=True))
    mice = ("imp", IterativeImputer(max_iter=10, random_state=0))
    scale = ("sc", StandardScaler())

    setups = {
        "LogReg + حذف الصفوف": ("drop", Pipeline([("pre", prep([median, scale])), ("m", LogisticRegression(max_iter=1000))])),
        "LogReg + median": ("full", Pipeline([("pre", prep([median, scale])), ("m", LogisticRegression(max_iter=1000))])),
        "LogReg + median + indicator": ("full", Pipeline([("pre", prep([median_ind, scale])), ("m", LogisticRegression(max_iter=1000))])),
        "LogReg + MICE": ("full", Pipeline([("pre", prep([mice, scale])), ("m", LogisticRegression(max_iter=1000))])),
        "RandomForest + median": ("full", Pipeline([("pre", prep([median])), ("m", RandomForestClassifier(n_estimators=200, random_state=0))])),
        "HistGB + NaN أصلية": ("native", None),
    }

    rows = []
    for name, (mode, pipe) in setups.items():
        if mode == "drop":
            keep = X.notna().all(axis=1)
            Xi, yi = X[keep], y[keep]
            n_used = int(keep.sum())
        elif mode == "native":
            Xi = X.copy()
            for c in CAT:
                Xi[c] = Xi[c].astype("category")
            yi = y
            n_used = len(X)
            pipe = HistGradientBoostingClassifier(
                max_iter=80, categorical_features="from_dtype", random_state=0
            )
        else:
            Xi, yi = X, y
            n_used = len(X)

        if yi.nunique() < 2 or len(yi) < 50:
            continue
        sc = cross_val_score(pipe, Xi, yi, cv=4, scoring="roc_auc", n_jobs=1)
        rows.append(
            {
                "setup": name,
                "roc_auc": float(sc.mean()),
                "std": float(sc.std()),
                "rows_used": n_used,
                "pct_rows": n_used / len(X) * 100,
            }
        )
    return pd.DataFrame(rows)


t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. من يقبل `NaN` ومن يرفضها؟ (scikit-learn)", divider="blue")
    st.markdown(
        """
| الخوارزمية | تقبل `NaN` مباشرة؟ | ملاحظة |
|---|---|---|
| `HistGradientBoosting*` | :green-badge[نعم] | تتعلّم اتجاه إرسال المفقود في كل تقسيم |
| `DecisionTree*` | :green-badge[نعم] | منذ الإصدار 1.3 |
| `RandomForest*` / `ExtraTrees*` | :green-badge[نعم] | منذ الإصدار 1.4 |
| `GradientBoosting*` (القديم) | :red-badge[لا] | يحتاج تعويضًا |
| `LogisticRegression` / `Ridge` / `SVM` | :red-badge[لا] | أي `NaN` يرمي استثناءً |
| `KNeighbors*` | :red-badge[لا] | المسافة غير معرّفة مع `NaN` |
| `KMeans` / `PCA` | :red-badge[لا] | تحتاج مصفوفة كاملة |
| `XGBoost` / `LightGBM` / `CatBoost` | :green-badge[نعم] | خارج scikit-learn، دعم أصلي ممتاز |

:orange-badge[مهم] «تقبل `NaN`» لا تعني «تتعامل معها بشكل مثالي». الأشجار
تتعلّم **اتجاهًا افتراضيًا** للقيم المفقودة في كل عقدة — وهذا ممتاز حين
يكون للفقد معنى (MNAR)، لكنه لا يصحّح تحيّز MAR تلقائيًا.
"""
    )

    st.subheader("٢. كيف يتأثّر كل نوع من النماذج؟", divider="green")
    ui.concept_grid(
        [
            (
                "النماذج الخطّية",
                "لا تقبل `NaN` إطلاقًا. والأسوأ: التعويض بالمتوسط يقلّل تباين "
                "المتغيّر ⇒ **يسحب المعامل نحو الصفر** (attenuation bias) "
                "ويضيّق الأخطاء المعيارية كذبًا.",
                "orange",
            ),
            (
                "النماذج القائمة على المسافة (KNN, SVM, K-Means)",
                "المسافة الإقليدية غير معرّفة مع `NaN`. والتعويض بقيمة ثابتة "
                "يخلق **تجمّعًا كاذبًا** عند تلك القيمة يشوّه الجيران والعناقيد.",
                "red",
            ),
            (
                "الأشجار والتجميعات (Trees / RF / GBM)",
                "الأكثر تسامحًا: التقسيم يعتمد على ترتيب لا على مسافة. تستطيع "
                "عزل القيم المفقودة كفرع مستقل — ولذلك يصلح معها التعويض بقيمة "
                "خارج المدى (`-1`, `999`).",
                "green",
            ),
            (
                "الشبكات العصبية",
                "لا تقبل `NaN` (تنتشر في كل الأوزان وتُنتج `NaN` loss). تحتاج "
                "تعويضًا + تقييسًا، وتستفيد كثيرًا من عمود المؤشّر لأنها تستطيع "
                "تعلّم تفاعل معه.",
                "violet",
            ),
            (
                "PCA وتقليل الأبعاد",
                "تتطلّب مصفوفة كاملة. التعويض بالمتوسط هنا يحرف المكوّنات لأنه "
                "يقلّل التباين في الأعمدة الأكثر فقدًا فيبدو مساهمتها أضعف.",
                "blue",
            ),
            (
                "النماذج الزمنية (ARIMA, Prophet)",
                "فجوة في السلسلة تكسر افتراض انتظام الخطوات. تحتاج استيفاءً "
                "يحترم الزمن — ولا تستعمل أبدًا قيمًا من المستقبل.",
                "orange",
            ),
        ],
        n_cols=3,
    )

    st.subheader("٣. الأثر على الاستدلال وليس التنبّؤ فقط", divider="red")
    st.markdown(
        """
- **تضخّم الثقة الكاذب:** التعويض المفرد يعامل القيم المُخترعة كأنها مرصودة
  ⇒ الأخطاء المعيارية أصغر مما ينبغي ⇒ قيم $p$ أصغر ⇒ «اكتشافات» غير حقيقية.
- **تخفيف المعاملات (attenuation):** تقلّص التباين يقلّل معامل الانحدار.
- **تشويه الارتباطات:** التعويض بالمتوسط يسحب كل الارتباطات نحو الصفر؛
  التعويض بنموذج انحداري (بلا ضوضاء) يضخّمها نحو الواحد.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. استغلال الدعم الأصلي للـ NaN", divider="green")
    ui.code(
        """
from sklearn.ensemble import HistGradientBoostingClassifier

# لا تعويض إطلاقًا — النموذج يتعلّم اتجاه المفقود في كل تقسيم
model = HistGradientBoostingClassifier(
    max_iter=300,
    categorical_features="from_dtype",   # الأعمدة dtype=category تُعامَل كفئوية
)
X["city"] = X["city"].astype("category")
model.fit(X, y)          # X فيها NaN ولا مشكلة
""",
        caption="هذا الخيار يُعدّ خطّ الأساس (baseline) القوي لأي جدول بيانات "
        "فيه فقد — جرّبه قبل أي استراتيجية تعويض معقّدة.",
    )

    st.subheader("٢. التعويض بقيمة خارج المدى للأشجار", divider="blue")
    ui.code(
        """
from sklearn.impute import SimpleImputer

# منطقي للأشجار فقط: تعزل المفقود بتقسيم واحد
tree_imp = SimpleImputer(strategy="constant", fill_value=-9999)

# كارثة للنماذج الخطّية/المسافية: -9999 تصبح نقطة بعيدة جدًا تجذب كل شيء
"""
    )

    st.subheader("٣. المؤشّر: أرخص تحسين في كل الدرس", divider="orange")
    ui.code(
        """
SimpleImputer(strategy="median", add_indicator=True)
# أو يدويًا:
for c in cols_with_na:
    df[f"{c}_missing"] = df[c].isna().astype("int8")
"""
    )

    st.subheader("٤. قارن الاستراتيجيات بشكل نزيه", divider="violet")
    ui.code(
        """
from sklearn.model_selection import cross_val_score

for name, pipe in strategies.items():
    s = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"{name:35s} {s.mean():.3f} ± {s.std():.3f}")

# القاعدة: المقارنة داخل CV، وكل خطوات التعويض داخل الـ Pipeline
"""
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("تجربة حيّة: من يفوز؟", divider="violet")
    st.markdown(
        "نفس البيانات، نفس الهدف (`churn`)، وست استراتيجيات مختلفة للتعامل مع "
        "الفقد. المقياس: **ROC-AUC** بتحقّق متقاطع رباعي."
    )

    c1, c2 = st.columns(2)
    with c1:
        mech = st.segmented_control(
            "آلية الفقد", ["MCAR", "MAR", "MNAR"], default="MNAR", key="imp2_mech"
        ) or "MNAR"
    with c2:
        rate = st.select_slider(
            "نسبة الفقد",
            options=[0.15, 0.3, 0.45],
            value=0.3,
            format_func=lambda v: f"{v:.0%}",
            key="imp2_rate",
        )

    res = experiment(mech, rate)

    def _render_bars(frame: int) -> None:
        sofar = res.iloc[: frame + 1]
        row = res.iloc[frame]

        k1, k2, k3 = st.columns(3)
        k1.metric("الاستراتيجية", row["setup"])
        k2.metric("ROC-AUC", f"{row['roc_auc']:.3f}", delta=f"± {row['std']:.3f}", delta_color="off")
        k3.metric("الصفوف المستعملة", f"{int(row['rows_used'])}", delta=f"{row['pct_rows']:.0f}%", delta_color="off")

        ch = charts.bar(
            sofar[["setup", "roc_auc"]], "setup", "roc_auc", horizontal=True, height=300, sort_desc=True
        )
        st.altair_chart(ch, width="stretch")

    play_animation(
        f"impact_{mech}_{rate}",
        len(res),
        _render_bars,
        fps=0.7,
        label="الاستراتيجية",
        frame_labels=res["setup"].tolist(),
    )

    st.dataframe(
        res.assign(roc_auc=res["roc_auc"].round(3), std=res["std"].round(3), pct_rows=res["pct_rows"].round(1)),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "ما يستحقّ ملاحظته: (١) حذف الصفوف يستعمل جزءًا فقط من البيانات؛ "
        "(٢) إضافة `indicator` كثيرًا ما تتفوّق على تعويض أذكى بلا مؤشّر، "
        "خصوصًا تحت MNAR؛ (٣) `HistGB` بدون أي تعويض خطّ أساس قوي. "
        "الفروق صغيرة أحيانًا — وهذا بحدّ ذاته درس: **لا تعقّد المعالجة دون "
        "قياس**."
    )

# ============================================================== القرار
with t_decide:
    st.subheader("الخوارزمية ⇐ استراتيجية الفقد", divider="orange")
    ui.decision_table(
        [
            ("HistGB / LightGBM / XGBoost / CatBoost", "مرّر `NaN` كما هي", "دعم أصلي يتعلّم اتجاه الفقد — وغالبًا أفضل من التعويض"),
            ("RandomForest / DecisionTree (sklearn ≥ 1.4)", "`NaN` مباشرة أو constant خارج المدى", "التقسيم ترتيبي لا مسافي"),
            ("LogisticRegression / Ridge / SVM", "median/MICE + indicator + تقييس", "لا تقبل NaN وحسّاسة للقيم المتطرّفة"),
            ("KNN / K-Means / DBSCAN", "MICE + تقييس إجباري", "المسافة تنهار مع التعويض الثابت"),
            ("الشبكات العصبية", "تعويض + تقييس + indicator", "`NaN` تنتشر في كامل الشبكة"),
            ("PCA / FactorAnalysis", "تعويض يحافظ على التباين (MICE)", "تعويض المتوسط يقلّل أهمية الأعمدة الأكثر فقدًا"),
            ("سلاسل زمنية", "استيفاء زمني أحادي الاتجاه", "منع تسريب المستقبل"),
            ("بحث استدلالي منشور", "MI + قواعد Rubin + تقرير النسب", "شرط المراجعة العلمية"),
        ],
        headers=("الخوارزمية", "الاستراتيجية", "السبب"),
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("scikit-learn — HistGradientBoostingClassifier (دعم NaN)", "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html"),
            ("scikit-learn — Imputation of missing values", "https://scikit-learn.org/stable/modules/impute.html"),
            ("scikit-learn — Release highlights (دعم NaN في الأشجار)", "https://scikit-learn.org/stable/whats_new.html"),
            ("XGBoost — Handling of missing values", "https://xgboost.readthedocs.io/en/stable/faq.html"),
            ("LightGBM — Advanced topics: missing value handle", "https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html"),
            ("scikit-learn — Common pitfalls (leakage)", "https://scikit-learn.org/stable/common_pitfalls.html"),
        ]
    )
    ui.terms(
        [
            ("Native NaN support", "دعم أصلي للقيم المفقودة"),
            ("Attenuation bias", "تخفيف المعامل نحو الصفر"),
            ("Default direction", "الاتجاه الافتراضي لإرسال المفقود في عقدة الشجرة"),
            ("ROC-AUC", "مساحة تحت منحنى ROC"),
            ("Baseline", "خطّ أساس للمقارنة"),
        ]
    )
