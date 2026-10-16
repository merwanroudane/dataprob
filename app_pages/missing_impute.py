"""التعويض Imputation: الطرق، الأثر، والقياس مقابل الحقيقة."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (مطلوب قبل الاستيراد)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge

from core import charts, ui
from core.anim import play_animation
from core.datasets import missing_demo

ui.page_header(
    "التعويض Imputation",
    "ملء القيم المفقودة بقيم معقولة. السؤال ليس «ما أدقّ طريقة؟» بل "
    "**«ما الذي أريد الحفاظ عليه؟»**: المتوسط؟ التباين؟ بنية الارتباط؟ "
    "لأن كل طريقة تحافظ على شيء وتدمّر شيئًا آخر.",
    icon=":material/healing:",
    badges=[
        ":blue-badge[Simple]",
        ":green-badge[KNN]",
        ":orange-badge[Iterative / MICE]",
        ":violet-badge[Indicator]",
    ],
)

NUM_FEATS = ["age", "experience_years", "satisfaction", "spending", "income"]
METHODS = [
    ("Mean", "التعويض بالمتوسط"),
    ("Median", "التعويض بالوسيط"),
    ("Constant (-1)", "قيمة ثابتة خارج المدى"),
    ("Random sample", "سحب عشوائي من القيم المرصودة"),
    ("KNN (k=5)", "متوسط أقرب ٥ جيران"),
    ("Iterative (MICE)", "نمذجة انحدارية تكرارية"),
]


@st.cache_data(ttl=1800, max_entries=32)
def run_imputations(mechanism: str, rate: float, n: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    """يرجع (جدول المقاييس، جدول القيم المعوّضة لكل طريقة)."""
    d = missing_demo(mechanism, rate, n)
    X = d[NUM_FEATS].copy()
    mask = X["income"].isna().to_numpy()
    truth = d.loc[mask, "income_true"].to_numpy()
    true_all = d["income_true"].to_numpy()
    rng = np.random.default_rng(11)

    filled: dict[str, np.ndarray] = {}

    filled["Mean"] = SimpleImputer(strategy="mean").fit_transform(X)[:, -1]
    filled["Median"] = SimpleImputer(strategy="median").fit_transform(X)[:, -1]
    filled["Constant (-1)"] = SimpleImputer(strategy="constant", fill_value=-1).fit_transform(X)[:, -1]

    obs = X["income"].dropna().to_numpy()
    rand = X["income"].to_numpy().copy()
    rand[mask] = rng.choice(obs, size=mask.sum(), replace=True)
    filled["Random sample"] = rand

    filled["KNN (k=5)"] = KNNImputer(n_neighbors=5, weights="distance").fit_transform(X)[:, -1]
    filled["Iterative (MICE)"] = IterativeImputer(
        estimator=BayesianRidge(), max_iter=12, random_state=0, sample_posterior=True
    ).fit_transform(X)[:, -1]

    rows = []
    for name, _ in METHODS:
        v = filled[name]
        imputed_only = v[mask]
        rows.append(
            {
                "method": name,
                "RMSE": float(np.sqrt(np.mean((imputed_only - truth) ** 2))),
                "MAE": float(np.mean(np.abs(imputed_only - truth))),
                "mean_after": float(v.mean()),
                "mean_true": float(true_all.mean()),
                "std_after": float(v.std()),
                "std_true": float(true_all.std()),
                "corr_with_spending": float(np.corrcoef(v, d["spending"].fillna(d["spending"].median()))[0, 1]),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics["mean_bias_%"] = (metrics["mean_after"] - metrics["mean_true"]) / metrics["mean_true"] * 100
    metrics["std_ratio"] = metrics["std_after"] / metrics["std_true"]

    long = pd.concat(
        [
            pd.DataFrame({"method": name, "income": filled[name], "is_imputed": mask})
            for name, _ in METHODS
        ]
        + [pd.DataFrame({"method": "TRUE (ground truth)", "income": true_all, "is_imputed": mask})],
        ignore_index=True,
    )
    return metrics, long


t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. ما الذي تكسره كل طريقة؟", divider="blue")
    st.markdown(
        """
كل تعويض **يخترع** بيانات. السؤال الصحيح: *ما الخاصية الإحصائية التي
سأحافظ عليها، وما التي سأضحّي بها؟*

| الطريقة | يحافظ على | يدمّر |
|---|---|---|
| المتوسط / الوسيط | المتوسط (في MCAR) | :red[التباين] و :red[الارتباطات] — قمّة حادّة مصطنعة عند المتوسط |
| قيمة ثابتة (-1, 999) | التمييز بين المفقود والمرصود | التوزيع تمامًا — صالح للأشجار فقط |
| سحب عشوائي | التوزيع والتباين | الارتباطات (القيمة لا تناسب صفّها) |
| KNN | الارتباطات محليًا | يقلّل التباين قليلًا؛ حسّاس للتقييس والأبعاد العالية |
| Iterative / MICE | الارتباطات والتباين (مع `sample_posterior`) | أبطأ؛ يفترض MAR وبنية قابلة للنمذجة |
| ffill / interpolate (زمني) | الاستمرارية الزمنية | يفترض سلاسة؛ خطر تسريب من المستقبل مع `bfill` |
"""
    )

    st.subheader("٢. التعويض المفرد مقابل المتعدّد", divider="green")
    st.markdown(
        """
- **Single imputation**: قيمة واحدة لكل فقد. المشكلة: تتعامل لاحقًا مع القيمة
  المُخترعة وكأنها مرصودة ⇒ **تقلّل الأخطاء المعيارية** وتعطي فترات ثقة
  أضيق مما يجب (تضخّم خطأ النوع الأول).
- **Multiple Imputation (MI)**: أنشئ $m$ نسخًا (عادة 5–20)، حلّل كلًّا منها،
  ثم ادمج النتائج بقواعد *Rubin*. هذا هو المعيار في البحث الإحصائي المنشور.
"""
    )
    st.latex(r"\bar{Q} = \frac{1}{m}\sum_{i=1}^{m} \hat{Q}_i, \qquad T = \bar{U} + \left(1 + \frac{1}{m}\right) B")
    st.caption(
        "$\\bar{U}$ = متوسط التباين داخل النسخ، $B$ = التباين بين النسخ. "
        "الحدّ $B$ هو ثمن عدم اليقين الناتج عن الفقد — وهو ما يُهمله التعويض المفرد."
    )
    st.markdown(
        "في التعلّم الآلي التنبّؤي (حيث الهدف دقّة التنبّؤ لا الاستدلال على "
        "المعاملات) يُقبل التعويض المفرد داخل `Pipeline`، لأن التقييم يتم "
        "بالتحقّق المتقاطع على بيانات غير مرئية. أمّا في **البحث الاستدلالي** "
        "(معاملات، فترات ثقة، قيم p) فالتعويض المتعدّد هو المطلوب."
    )

    st.subheader("٣. القاعدة التي ينساها الجميع: مؤشّر الفقد", divider="orange")
    st.markdown(
        """
أضف دائمًا عمودًا ثنائيًا `was_missing` قبل التعويض:

- في MNAR: الفقد نفسه يحمل الإشارة (من لم يصرّح بدخله ≈ دخل مرتفع).
- في البيانات الطبية: عدم إجراء التحليل يعني أن الطبيب لم يرَ داعيًا له —
  وهذه معلومة سريرية.
- التكلفة: عمود إضافي. العائد: استرجاع معلومة ضاعت.

`SimpleImputer(..., add_indicator=True)` يفعلها آليًا.
"""
    )

    ui.beginner_note(
        """
التعويض بالمتوسط يشبه ملء الفراغات في امتحان بالإجابة «المتوسطة» لكل سؤال:
ستبدو ورقتك متماسكة من بعيد، لكن **التنوّع الحقيقي اختفى**. ولهذا يقلّ
التباين، وتبدو النتائج أدقّ مما هي عليه فعلًا.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. التعويض البسيط", divider="blue")
    ui.code(
        """
from sklearn.impute import SimpleImputer

num_imp = SimpleImputer(strategy="median", add_indicator=True)   # رقمي منحرف
cat_imp = SimpleImputer(strategy="most_frequent")                # فئوي
cst_imp = SimpleImputer(strategy="constant", fill_value="Missing")  # فئة صريحة

# pandas المباشر (للتنظيف الاستكشافي فقط — لا داخل التدريب)
df["income"] = df["income"].fillna(df["income"].median())
df["education"] = df["education"].fillna("Missing")
"""
    )

    st.subheader("٢. التعويض بالنماذج", divider="green")
    ui.code(
        """
from sklearn.experimental import enable_iterative_imputer   # مطلوب أولًا
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor

# KNN: يتطلّب تقييسًا مسبقًا لأنه قائم على المسافة
knn = KNNImputer(n_neighbors=5, weights="distance")

# MICE: انحدار تكراري لكل عمود على البقيّة
mice = IterativeImputer(max_iter=20, random_state=0, sample_posterior=True)

# MissForest-style: غابة عشوائية، تلتقط اللاخطّية والتفاعلات
rf_imp = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=100, random_state=0),
    max_iter=10, random_state=0,
)
""",
        caption="`sample_posterior=True` يضيف عشوائية تحاكي عدم اليقين — "
        "ضروري إن أردت تعويضًا متعدّدًا بتغيير `random_state`.",
    )

    st.subheader("٣. التعويض المتعدّد (MI) عمليًا", divider="orange")
    ui.code(
        """
# m نسخ مختلفة ⇒ حلّل كل نسخة ⇒ ادمج
import numpy as np
estimates = []
for seed in range(10):
    imp = IterativeImputer(sample_posterior=True, random_state=seed, max_iter=20)
    Xi = imp.fit_transform(X_train)
    model.fit(Xi, y_train)
    estimates.append(model.coef_)

coefs = np.array(estimates)
pooled_mean = coefs.mean(axis=0)
between_var = coefs.var(axis=0, ddof=1)        # الحدّ B في قواعد Rubin
print(pooled_mean, between_var)

# بديل ناضج في Python: statsmodels MICE
from statsmodels.imputation.mice import MICEData, MICE
mice_data = MICEData(df)
res = MICE("y ~ x1 + x2", sm.OLS, mice_data).fit(n_imputations=10)
print(res.summary())
"""
    )

    st.subheader("٤. التعويض الزمني (Time series)", divider="violet")
    ui.code(
        """
s = df.set_index("date")["sensor"]

s.ffill()                      # آخر قيمة معروفة — آمن زمنيًا
s.ffill(limit=3)               # لا تملأ فجوة أطول من 3 خطوات
s.interpolate(method="time")   # استيفاء يراعي المسافات الزمنية
s.interpolate(method="spline", order=2)

# s.bfill() يستعمل المستقبل ⇒ تسريب في أي مهمّة تنبّؤية
""",
        caption=":red-badge[انتبه] `bfill` و `interpolate` ثنائي الاتجاه يسرّبان "
        "معلومة من المستقبل. في التنبّؤ الزمني استعمل `ffill` فقط.",
    )

    st.subheader("٥. المكان الصحيح: داخل Pipeline", divider="red")
    ui.code(
        """
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("sc", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False)),
    ]), cat_cols),
])
"""
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("قارن الطرق مقابل الحقيقة المعلومة", divider="violet")
    st.markdown(
        "بياناتنا مُصنّعة، لذلك نعرف القيمة الحقيقية لكل خلية حذفناها. "
        "في كل إطار نعوّض بطريقة مختلفة ونقيس: **خطأ التعويض**، **انزياح "
        "المتوسط**، و**نسبة التباين الباقية**."
    )

    c1, c2 = st.columns(2)
    with c1:
        mech = st.segmented_control(
            "آلية الفقد", ["MCAR", "MAR", "MNAR"], default="MAR", key="imp_mech"
        ) or "MAR"
    with c2:
        rate = st.select_slider(
            "نسبة الفقد",
            options=[0.1, 0.2, 0.3, 0.4],
            value=0.3,
            format_func=lambda v: f"{v:.0%}",
            key="imp_rate",
        )

    metrics, long = run_imputations(mech, rate)

    def _render_method(frame: int) -> None:
        name, label_ar = METHODS[frame]
        row = metrics[metrics["method"] == name].iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("RMSE للتعويض", f"{row['RMSE']:,.0f}", help="أصغر = أقرب للقيم الحقيقية")
        k2.metric("انزياح المتوسط", f"{row['mean_bias_%']:+.2f}%", delta_color="off")
        k3.metric(
            "نسبة التباين الباقية",
            f"{row['std_ratio']:.2f}",
            help="1.00 = حافظنا على التشتّت الحقيقي؛ أقل من ذلك = تقلّص مصطنع",
        )
        k4.metric("الارتباط مع spending", f"{row['corr_with_spending']:.2f}")

        left, right = st.columns([3, 2])
        with left:
            st.markdown("###### التوزيع: الحقيقة مقابل ما بعد التعويض")
            sub = long[long["method"].isin([name, "TRUE (ground truth)"])]
            sub = sub[sub["income"] > -1e5]
            ch = (
                alt.Chart(sub)
                .mark_area(opacity=0.45, interpolate="step")
                .encode(
                    x=alt.X("income:Q", bin=alt.Bin(maxbins=45), title="income"),
                    y=alt.Y("count()", title="count", stack=None),
                    color=alt.Color(
                        "method:N",
                        legend=alt.Legend(orient="top", title=None),
                        scale=alt.Scale(range=["#4C6EF5", "#0CA678"]),
                    ),
                )
                .properties(height=300)
            )
            st.altair_chart(ch, width="stretch")
        with right:
            st.markdown("###### RMSE حتى الآن")
            sofar = metrics.iloc[: frame + 1][["method", "RMSE"]]
            st.altair_chart(
                charts.bar(sofar, "method", "RMSE", horizontal=True, height=300, sort_desc=False),
                width="stretch",
            )

        notes = {
            "Mean": "قمّة حادّة مصطنعة عند المتوسط. التباين ينهار، والارتباط مع "
            "المتغيّرات الأخرى يضعف. سريع لكنه الأسوأ إحصائيًا.",
            "Median": "أمتن من المتوسط أمام الانحراف والشواذ، لكنه يعاني نفس "
            "مشكلة تقلّص التباين.",
            "Constant (-1)": "لا يحاول تقدير القيمة إطلاقًا — يصنع «فئة» منفصلة. "
            "مدمّر للنماذج الخطّية والمسافية، لكنه **مقبول ومفيد للأشجار** "
            "التي تستطيع عزل هذا المدى بتقسيم واحد.",
            "Random sample": "يحافظ على شكل التوزيع والتباين بشكل ممتاز، "
            "لكن القيمة لا تناسب الصف ⇒ الارتباطات تضعف.",
            "KNN (k=5)": "يستعمل الصفوف المتشابهة ⇒ يحترم الارتباطات. يتطلّب "
            "تقييسًا مسبقًا، ويضعف في الأبعاد العالية (لعنة الأبعاد).",
            "Iterative (MICE)": "ينمذج `income` انحداريًا من بقيّة الأعمدة تكراريًا. "
            "الأفضل عادة تحت MAR، ومع `sample_posterior=True` يحافظ على التباين.",
        }
        st.markdown(f"**:blue-badge[{name}]** — {label_ar}. {notes[name]}")

    play_animation(
        f"impute_{mech}_{rate}",
        len(METHODS),
        _render_method,
        fps=0.5,
        label="الطريقة",
        frame_labels=[m[0] for m in METHODS],
    )

    st.markdown("###### جدول المقارنة الكامل")
    st.dataframe(
        metrics[["method", "RMSE", "MAE", "mean_bias_%", "std_ratio", "corr_with_spending"]]
        .round(3),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "جرّب تغيير الآلية إلى **MNAR**: سترى أن **كل** الطرق تُبقي انزياحًا "
        "في المتوسط — وهذا بالضبط ما تعنيه عبارة «لا توجد معالجة إحصائية "
        "تحلّ MNAR»."
    )

# ============================================================== القرار
with t_decide:
    st.subheader("أي طريقة تعويض أختار؟", divider="orange")
    ui.decision_table(
        [
            ("رقمي، توزيع متماثل، فقد قليل", "Mean + indicator", "بسيط وسريع والانحياز محدود"),
            ("رقمي منحرف أو فيه شواذ", "Median + indicator", "الوسيط أمتن (robust)"),
            ("رقمي مرتبط بأعمدة أخرى", "Iterative (MICE) أو KNN", "يستغل بنية الارتباط فيقلّل الخطأ"),
            ("نموذج شجري (RF/GBM)", "Constant خارج المدى أو NaN مباشرة", "الأشجار تعزل القيمة بتقسيم؛ HistGB وLightGBM تقبل NaN أصلًا"),
            ("نموذج خطّي / شبكة عصبية", "MICE + تقييس", "حسّاسة للقيم غير المنطقية ولا تقبل NaN"),
            ("فئوي", "فئة `Missing` صريحة", "أفضل من المنوال: يحتفظ بمعلومة الفقد"),
            ("سلسلة زمنية", "`ffill` (بحدّ limit) أو interpolate زمني", "احترام ترتيب الزمن ومنع تسريب المستقبل"),
            ("بحث استدلالي (معاملات، p-values)", "Multiple Imputation + قواعد Rubin", "التعويض المفرد يضيّق فترات الثقة كذبًا"),
            ("MNAR مشتبه به", "مؤشّر فقد + تحليل حساسية", "لا طريقة تُزيل التحيّز؛ صرّح بالافتراض"),
            ("فقد في الهدف", "احذف الصف", "التعويض في `y` تسريب صريح"),
        ],
        headers=("الحالة", "الطريقة", "السبب"),
    )

    ui.danger(
        "قاعدة التنفيذ: **المعاملات تُحسب من التدريب فقط.** "
        "`imputer.fit(X_train)` ثم `imputer.transform(X_test)`. "
        "أي `fit_transform` على كامل البيانات = تسريب."
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("scikit-learn — Imputation of missing values", "https://scikit-learn.org/stable/modules/impute.html"),
            ("scikit-learn — IterativeImputer", "https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html"),
            ("scikit-learn — KNNImputer", "https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html"),
            ("scikit-learn — Imputing missing values before building an estimator", "https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html"),
            ("statsmodels — MICE", "https://www.statsmodels.org/stable/imputation.html"),
            ("van Buuren — FIMD (مرجع التعويض المتعدّد)", "https://stefvanbuuren.name/fimd/"),
            ("pandas — fillna / interpolate", "https://pandas.pydata.org/docs/user_guide/missing_data.html#filling-missing-data"),
        ]
    )
    ui.terms(
        [
            ("Imputation", "التعويض"),
            ("Single imputation", "تعويض مفرد"),
            ("Multiple Imputation (MI)", "تعويض متعدّد"),
            ("MICE", "تعويض متسلسل بالمعادلات المتسلسلة"),
            ("Rubin's rules", "قواعد دمج نتائج النسخ المتعدّدة"),
            ("add_indicator", "إضافة عمود مؤشّر للفقد"),
            ("Variance shrinkage", "تقلّص التباين بعد التعويض"),
        ]
    )
