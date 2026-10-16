"""هندسة المتغيّرات (Feature engineering)."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core import charts, ui
from core.anim import play_animation

ui.page_header(
    "هندسة المتغيّرات (Feature engineering)",
    "في البيانات الجدولية، **تحسين المتغيّرات يتفوّق على تحسين النموذج** في "
    "معظم الأحيان. النموذج يرى ما تعطيه فقط؛ وإن كانت المعلومة مخبّأة في "
    "نسبة بين عمودين، فلن يجدها نموذج خطّي أبدًا ما لم تحسبها أنت.",
    icon=":material/construction:",
    badges=[
        ":blue-badge[Ratios]",
        ":green-badge[Interactions]",
        ":orange-badge[Datetime]",
        ":violet-badge[Aggregations]",
        ":red-badge[Lag features]",
    ],
)


# ------------------------------------------------- بيانات مُصمَّمة للتوضيح
@st.cache_data(ttl=1800, max_entries=4)
def designed_task(n: int = 900) -> pd.DataFrame:
    """هدف يعتمد على **نسبة** و**تفاعل** و**لوغاريتم** — لا على الأعمدة الخام."""
    rng = np.random.default_rng(13)
    income = rng.lognormal(10.6, 0.45, n)
    spending = income * rng.beta(2, 5, n) * 1.6
    age = rng.normal(40, 11, n).clip(18, 80)
    satisfaction = rng.normal(6.5, 1.8, n).clip(1, 10)
    city = rng.choice(["Algiers", "Oran", "Constantine", "Annaba"], n, p=[0.4, 0.25, 0.2, 0.15])
    signup = pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 1400, n), unit="D")

    burn = spending / income                      # نسبة
    inter = (age / 40) * (satisfaction / 7)       # تفاعل
    risk = (
        30 * burn
        + 8 * inter
        + 2.2 * np.log(income)
        - 0.02 * (pd.Series(signup).dt.month.to_numpy() - 6) ** 2
        + rng.normal(0, 1.1, n)
    )
    return pd.DataFrame(
        {
            "income": income,
            "spending": spending,
            "age": age,
            "satisfaction": satisfaction,
            "city": city,
            "signup_date": signup,
            "risk_score": risk,
        }
    )


FEATURE_SETS = [
    ("الأعمدة الخام فقط", ["income", "spending", "age", "satisfaction"]),
    ("+ نسبة الإنفاق `spending/income`", ["income", "spending", "age", "satisfaction", "burn_rate"]),
    ("+ تفاعل `age × satisfaction`", ["income", "spending", "age", "satisfaction", "burn_rate", "age_x_sat"]),
    ("+ `log(income)`", ["income", "spending", "age", "satisfaction", "burn_rate", "age_x_sat", "log_income"]),
    ("+ سمات التاريخ (شهر، أقدمية)", ["income", "spending", "age", "satisfaction", "burn_rate", "age_x_sat", "log_income", "month_sin", "month_cos", "tenure_days"]),
    ("+ إحصاءات المجموعة (نسبة للمدينة)", ["income", "spending", "age", "satisfaction", "burn_rate", "age_x_sat", "log_income", "month_sin", "month_cos", "tenure_days", "income_vs_city"]),
]


@st.cache_data(ttl=1800, max_entries=4)
def engineered(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["burn_rate"] = d["spending"] / d["income"]
    d["age_x_sat"] = (d["age"] / 40) * (d["satisfaction"] / 7)
    d["log_income"] = np.log1p(d["income"])
    m = d["signup_date"].dt.month
    d["month_sin"] = np.sin(2 * np.pi * m / 12)
    d["month_cos"] = np.cos(2 * np.pi * m / 12)
    d["tenure_days"] = (pd.Timestamp("2025-01-01") - d["signup_date"]).dt.days
    d["income_vs_city"] = d["income"] / d.groupby("city")["income"].transform("median")
    return d


@st.cache_data(ttl=1800, max_entries=4, show_spinner="جارٍ تقييم مجموعات المتغيّرات…")
def fe_scores() -> pd.DataFrame:
    d = engineered(designed_task())
    y = d["risk_score"]
    rows = []
    for name, cols in FEATURE_SETS:
        X = d[cols]
        lin = Pipeline([("sc", StandardScaler()), ("m", Ridge())])
        r2_lin = cross_val_score(lin, X, y, cv=5, scoring="r2").mean()
        rf = RandomForestRegressor(n_estimators=120, random_state=0, n_jobs=1)
        r2_rf = cross_val_score(rf, X, y, cv=5, scoring="r2").mean()
        rows.append(
            {"feature_set": name, "n_features": len(cols), "Ridge": r2_lin, "RandomForest": r2_rf}
        )
    return pd.DataFrame(rows)


t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. سبع عائلات من المتغيّرات المشتقّة", divider="blue")
    ui.concept_grid(
        [
            (
                "١. النسب والفروق",
                "`spending / income`، `debt / assets`، `price_per_m2`، "
                "`BMI = weight / height²`.\n\n"
                "لماذا؟ لأن المعنى غالبًا في **العلاقة** لا في القيمة المطلقة. "
                "النموذج الخطّي لا يستطيع تكوين نسبة بنفسه أبدًا.",
                "blue",
            ),
            (
                "٢. التفاعلات والحدود كثيرة الحدود",
                "`age × income`، `x²`، `PolynomialFeatures(degree=2)`.\n\n"
                "لماذا؟ حين يختلف أثر متغيّر باختلاف قيمة آخر. "
                ":orange[حذر] عدد المتغيّرات ينفجر تربيعيًا.",
                "green",
            ),
            (
                "٣. سمات التاريخ والوقت",
                "سنة، شهر، يوم الأسبوع، أهو عطلة؟ ساعة اليوم، **الأقدمية** "
                "(tenure)، الفارق بين تاريخين.\n\n"
                "التاريخ الخام عديم الفائدة للتعميم؛ المشتقّات هي الذهب.",
                "orange",
            ),
            (
                "٤. إحصاءات المجموعة (Aggregations)",
                "متوسط إنفاق المدينة، عدد طلبات العميل، `income / median(income by city)`.\n\n"
                ":red[حذر شديد] تُحسب من **التدريب فقط**، وإلا سرّبت معلومة "
                "عن صفوف الاختبار.",
                "violet",
            ),
            (
                "٥. متغيّرات التأخّر والنوافذ (زمنية)",
                "`lag_1`, `lag_7`, `rolling_mean_30`, `diff`, `pct_change`.\n\n"
                ":red[قاعدة حديدية] استعمل الماضي فقط: `shift(1)` قبل أي "
                "`rolling`، وإلا تسرّب المستقبل.",
                "red",
            ),
            (
                "٦. متغيّرات المجال",
                "مؤشّرات مالية (مديونية/سيولة)، مؤشّرات طبية (BMI, eGFR)، "
                "مؤشّرات تسويقية (RFM).\n\n"
                ":green[الأعلى مردودًا] لأنها تضخّ معرفة لا توجد في البيانات.",
                "green",
            ),
            (
                "٧. متغيّرات من البنية نفسها",
                "`n_missing_in_row`، `was_missing`، `is_outlier`، طول النص، "
                "عدد الرموز الخاصة في البريد.\n\n"
                "رخيصة جدًا ومفيدة بشكل مدهش — خصوصًا في كشف الاحتيال.",
                "blue",
            ),
            (
                "٨. التجزيء والترميز الدوري",
                "شرائح عمرية؛ `sin/cos` للساعة والشهر.\n\n"
                "يسمح للنموذج الخطّي بالتقاط الانحناء والدورية.",
                "orange",
            ),
        ],
        n_cols=4,
    )

    st.subheader("٢. ثلاث قواعد تحكم كل شيء", divider="red")
    st.markdown(
        """
1. **لا متغيّر من المستقبل.** اسأل عن كل متغيّر: *«هل ستكون هذه المعلومة
   متاحة لحظة التنبّؤ فعليًا؟»* عمود `تاريخ_إغلاق_الشكوى` غير متاح وقت
   فتحها. هذا أشهر مصادر التسريب في المشاريع الحقيقية.
2. **كل إحصاء مُتعلَّم داخل `Pipeline`.** متوسطات المجموعات، الحدود،
   التعيينات — كلها تُحسب من التدريب وتُطبّق على الاختبار.
3. **قِس، لا تخمّن.** أضف المتغيّر، وأعد التحقّق المتقاطع. المتغيّر الذي لا
   يحسّن شيئًا يزيد الضجيج وخطر فرط الملاءمة وتكلفة الصيانة.
"""
    )

    ui.beginner_note(
        """
مثال يوضّح كل شيء: عمودان `الإنفاق` و`الدخل`. النموذج الخطّي يستطيع تعلّم
`a·إنفاق + b·دخل`، لكنه **لا يستطيع** تعلّم `إنفاق ÷ دخل` مهما درّبته،
لأن القسمة ليست عملية خطّية. فإن كانت الظاهرة الحقيقية تعتمد على النسبة
(وهي كذلك في المخاطر الائتمانية)، فالمتغيّر الذي تحسبه أنت بيدك هو الفرق
بين نموذج فاشل وآخر ناجح.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. النسب والفروق والتفاعلات", divider="blue")
    ui.code(
        """
import numpy as np

df["burn_rate"]   = df["spending"] / df["income"].replace(0, np.nan)
df["savings"]     = df["income"] - df["spending"]
df["age_x_sat"]   = df["age"] * df["satisfaction"]
df["log_income"]  = np.log1p(df["income"])

# آليًا لكل الأزواج (انتبه لانفجار الأبعاد)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X[num_cols])
print(poly.get_feature_names_out()[:12])
""",
        caption="`replace(0, np.nan)` يتجنّب القسمة على صفر؛ ثم يعالَج الناتج "
        "في خطوة التعويض.",
    )

    st.subheader("٢. سمات التاريخ", divider="orange")
    ui.code(
        """
d = df["signup_date"]
df["year"]        = d.dt.year
df["month"]       = d.dt.month
df["dayofweek"]   = d.dt.dayofweek
df["is_weekend"]  = d.dt.dayofweek.isin([4, 5]).astype("int8")   # الجمعة والسبت
df["quarter"]     = d.dt.quarter
df["tenure_days"] = (reference_date - d).dt.days                  # الأقدمية

# ترميز دوري للشهر والساعة
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
""",
        caption=":red-badge[انتبه] `reference_date` يجب أن يكون تاريخ **التنبّؤ**، "
        "لا `today()` ولا آخر تاريخ في الجدول — وإلا تسرّب المستقبل.",
    )

    st.subheader("٣. إحصاءات المجموعة بلا تسريب", divider="violet")
    ui.code(
        '''
from sklearn.base import BaseEstimator, TransformerMixin

class GroupStat(BaseEstimator, TransformerMixin):
    """نسبة قيمة العمود إلى وسيط مجموعته — تُتعلَّم من التدريب فقط."""

    def __init__(self, by="city", col="income"):
        self.by, self.col = by, col

    def fit(self, X, y=None):
        self.map_ = X.groupby(self.by)[self.col].median()
        self.global_ = X[self.col].median()
        return self

    def transform(self, X):
        X = X.copy()
        ref = X[self.by].map(self.map_).fillna(self.global_)   # مجموعة جديدة ⇒ العام
        X[f"{self.col}_vs_{self.by}"] = X[self.col] / ref
        return X
''',
        caption="بديل سريع (استكشافي فقط): "
        "`df['x_vs_g'] = df['x'] / df.groupby('g')['x'].transform('median')` — "
        "لكنه يحسب من كامل البيانات ⇒ لا يصلح داخل التدريب.",
    )

    st.subheader("٤. المتغيّرات الزمنية (lag / rolling)", divider="red")
    ui.code(
        """
df = df.sort_values(["customer_id", "date"])
g = df.groupby("customer_id")["amount"]

df["lag_1"]      = g.shift(1)
df["lag_7"]      = g.shift(7)
df["roll_mean_30"] = g.shift(1).rolling(30, min_periods=5).mean()   # shift أولًا!
df["diff_1"]     = df["amount"] - df["lag_1"]
df["pct_change"] = g.pct_change()
""",
        caption=":red-badge[الخطأ القاتل] `rolling(30).mean()` **بدون** `shift(1)` "
        "يُدخل قيمة اليوم نفسه في متوسطه ⇒ تسريب مباشر من الهدف.",
    )

    st.subheader("٥. متغيّرات من بنية البيانات", divider="green")
    ui.code(
        """
df["n_missing_row"]  = df.isna().sum(axis=1)
df["income_missing"] = df["income"].isna().astype("int8")
df["email_len"]      = df["email"].str.len()
df["email_digits"]   = df["email"].str.count(r"\\d")
df["name_tokens"]    = df["full_name"].str.split().str.len()
""",
        caption="هذه المتغيّرات «المجّانية» كثيرًا ما تكون من أقوى المتنبّئات "
        "في كشف الاحتيال وجودة البيانات.",
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("أضف متغيّرًا… وشاهد الأداء يقفز", divider="violet")
    st.markdown(
        "صمّمنا بيانات هدفها يعتمد على **نسبة** و**تفاعل** و**لوغاريتم** "
        "و**دورة شهرية** — أي على أشياء غير موجودة في الأعمدة الخام. "
        "في كل إطار نضيف عائلة متغيّرات ونعيد القياس بتحقّق متقاطع خماسي."
    )

    sc = fe_scores()

    def _render_fe(frame: int) -> None:
        row = sc.iloc[frame]
        base = sc.iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("مجموعة المتغيّرات", f"{frame + 1}/{len(sc)}")
        k2.metric("عدد المتغيّرات", int(row["n_features"]))
        k3.metric(
            "R² للنموذج الخطّي (Ridge)",
            f"{row['Ridge']:.3f}",
            delta=f"{row['Ridge'] - base['Ridge']:+.3f}",
        )
        k4.metric(
            "R² للغابة العشوائية",
            f"{row['RandomForest']:.3f}",
            delta=f"{row['RandomForest'] - base['RandomForest']:+.3f}",
        )

        sofar = sc.iloc[: frame + 1].melt(
            id_vars=["feature_set", "n_features"],
            value_vars=["Ridge", "RandomForest"],
            var_name="model",
            value_name="r2",
        )
        ch = (
            alt.Chart(sofar)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("n_features:Q", title="number of features"),
                y=alt.Y("r2:Q", title="cross-validated R²", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("model:N", legend=alt.Legend(orient="top", title=None)),
                tooltip=["feature_set:N", "model:N", "r2:Q"],
            )
            .properties(height=330)
        )
        st.altair_chart(ch, width="stretch")
        st.markdown(f"**:violet-badge[{row['feature_set']}]**")

    play_animation(
        "fe_build",
        len(sc),
        _render_fe,
        fps=0.6,
        label="المجموعة",
        frame_labels=sc["feature_set"].tolist(),
    )

    st.dataframe(sc.round(3), width="stretch", hide_index=True)
    st.caption(
        "اقرأ النتيجة بدقّة: **النموذج الخطّي** يقفز قفزة كبيرة لأنه عاجز "
        "بنيويًا عن تكوين النسب والتفاعلات بنفسه. **الغابة العشوائية** تبدأ "
        "أعلى (تقرّب اللاخطّية) لكنها تستفيد أيضًا — لأن المتغيّر الصريح أسهل "
        "عليها من تقريبه بعشرات التقسيمات."
    )

    st.subheader("شاهد المتغيّر المشتقّ مباشرة", divider="orange")
    d = engineered(designed_task())
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("###### `spending` الخام مقابل الهدف")
        st.altair_chart(
            charts.scatter(d.sample(400, random_state=1), "spending", "risk_score", height=280),
            width="stretch",
        )
    with c2:
        st.markdown("###### `burn_rate = spending/income` مقابل الهدف")
        st.altair_chart(
            charts.scatter(d.sample(400, random_state=1), "burn_rate", "risk_score", height=280),
            width="stretch",
        )
    st.caption(
        "نفس البيانات تمامًا. اليسار سحابة بلا بنية واضحة؛ اليمين علاقة خطّية "
        "شبه مثالية. **الفرق عملية قسمة واحدة.**"
    )

# ============================================================== القرار
with t_decide:
    st.subheader("ماذا أهندس ومتى؟", divider="orange")
    ui.decision_table(
        [
            ("نموذج خطّي والعلاقة غير خطّية", "نسب، لوغاريتمات، تفاعلات، شرائح", "النموذج لا يكوّنها بنفسه"),
            ("نموذج شجري", "ركّز على النسب وإحصاءات المجموعة", "الأشجار تلتقط التفاعلات لكنها تعجز عن القسمة"),
            ("عمود تاريخ", "استخرج مشتقّاته دائمًا", "التاريخ الخام لا يعمّم على المستقبل"),
            ("بيانات معاملات / سلاسل", "lag و rolling **بعد** `shift(1)`", "منع تسريب المستقبل"),
            ("عمود عالي الكاردينالية", "إحصاءات مجموعة + Target encoding", "يلخّص الفئة برقم دالّ"),
            ("أعمدة كثيرة ومتغيّرات مشتقّة كثيرة", "اختيار متغيّرات بعد الهندسة", "الضجيج يتراكم ويزيد فرط الملاءمة"),
            ("لديك خبير مجال", "اجلس معه قبل أي كود", "أعلى عائد على الوقت في المشروع كلّه"),
            ("متغيّر لا يحسّن CV", "احذفه", "تكلفة صيانة وخطر بلا عائد"),
            ("متغيّر يرفع الأداء رفعًا مذهلًا", "اشكّ في تسريب فورًا", "القفزات غير المتوقّعة تسريب حتى يثبت العكس"),
        ]
    )

    ui.danger(
        "اختبار «لحظة التنبّؤ»: اكتب بجانب كل متغيّر **متى** تصبح قيمته "
        "معروفة. أي متغيّر يصبح معروفًا بعد لحظة التنبّؤ — احذفه، مهما كانت "
        "قوّته التنبّؤية. بل **خصوصًا** إن كانت قوّته مذهلة."
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("scikit-learn — Preprocessing: polynomial & spline features", "https://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features"),
            ("scikit-learn — Time-related feature engineering", "https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html"),
            ("scikit-learn — FunctionTransformer", "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html"),
            ("pandas — Group by: split-apply-combine", "https://pandas.pydata.org/docs/user_guide/groupby.html"),
            ("pandas — Windowing operations (rolling)", "https://pandas.pydata.org/docs/user_guide/window.html"),
            ("feature-engine — مكتبة متخصّصة في هندسة المتغيّرات", "https://feature-engine.trainindata.com/"),
            ("Kaufman et al. (2012) — Leakage in Data Mining", "https://dl.acm.org/doi/10.1145/2020408.2020496"),
        ]
    )
    ui.terms(
        [
            ("Feature engineering", "هندسة المتغيّرات"),
            ("Interaction term", "حدّ التفاعل"),
            ("Aggregation feature", "متغيّر تجميعي من مجموعة"),
            ("Lag feature", "متغيّر متأخّر زمنيًا"),
            ("Rolling window", "نافذة متدحرجة"),
            ("Domain knowledge", "معرفة المجال"),
            ("Prediction time", "لحظة التنبّؤ"),
        ]
    )
