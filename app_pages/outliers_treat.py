"""معالجة القيم الشاذة: الاحتفاظ، الحذف، القص، التحويل."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import HuberRegressor, LinearRegression, TheilSenRegressor

from core import charts, ui
from core.anim import play_animation
from core.datasets import clean_customers

ui.page_header(
    "المعالجة: نحتفظ أم نحذف؟",
    "أصعب قرار في تنظيف البيانات. الحذف الآلي لكل ما تجاوز ثلاثة انحرافات "
    "هو **أكثر الممارسات شيوعًا وأكثرها ضررًا**: قد تحذف بالضبط الظاهرة التي "
    "تبحث عنها.",
    icon=":material/rule:",
    badges=[
        ":green-badge[Keep]",
        ":red-badge[Remove]",
        ":orange-badge[Winsorize]",
        ":blue-badge[Transform]",
        ":violet-badge[Robust model]",
    ],
)

t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. ثمانية خيارات، لا خياران", divider="blue")
    ui.concept_grid(
        [
            (
                "١. الاحتفاظ + نموذج أمتن",
                "لا تلمس البيانات، واستعمل نموذجًا لا يتأثّر: `HuberRegressor`, "
                "`TheilSenRegressor`, `RANSAC`, أو شجرة/غابة.\n\n"
                ":green[الأفضل علميًا] لأنك لا تحذف دليلًا — تغيّر الأداة.",
                "green",
            ),
            (
                "٢. الحذف",
                "فقط إذا أثبتّ أن القيمة **خطأ** (إدخال/قياس/عيّنة من مجتمع آخر). "
                "وثّق: كم صفًا، ولماذا، وما أثر الحذف على النتائج.\n\n"
                ":red[خطر] حذف الظواهر الحقيقية = تجميل البيانات.",
                "red",
            ),
            (
                "٣. القص / Winsorizing",
                "استبدل القيم فوق المئين 99 بقيمة المئين 99 نفسها (وكذلك الأسفل). "
                "تحتفظ بالصف وتحدّ من تأثيره.\n\n"
                ":orange[حذر] يشوّه التوزيع ويقلّل التباين؛ ويجب أن تُحسب "
                "الحدود من **التدريب** فقط.",
                "orange",
            ),
            (
                "٤. التحويل",
                "`log1p`, `sqrt`, Box-Cox, Yeo-Johnson: تضغط الذيل الطويل "
                "فتقلّ «شذوذية» القيم الكبيرة دون حذف أي صف.\n\n"
                ":blue[ممتاز] للمتغيّرات المنحرفة طبيعيًا كالدخل والأسعار.",
                "blue",
            ),
            (
                "٥. التحويل إلى `NaN` ثم التعويض",
                "إن كانت القيمة خطأ قياس واضحًا (عمر = 199)، فهي ليست «شاذّة» "
                "بل **مفقودة مُقنَّعة**. حوّلها إلى `NaN` وعالجها بدرس الفقد.\n\n"
                ":violet[الأنظف منهجيًا] في أخطاء الإدخال.",
                "violet",
            ),
            (
                "٦. التجزيء (Binning)",
                "حوّل المتغيّر إلى فئات (شرائح دخل). القيمة المتطرّفة تقع في "
                "الشريحة العليا وينتهي تأثيرها الرقمي.\n\n"
                ":orange[الثمن] خسارة معلومة داخل الشريحة.",
                "orange",
            ),
            (
                "٧. علم (flag) كمتغيّر",
                "`is_outlier` كعمود ثنائي + الاحتفاظ بالقيمة. النموذج يقرّر "
                "بنفسه قيمة هذه المعلومة.\n\n"
                ":green[رخيص ومفيد] خصوصًا مع النماذج الشجرية.",
                "green",
            ),
            (
                "٨. نموذج منفصل",
                "إن كان الشواذ مجموعة متماسكة (عملاء مؤسّسات وسط أفراد)، "
                "فربّما هم **مجتمع آخر** يستحق نموذجه الخاص.\n\n"
                ":blue[قرار بحثي] وليس تنظيفًا.",
                "blue",
            ),
        ],
        n_cols=4,
    )

    st.subheader("٢. أثر الشاذّة يختلف باختلاف النموذج", divider="red")
    st.markdown(
        """
| النموذج | الحساسية | السبب |
|---|---|---|
| OLS / الانحدار الخطّي | :red[قصوى] | يقلّل **مربّع** الخطأ ⇒ الخطأ الكبير يُضخَّم تربيعيًا |
| المتوسط والانحراف المعياري | :red[قصوى] | نقطة انهيار 0٪ |
| KNN / K-Means / SVM | :orange[عالية] | المسافة والمراكز تنجذب للشاذّة |
| الشبكات العصبية | :orange[عالية] | تدرّجات ضخمة تزعزع التدريب (إلا مع Huber loss) |
| PCA | :orange[عالية] | تعظّم التباين، والشاذّة تحتكر المكوّن الأول |
| الأشجار / RF / GBM | :green[منخفضة] | التقسيم بالترتيب لا بالمسافة |
| الوسيط / MAD / الانحدار الكمّي | :green[منخفضة جدًا] | مقاييس أمتن بالتصميم |

**استنتاج عملي:** إذا كنت تستعمل Random Forest أو Gradient Boosting، فكثير
من «معالجة الشواذ» عمل زائد. أمّا مع الانحدار الخطّي فهي حاسمة.
"""
    )

    st.subheader("٣. الشاذّة في المتغيّرات أم في الهدف؟", divider="orange")
    st.markdown(
        """
- **شاذّة في `X` فقط** مع `y` متّسق معها ⇒ نقطة **رافعة (high leverage)**
  لكنها غالبًا **غير ضارّة**، بل تزيد دقّة تقدير الميل.
- **شاذّة في `y`** عند `x` عادي ⇒ بقيّة كبيرة (large residual) تسحب الخط.
- **شاذّة في الاثنين معًا بعيدًا عن العلاقة** ⇒ **نقطة مؤثّرة** تقلب النتيجة:
  هذه التي تظهر في مسافة كوك.

ولهذا: **لا تحكم على الشذوذ بالنظر إلى عمود واحد في مسألة انحدار** —
انظر إلى البواقي (residuals) والرافعة معًا.
"""
    )

    ui.danger(
        "في بيانات **الاختبار** وفي الإنتاج: لا تحذف الشواذ. الاختبار يجب أن "
        "يمثّل الواقع كما هو. القص (clipping) بحدود **متعلَّمة من التدريب** "
        "مسموح، لأنه جزء من التحويل لا من التنظيف."
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. القص داخل Pipeline (الطريقة الصحيحة)", divider="blue")
    ui.code(
        '''
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Winsorizer(BaseEstimator, TransformerMixin):
    """يتعلّم الحدود من التدريب ويطبّقها على أي بيانات لاحقة."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

# الاستعمال
pipe = Pipeline([
    ("winsor", Winsorizer(0.01, 0.99)),
    ("scale", StandardScaler()),
    ("model", Ridge()),
])
''',
        caption="لاحظ: `fit` يحفظ الحدود، و `transform` يطبّقها — "
        "فلا يتسرّب شيء من الاختبار.",
    )

    st.subheader("٢. بدائل جاهزة من scikit-learn", divider="green")
    ui.code(
        """
from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer

# تقييس أمتن: يستعمل الوسيط و IQR بدل المتوسط والانحراف
RobustScaler(quantile_range=(25.0, 75.0))

# يحوّل أي توزيع إلى طبيعي/منتظم ⇒ يبتلع الشواذ تمامًا
QuantileTransformer(output_distribution="normal", n_quantiles=1000)

# يقلّل الانحراف مع الحفاظ على الترتيب (يقبل القيم السالبة)
PowerTransformer(method="yeo-johnson", standardize=True)
"""
    )

    st.subheader("٣. نماذج أمتن بدل حذف البيانات", divider="violet")
    ui.code(
        """
from sklearn.linear_model import HuberRegressor, TheilSenRegressor, RANSACRegressor, QuantileRegressor

HuberRegressor(epsilon=1.35)      # خسارة تربيعية قرب الصفر وخطّية بعيدًا عنه
TheilSenRegressor(random_state=0) # وسيط ميول كل الأزواج — نقطة انهيار ~29%
RANSACRegressor(random_state=0)   # يلائم «المتوافقين» ويتجاهل الباقي
QuantileRegressor(quantile=0.5)   # انحدار الوسيط (L1)

# في الانحدار التدرّجي: استعمل خسارة أمتن
from sklearn.ensemble import HistGradientBoostingRegressor
HistGradientBoostingRegressor(loss="absolute_error")
"""
    )

    st.subheader("٤. وثّق القرار (شرط للنشر)", divider="orange")
    ui.code(
        '''
report = {
    "rule": "modified z-score (MAD) > 3.5 على income",
    "n_flagged": int(mask.sum()),
    "pct_flagged": float(mask.mean()),
    "action": "winsorize at 1%/99% (ليس حذفًا)",
    "justification": "الذيل الأيمن طبيعي في توزيع الدخل؛ الحذف سيحذف شريحة حقيقية",
    "sensitivity": "أُعيد التحليل بدون معالجة: تغيّر المعامل من 0.42 إلى 0.45",
}
pd.Series(report).to_json("outlier_decisions.json", force_ascii=False)
''',
        caption="**تحليل الحساسية** (إعادة التحليل مع وبدون المعالجة) هو ما "
        "يطلبه المُحكّمون — وهو ما يحمي نتيجتك.",
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("شاهد: نقطة واحدة تقلب الانحدار", divider="violet")
    st.markdown(
        "نأخذ علاقة سليمة بين `experience_years` و `income`، ثم نُبعد **نقطة "
        "واحدة** تدريجيًا. راقب: الخط الأزرق (OLS) يلاحقها، بينما الأخضر "
        "(Theil-Sen) والبرتقالي (Huber) يكادان لا يتحرّكان."
    )

    base = clean_customers(160).copy()
    x0 = base["experience_years"].to_numpy(dtype=float)
    y0 = base["income"].to_numpy(dtype=float)
    SHIFTS = [0, 1, 2, 3, 4, 6, 8, 11, 15]

    @st.cache_data(ttl=1800, max_entries=16)
    def _fit_all(shift_idx: int) -> dict:
        mult = SHIFTS[shift_idx]
        x = np.append(x0, 42.0)
        y = np.append(y0, y0.mean() + mult * y0.std())
        X = x.reshape(-1, 1)
        out = {"x": x.tolist(), "y": y.tolist()}
        for name, est in (
            ("OLS", LinearRegression()),
            ("Huber", HuberRegressor(epsilon=1.35, max_iter=300)),
            ("Theil-Sen", TheilSenRegressor(random_state=0, max_subpopulation=2000)),
        ):
            est.fit(X, y)
            out[name] = float(np.ravel(est.coef_)[0])
            out[f"{name}_int"] = float(np.ravel(est.intercept_)[0])
        out["corr"] = float(np.corrcoef(x, y)[0, 1])
        out["mean"] = float(y.mean())
        out["std"] = float(y.std())
        return out

    def _render_leverage(frame: int) -> None:
        r = _fit_all(frame)
        d = pd.DataFrame({"x": r["x"], "y": r["y"]})
        d["kind"] = ["normal"] * (len(d) - 1) + ["the outlier"]

        clean_r = _fit_all(0)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ميل OLS", f"{r['OLS']:,.0f}", delta=f"{r['OLS'] - clean_r['OLS']:+,.0f}", delta_color="inverse")
        k2.metric("ميل Huber", f"{r['Huber']:,.0f}", delta=f"{r['Huber'] - clean_r['Huber']:+,.0f}", delta_color="off")
        k3.metric("ميل Theil-Sen", f"{r['Theil-Sen']:,.0f}", delta=f"{r['Theil-Sen'] - clean_r['Theil-Sen']:+,.0f}", delta_color="off")
        k4.metric("معامل الارتباط", f"{r['corr']:.3f}", delta=f"{r['corr'] - clean_r['corr']:+.3f}", delta_color="off")

        xs = np.linspace(d["x"].min(), d["x"].max(), 50)
        lines = pd.concat(
            [
                pd.DataFrame({"x": xs, "y": r[f"{m}_int"] + r[m] * xs, "model": m})
                for m in ("OLS", "Huber", "Theil-Sen")
            ],
            ignore_index=True,
        )
        pts = (
            alt.Chart(d)
            .mark_circle(size=80, opacity=0.7)
            .encode(
                x=alt.X("x:Q", title="experience_years", scale=alt.Scale(zero=False)),
                y=alt.Y("y:Q", title="income", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(domain=["normal", "the outlier"], range=["#ADB5BD", "#E8590C"]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                size=alt.Size("kind:N", scale=alt.Scale(domain=["normal", "the outlier"], range=[70, 400]), legend=None),
            )
        )
        lc = (
            alt.Chart(lines)
            .mark_line(strokeWidth=3)
            .encode(
                x="x:Q",
                y="y:Q",
                color=alt.Color(
                    "model:N",
                    scale=alt.Scale(domain=["OLS", "Huber", "Theil-Sen"], range=["#4C6EF5", "#F08C00", "#0CA678"]),
                    legend=alt.Legend(orient="top", title=None),
                ),
            )
        )
        st.altair_chart((pts + lc).properties(height=380), width="stretch")

        drift = abs(r["OLS"] - clean_r["OLS"]) / max(abs(clean_r["OLS"]), 1) * 100
        if drift > 25:
            st.markdown(
                f":red-badge[انحراف {drift:.0f}%] نقطة **واحدة** من أصل "
                f"{len(d)} غيّرت ميل OLS بأكثر من الربع. هذه هي النقطة المؤثّرة "
                "(influential point) التي تقيسها مسافة كوك."
            )
        else:
            st.markdown(f":green-badge[انحراف {drift:.0f}%] ما زال التأثير محدودًا.")

    play_animation("outlier_leverage", len(SHIFTS), _render_leverage, fps=0.8, label="بُعد الشاذّة")

    st.subheader("وماذا تفعل المعالجات بالتوزيع؟", divider="orange")
    inc = clean_customers(500)["income"].to_numpy(dtype=float)
    inc = np.append(inc, inc.max() * np.array([4.0, 5.5, 7.0]))
    lo, hi = np.percentile(inc, [1, 99])
    variants = pd.concat(
        [
            pd.DataFrame({"value": inc, "treatment": "original (with outliers)"}),
            pd.DataFrame({"value": np.clip(inc, lo, hi), "treatment": "winsorized 1-99%"}),
            pd.DataFrame({"value": np.log1p(inc), "treatment": "log1p"}),
        ],
        ignore_index=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(
            charts.hist(
                variants[variants["treatment"] != "log1p"],
                "value",
                bins=50,
                color="treatment",
                height=280,
            ),
            width="stretch",
        )
    with c2:
        st.altair_chart(
            charts.hist(
                variants[variants["treatment"] == "log1p"], "value", bins=50, height=280
            ),
            width="stretch",
        )
    stats_tbl = (
        variants.groupby("treatment")["value"]
        .agg(["mean", "std", "skew", "max"])
        .round(2)
        .reset_index()
    )
    st.dataframe(stats_tbl, width="stretch", hide_index=True)
    st.caption(
        "Winsorize يُبقي عدد الصفوف ثابتًا ويقلّل التطرّف؛ `log1p` يُبقي كل "
        "المعلومة ويغيّر المقياس فقط — ولهذا يُفضَّل حين يكون الانحراف طبيعيًا "
        "في الظاهرة."
    )

# ============================================================== القرار
with t_decide:
    st.subheader("شجرة قرار الشواذ", divider="orange")
    ui.decision_table(
        [
            ("القيمة مستحيلة فيزيائيًا (عمر 199)", ":violet-badge[حوّلها إلى NaN ثم عوّض]", "خطأ إدخال، وليست ظاهرة"),
            ("خطأ وحدات أو فاصلة عشرية", ":green-badge[صحّحها]", "المعلومة سليمة، الترميز خاطئ"),
            ("نادرة لكنها حقيقية (احتيال، مرض نادر)", ":red-badge[احتفظ بها — قد تكون الهدف]", "حذفها يحذف الظاهرة موضوع الدراسة"),
            ("ذيل طبيعي في توزيع منحرف (الدخل)", ":blue-badge[حوّل (log/Yeo-Johnson)]", "ليست شواذ، بل شكل التوزيع"),
            ("نقاط قليلة تقلب نموذجًا خطّيًا", ":violet-badge[نموذج أمتن أو Winsorize]", "تغيير الأداة أنظف من حذف الدليل"),
            ("ستستعمل شجرة / غابة / GBM", ":green-badge[اتركها غالبًا]", "الأشجار غير حسّاسة للقيم المتطرّفة"),
            ("ستستعمل KNN / SVM / K-Means", ":orange-badge[عالجها + RobustScaler]", "المسافات تنهار أمام القيم المتطرّفة"),
            ("مجموعة متماسكة من الشواذ", ":blue-badge[افحص: هل هي مجتمع آخر؟]", "قد تستحق نموذجًا منفصلًا أو متغيّر مجموعة"),
            ("لا تعرف السبب", ":orange-badge[احتفظ + علم (flag) + حلّل الحساسية]", "القرار المؤجّل أفضل من حذف غير مبرّر"),
            ("بيانات الاختبار / الإنتاج", ":red-badge[لا تحذف — اقصص بحدود التدريب]", "الاختبار يمثّل الواقع"),
        ]
    )

    ui.tip(
        "السؤال الذهبي قبل أي حذف: **«هل أستطيع أن أكتب في ورقتي البحثية "
        "سبب حذف هذا الصف بالتحديد؟»** إن كان الجواب لا — لا تحذفه."
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("scikit-learn — Preprocessing data (RobustScaler, PowerTransformer)", "https://scikit-learn.org/stable/modules/preprocessing.html"),
            ("scikit-learn — Robust linear estimators (Huber, RANSAC, Theil-Sen)", "https://scikit-learn.org/stable/auto_examples/linear_model/plot_robust_fit.html"),
            ("scikit-learn — HuberRegressor", "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html"),
            ("SciPy — winsorize", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html"),
            ("Rousseeuw & Leroy — Robust Regression and Outlier Detection", "https://doi.org/10.1002/0471725382"),
            ("Leys et al. (2013) — Detecting outliers: do not use SD around the mean", "https://doi.org/10.1016/j.jesp.2013.03.013"),
        ]
    )
    ui.terms(
        [
            ("Winsorizing", "قصّ القيم عند مئينات محدّدة"),
            ("Clipping", "تحديد سقف وأرضية للقيم"),
            ("Robust regression", "انحدار أمتن"),
            ("Huber loss", "خسارة هوبر (تربيعية ثم خطّية)"),
            ("Leverage point", "نقطة رافعة في فضاء المتغيّرات"),
            ("Influential point", "نقطة مؤثّرة على نتائج النموذج"),
            ("Sensitivity analysis", "تحليل الحساسية"),
        ]
    )
