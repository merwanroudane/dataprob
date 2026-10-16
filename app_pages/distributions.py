"""تحويل التوزيعات والتجزيء."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer, PowerTransformer, QuantileTransformer

from core import charts, ui
from core.anim import play_animation
from core.datasets import clean_customers

ui.page_header(
    "تحويل التوزيعات والتجزيء",
    "التقييس يغيّر **الوحدة**؛ التحويل يغيّر **الشكل**. هنا نتعلّم متى يكون "
    "تغيير الشكل ضروريًا، ومتى يكون تشويهًا لظاهرة حقيقية — ومتى نحوّل "
    "المتغيّر الرقمي إلى فئات أصلًا.",
    icon=":material/area_chart:",
    badges=[
        ":blue-badge[Skewness]",
        ":green-badge[log / sqrt]",
        ":orange-badge[Box-Cox]",
        ":violet-badge[Yeo-Johnson]",
        ":red-badge[Binning]",
    ],
)


@st.cache_data(ttl=1800, max_entries=4)
def income_sample(n: int = 800) -> np.ndarray:
    return clean_customers(n)["income"].to_numpy(dtype=float)


t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. الالتواء والتفلطح — ماذا يقيسان؟", divider="blue")
    st.latex(r"\text{Skewness} = E\left[\left(\frac{X-\mu}{\sigma}\right)^3\right], \qquad \text{Kurtosis} = E\left[\left(\frac{X-\mu}{\sigma}\right)^4\right] - 3")
    st.markdown(
        """
- **Skewness (الالتواء)**: عدم التماثل. موجب = ذيل أيمن طويل (الدخل، الأسعار،
  مدّة الانتظار). سالب = ذيل أيسر (درجات امتحان سهل).
  - $|skew| < 0.5$ ⇒ شبه متماثل.
  - $0.5 - 1$ ⇒ التواء معتدل.
  - $> 1$ ⇒ التواء قوي، التحويل مُرجَّح أن يساعد.
- **Kurtosis (التفلطح الزائد)**: ثقل الذيول. قيمة عالية = شواذ أكثر مما
  يتوقّعه التوزيع الطبيعي.
"""
    )

    st.subheader("٢. تصحيح شائع: الطبيعية مطلوبة أين بالضبط؟", divider="red")
    st.markdown(
        """
:red-badge[خرافة] «يجب أن تكون كل المتغيّرات موزّعة طبيعيًا قبل التعلّم الآلي.»

**الحقيقة:**
- الانحدار الخطّي (OLS) يفترض طبيعية **البواقي (residuals)** — لا المتغيّرات
  المستقلّة. ولا يفترض شيئًا عن توزيع $X$ إطلاقًا.
- النماذج الشجرية و KNN و Random Forest **لا تفترض أي توزيع**.
- الاختبارات البارامترية (t، ANOVA) هي التي تفترض الطبيعية — وغالبًا على
  توزيع المعاينة لا على البيانات نفسها (مبرهنة النهاية المركزية).

**إذن لماذا نحوّل؟** لأسباب عملية لا شعائرية:
1. تقليل **سطوة الذيل الطويل** وبالتالي تأثير القيم المتطرّفة.
2. **تخطيط العلاقة**: `log(income)` كثيرًا ما تكون علاقتها بالهدف خطّية
   بينما `income` الخام ليست كذلك.
3. **استقرار التباين (variance stabilisation)**: شرط صحّة OLS الحقيقي
   (homoscedasticity).
4. **تحسين تقارب** النماذج القائمة على التدرّج.
"""
    )

    st.subheader("٣. التحويلات المتاحة", divider="green")
    st.markdown(
        """
| التحويل | الصيغة | يتطلّب | ملاحظة |
|---|---|---|---|
| `log` | $\\ln(x)$ | $x > 0$ | الأقوى للالتواء الأيمن |
| `log1p` | $\\ln(1+x)$ | $x \\ge 0$ | يتحمّل الأصفار — المفضّل عمليًا |
| `sqrt` | $\\sqrt{x}$ | $x \\ge 0$ | أخف من log؛ مناسب لبيانات العدّ |
| `reciprocal` | $1/x$ | $x \\ne 0$ | قوي جدًا، ويقلب الترتيب |
| **Box-Cox** | $\\frac{x^{\\lambda}-1}{\\lambda}$ | $x > 0$ فقط | يختار $\\lambda$ بأرجحية عظمى |
| **Yeo-Johnson** | امتداد لـ Box-Cox | أي قيمة (سالبة وصفر) | :green-badge[الخيار العام الأفضل] |
| `QuantileTransformer` | رتبة ← توزيع هدف | — | غير خطّي؛ يفرض الشكل فرضًا |
| `RankGauss` | رتبة ← طبيعي | — | شائع في المسابقات مع الشبكات |

عند $\\lambda = 0$ يصبح Box-Cox هو اللوغاريتم تمامًا، وعند $\\lambda = 1$
لا يغيّر الشكل (إزاحة فقط).
"""
    )

    st.subheader("٤. التجزيء (Discretization / Binning)", divider="orange")
    ui.concept_grid(
        [
            (
                "عرض متساوٍ (uniform)",
                "شرائح متساوية العرض. بسيط، لكن مع التوزيعات المنحرفة تتكدّس "
                "كل البيانات في شريحة واحدة.",
                "blue",
            ),
            (
                "تكرار متساوٍ (quantile)",
                "كل شريحة بنفس عدد المشاهدات. :green[الأكثر استعمالًا]؛ "
                "لكن حدود الشرائح تُتعلَّم من البيانات ⇒ **داخل Pipeline**.",
                "green",
            ),
            (
                "k-means",
                "يحدّد الحدود بتجميع القيم. يحترم بنية التكدّس الطبيعية في "
                "البيانات.",
                "orange",
            ),
            (
                "حدود المجال (domain)",
                "شرائح العمر الطبية، شرائح الضريبة، BMI. :green[الأفضل حين "
                "توجد] لأنها قابلة للتفسير ومستقرّة عبر الزمن.",
                "violet",
            ),
        ],
        n_cols=4,
    )
    st.markdown(
        """
**لماذا نجزّئ؟** لالتقاط علاقة غير خطّية في نموذج خطّي، ولتقليل أثر الشواذ،
ولقابلية التفسير (شرائح عمرية يفهمها الطبيب).

**ولماذا لا نجزّئ؟** لأن التجزيء **يفقد معلومة**: 29.9 و 30.1 تصبحان في
شريحتين مختلفتين بينما 30.1 و 39.9 في شريحة واحدة. والحدود التعسّفية قد
تنتج قفزات مصطنعة. :red-badge[القاعدة] مع النماذج الشجرية التجزيء شبه عديم
الفائدة — الشجرة تتعلّم الحدود المثلى بنفسها.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. قياس الالتواء أولًا", divider="blue")
    ui.code(
        """
from scipy import stats

for c in df.select_dtypes("number"):
    s = df[c].dropna()
    print(f"{c:20s} skew={s.skew():+.2f}  kurtosis={s.kurtosis():+.2f}")

# اختبار طبيعية (للعيّنات < 5000)
stat, p = stats.shapiro(df["income"].dropna().sample(500, random_state=0))
print(f"Shapiro-Wilk p={p:.4g}")      # p صغيرة ⇒ ليست طبيعية
""",
        caption="في العيّنات الكبيرة ترفض اختبارات الطبيعية الفرضية دائمًا "
        "لأتفه انحراف. اعتمد على **الرسم** و **قيمة الالتواء** أكثر من قيمة p.",
    )

    st.subheader("٢. التحويلات", divider="green")
    ui.code(
        """
import numpy as np
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, FunctionTransformer

# يدويًا
df["log_income"] = np.log1p(df["income"])

# داخل Pipeline (قابل للعكس وآمن)
log_tf = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)

# Yeo-Johnson: يقدّر lambda من التدريب ويطبّقها على الاختبار
pt = PowerTransformer(method="yeo-johnson", standardize=True).fit(X_train[["income"]])
print("lambda =", pt.lambdas_)
X_train_t = pt.transform(X_train[["income"]])

# Box-Cox: للقيم الموجبة فقط
pt_bc = PowerTransformer(method="box-cox")        # يرمي خطأ إن وُجد صفر أو سالب

# QuantileTransformer: يفرض الشكل فرضًا
qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0)
"""
    )

    st.subheader("٣. التجزيء", divider="orange")
    ui.code(
        """
from sklearn.preprocessing import KBinsDiscretizer

# تكرار متساوٍ، ترميز ترتيبي
kb = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
X_binned = kb.fit_transform(X_train[["income"]])
print(kb.bin_edges_)        # الحدود المُتعلَّمة — من التدريب فقط

# أو One-Hot مباشرة
KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="kmeans")

# pandas: حدود مجال صريحة (آمنة قبل التقسيم لأنها معرفة مسبقة)
bins = [0, 18, 30, 45, 60, 200]
labels = ["<18", "18-29", "30-44", "45-59", "60+"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

# شرائح بالتكرار (مُتعلَّمة ⇒ انتبه للتسريب)
df["income_q"] = pd.qcut(df["income"], q=5, labels=False, duplicates="drop")
""",
        caption=":red-badge[تمييز مهم] `pd.cut` بحدود تعرفها سلفًا = آمن قبل "
        "التقسيم. `pd.qcut` يتعلّم الحدود من البيانات = يجب أن يكون داخل "
        "`Pipeline` بعد التقسيم.",
    )

    st.subheader("٤. هل ساعد التحويل فعلًا؟", divider="violet")
    ui.code(
        """
before = df["income"].skew()
after = np.log1p(df["income"]).skew()
print(f"skew: {before:+.2f} ← {after:+.2f}")

# وفي النهاية: المعيار هو الأداء بتحقّق متقاطع، لا جمال الرسم
from sklearn.model_selection import cross_val_score
for name, pipe in {"raw": pipe_raw, "log": pipe_log, "yeo": pipe_yeo}.items():
    print(name, cross_val_score(pipe, X_train, y_train, cv=5).mean())
"""
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("أ. مسح معامل λ في Box-Cox", divider="violet")
    st.markdown(
        "نطبّق تحويل Box-Cox بقيم $\\lambda$ متدرّجة على عمود الدخل (منحرف "
        "يمينًا). راقب كيف ينتقل الالتواء من موجب كبير إلى قرب الصفر ثم إلى "
        "سالب — و $\\lambda$ المثلى هي التي يختارها `PowerTransformer` تلقائيًا."
    )

    inc = income_sample()
    LAMBDAS = [-1.0, -0.5, -0.25, 0.0, 0.15, 0.25, 0.5, 0.75, 1.0]

    opt_lambda = float(stats.boxcox_normmax(inc, method="mle"))

    def _boxcox(x: np.ndarray, lam: float) -> np.ndarray:
        return np.log(x) if abs(lam) < 1e-9 else (np.power(x, lam) - 1) / lam

    def _render_lambda(frame: int) -> None:
        lam = LAMBDAS[frame]
        v = _boxcox(inc, lam)
        v = (v - v.mean()) / v.std()
        sk = float(pd.Series(v).skew())

        k1, k2, k3 = st.columns(3)
        k1.metric("λ الحالية", f"{lam:+.2f}")
        k2.metric("الالتواء بعد التحويل", f"{sk:+.3f}", delta=f"من {pd.Series(inc).skew():+.2f}", delta_color="off")
        k3.metric("λ المثلى (MLE)", f"{opt_lambda:+.3f}", help="ما يقدّره PowerTransformer تلقائيًا")

        d = pd.DataFrame({"value": v})
        base = charts.hist(d, "value", bins=50, height=300)
        st.altair_chart(base + charts.vrule(0, color="#5C6B84"), width="stretch")

        if abs(sk) < 0.2:
            st.markdown(":green-badge[شبه متماثل] هذه منطقة λ المناسبة.")
        elif sk > 0:
            st.markdown(":orange-badge[ما زال ملتويًا يمينًا] λ أكبر من اللازم — جرّب قيمة أصغر.")
        else:
            st.markdown(":red-badge[انقلب إلى التواء يسار] λ صغيرة جدًا — بالغنا في التحويل.")

    play_animation("boxcox_lambda", len(LAMBDAS), _render_lambda, fps=0.8, label="λ")

    st.subheader("ب. مقارنة التحويلات على نفس العمود", divider="orange")
    TF = [
        ("الأصل", lambda x: x),
        ("sqrt", np.sqrt),
        ("log1p", np.log1p),
        ("1/x (reciprocal)", lambda x: 1 / np.clip(x, 1, None)),
        ("Box-Cox (MLE)", lambda x: PowerTransformer(method="box-cox").fit_transform(x.reshape(-1, 1)).ravel()),
        ("Yeo-Johnson", lambda x: PowerTransformer(method="yeo-johnson").fit_transform(x.reshape(-1, 1)).ravel()),
        ("Quantile → normal", lambda x: QuantileTransformer(output_distribution="normal", n_quantiles=400, random_state=0).fit_transform(x.reshape(-1, 1)).ravel()),
    ]

    def _render_tf(frame: int) -> None:
        name, fn = TF[frame]
        v = fn(inc.copy())
        s = pd.Series(v)

        k1, k2, k3 = st.columns(3)
        k1.metric("الالتواء (skew)", f"{s.skew():+.3f}")
        k2.metric("التفلطح (kurtosis)", f"{s.kurtosis():+.3f}")
        k3.metric("المدى", f"{s.min():.2f} → {s.max():.2f}")

        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(charts.hist(pd.DataFrame({"v": v}), "v", bins=50, height=280), width="stretch")
        with c2:
            # Q-Q plot: كلما اقتربت النقاط من الخط كان التوزيع أقرب للطبيعي
            z = (v - v.mean()) / v.std()
            theo = stats.norm.ppf((np.arange(len(z)) + 0.5) / len(z))
            qq = pd.DataFrame({"theoretical": theo, "sample": np.sort(z)})
            line = pd.DataFrame({"theoretical": [-3, 3], "sample": [-3, 3]})
            ch = (
                alt.Chart(qq).mark_circle(size=28, opacity=0.6).encode(
                    x=alt.X("theoretical:Q", title="theoretical quantiles"),
                    y=alt.Y("sample:Q", title="sample quantiles"),
                )
                + alt.Chart(line).mark_line(color="#E8590C", strokeDash=[6, 4]).encode(
                    x="theoretical:Q", y="sample:Q"
                )
            ).properties(height=280, title="Q-Q plot")
            st.altair_chart(ch, width="stretch")

        notes = {
            "الأصل": "التواء يميني واضح — نموذجي لبيانات الدخل والأسعار.",
            "sqrt": "تخفيف معتدل؛ مناسب لبيانات العدّ (counts) أكثر من الدخل.",
            "log1p": "الأكثر استعمالًا للالتواء اليميني؛ يتحمّل الأصفار ويبقى قابلًا للتفسير (تغيّر نسبي).",
            "1/x (reciprocal)": "تحويل عنيف **يقلب الترتيب** — نادرًا ما يكون الخيار الصحيح.",
            "Box-Cox (MLE)": "يختار λ بأقصى أرجحية ⇒ أفضل تقريب للطبيعية. :orange[قيد] القيم الموجبة فقط.",
            "Yeo-Johnson": "نفس الفكرة لكن يقبل الأصفار والسوالب ⇒ الخيار العام الأأمن.",
            "Quantile → normal": "يفرض الشكل الطبيعي فرضًا عبر الرتب. :red[حذر] يشوّه المسافات ويحتاج عيّنة كبيرة، وقد يبالغ في الملاءمة.",
        }
        st.markdown(f"**:blue-badge[{name}]** {notes[name]}")

    play_animation("tf_tour", len(TF), _render_tf, fps=0.5, label="التحويل", frame_labels=[t[0] for t in TF])

    st.subheader("ج. التجزيء: ثلاث استراتيجيات", divider="green")
    nb = st.slider("عدد الشرائح", 3, 10, 5, key="nbins")
    strategies = ["uniform", "quantile", "kmeans"]
    cols = st.columns(3)
    for col, strat in zip(cols, strategies):
        kb = KBinsDiscretizer(n_bins=nb, encode="ordinal", strategy=strat, subsample=None)
        b = kb.fit_transform(inc.reshape(-1, 1)).ravel()
        counts = pd.Series(b).value_counts().sort_index().reset_index()
        counts.columns = ["bin", "count"]
        counts["bin"] = counts["bin"].astype(int).astype(str)
        with col:
            st.markdown(f"###### `strategy=\"{strat}\"`")
            st.altair_chart(charts.bar(counts, "bin", "count", height=220, sort_desc=False), width="stretch")
    st.caption(
        "`uniform` يترك شرائح شبه فارغة مع التوزيعات المنحرفة؛ `quantile` "
        "يوزّع المشاهدات بالتساوي؛ `kmeans` يتبع تكدّس القيم الطبيعي."
    )

# ============================================================== القرار
with t_decide:
    st.subheader("أحوّل أم لا؟ وبماذا؟", divider="orange")
    ui.decision_table(
        [
            ("|skew| < 0.5", "لا تحوّل", "التوزيع شبه متماثل أصلًا"),
            ("التواء يميني موجب، قيم موجبة", "`log1p` أو Box-Cox", "أقوى تأثير وأبسط تفسير"),
            ("توجد أصفار أو قيم سالبة", "Yeo-Johnson", "Box-Cox يرفض غير الموجب"),
            ("بيانات عدّ (counts)", "`sqrt` أو `log1p` أو نموذج بواسون", "تباين متناسب مع المتوسط"),
            ("نموذج شجري", "التحويل غير ضروري", "التقسيم رتيب ⇒ لا يتأثّر بالتحويلات الرتيبة"),
            ("نموذج خطّي والعلاقة غير خطّية", "حوّل أو جزّئ", "يسمح للنموذج الخطّي بالتقاط الانحناء"),
            ("تحتاج تفسيرًا للمعاملات", "`log` (مرونة) أو لا تحوّل", "`log-log` يعطي تفسير مرونة مباشر"),
            ("توزيع متعدّد القمم", "التجزيء أو QuantileTransformer", "التحويلات الأسّية لا تعالج تعدّد القمم"),
            ("حدود معروفة في المجال", "`pd.cut` بحدود صريحة", "قابل للتفسير ومستقرّ ولا يُسرّب"),
            ("الهدف `y` منحرف في الانحدار", "`TransformedTargetRegressor`", "يحوّل ويعكس آليًا دون تسريب"),
        ],
        headers=("الحالة", "القرار", "السبب"),
    )

    ui.tip(
        "التحويل ليس هدفًا بذاته. المعيار الوحيد: **هل تحسّن أداء التحقّق "
        "المتقاطع أو صحّة الافتراضات؟** إن لم يتحسّن شيء، فقد خسرت قابلية "
        "التفسير بلا مقابل."
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("scikit-learn — Non-linear transformation (Quantile & Power)", "https://scikit-learn.org/stable/modules/preprocessing.html#non-linear-transformation"),
            ("scikit-learn — Map data to a normal distribution", "https://scikit-learn.org/stable/auto_examples/preprocessing/plot_map_data_to_normal.html"),
            ("scikit-learn — KBinsDiscretizer", "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html"),
            ("scikit-learn — TransformedTargetRegressor", "https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html"),
            ("SciPy — boxcox / yeojohnson", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html"),
            ("Box & Cox (1964) — An Analysis of Transformations", "https://doi.org/10.1111/j.2517-6161.1964.tb00553.x"),
            ("Yeo & Johnson (2000) — A new family of power transformations", "https://doi.org/10.1093/biomet/87.4.954"),
        ]
    )
    ui.terms(
        [
            ("Skewness", "الالتواء"),
            ("Kurtosis", "التفلطح"),
            ("Heteroscedasticity", "عدم تجانس التباين"),
            ("Box-Cox transform", "تحويل بوكس-كوكس"),
            ("Yeo-Johnson transform", "تحويل يو-جونسون"),
            ("Discretization / Binning", "التجزيء إلى شرائح"),
            ("Q-Q plot", "رسم المئينات المقارن"),
        ]
    )
