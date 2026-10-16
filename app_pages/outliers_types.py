"""أنواع القيم الشاذة."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import charts, ui
from core.anim import play_animation
from core.datasets import outlier_demo

ui.page_header(
    "أنواع القيم الشاذة (Outliers)",
    "«قيمة شاذة» ليست حكمًا واحدًا. التعريف الكلاسيكي لـ *Hawkins (1980)*: "
    "**مشاهدة تنحرف كثيرًا عن بقيّة المشاهدات حتى تثير الشك في أنها وُلّدت "
    "بآلية مختلفة.** الكلمة المفتاح هي **آلية مختلفة** — وهذا ما يحدّد القرار.",
    icon=":material/scatter_plot:",
    badges=[
        ":blue-badge[Global]",
        ":green-badge[Contextual]",
        ":orange-badge[Collective]",
        ":violet-badge[Multivariate]",
        ":red-badge[Influential]",
    ],
)

t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. التصنيف حسب الشكل", divider="blue")
    ui.concept_grid(
        [
            (
                "Global / Point outlier",
                "قيمة بعيدة عن **كل** البيانات: راتب 50 مليون في جدول رواتب "
                "متوسطها 60 ألف.\n\n"
                "الأسهل كشفًا: z-score، IQR، المئينات.",
                "blue",
            ),
            (
                "Contextual / Conditional outlier",
                "القيمة **عادية في ذاتها، شاذة في سياقها**: استهلاك تدفئة مرتفع "
                "— عادي في يناير، شاذ في يوليو. إنفاق 2000 دج — عادي لعميل "
                "كبير، شاذ لطالب.\n\n"
                "لا يُكشف إلا بتقسيم السياق (المجموعة، الموسم، المنطقة).",
                "green",
            ),
            (
                "Collective outlier",
                "كل نقطة وحدها طبيعية، لكن **المجموعة معًا** شاذة: نبضات قلب "
                "منتظمة لكن بتردّد غريب لمدّة دقيقتين؛ 50 معاملة صغيرة في دقيقة "
                "واحدة.\n\n"
                "يحتاج تحليل تسلسل أو نافذة زمنية، لا نقطة مفردة.",
                "orange",
            ),
            (
                "Multivariate outlier",
                "كل متغيّر في مداه الطبيعي، لكن **التركيبة** مستحيلة: عمر 16 سنة "
                "مع خبرة 20 سنة؛ طول 150 سم مع وزن 140 كغ.\n\n"
                "لا يُكشف أبدًا بفحص عمود عمود — يحتاج Mahalanobis أو LOF أو "
                "Isolation Forest.",
                "violet",
            ),
        ],
        n_cols=2,
    )

    st.subheader("٢. التصنيف حسب السبب — وهو الأهم للقرار", divider="red")
    st.markdown(
        """
| السبب | مثال | القرار النموذجي |
|---|---|---|
| **خطأ إدخال (data entry)** | عمر = 199؛ فاصلة عشرية في غير موضعها | :green[صحّح إن أمكن، وإلّا حوّله إلى `NaN`] |
| **خطأ قياس (measurement)** | حسّاس معطّل يعطي 0 أو قيمة التشبّع | :green[حوّله إلى `NaN` وعوّض] |
| **خطأ معالجة (processing)** | دمج خاطئ، وحدات مختلطة (دج/أورو)، ضرب في 1000 | :green[أصلح المصدر — لا تحذف] |
| **خطأ أخذ عيّنة (sampling)** | صفّ من مجتمع آخر دخل الجدول بالخطأ | :orange[احذفه مع التوثيق] |
| **تباين طبيعي (natural)** | راتب مدير تنفيذي؛ مدينة ضخمة بين مدن صغيرة | :red[**احتفظ به**] |
| **حدث نادر حقيقي (novelty)** | احتيال، عطب، مرض نادر | :red[**هو هدفك أصلًا — لا تحذفه أبدًا**] |

:red-badge[القاعدة] السؤال ليس «هل هي بعيدة؟» بل **«هل وُلّدت بآلية مختلفة
عن بقيّة البيانات؟»** — الإحصاء يجيب عن السؤال الأول فقط؛ أنت من يجيب عن
الثاني.
"""
    )

    st.subheader("٣. ثلاثة مفاهيم يُخلط بينها دائمًا", divider="orange")
    ui.concept_grid(
        [
            (
                "Outlier detection",
                "البحث عن الشواذ **داخل** مجموعة تدريب ملوّثة أصلًا. "
                "النموذج يتعلّم من بيانات فيها شواذ. "
                "مثال: `IsolationForest`, `LocalOutlierFactor(novelty=False)`.",
                "blue",
            ),
            (
                "Novelty detection",
                "التدريب على بيانات **نظيفة** فقط، ثم كشف ما يخرج عنها لاحقًا. "
                "مثال: `OneClassSVM`, `LocalOutlierFactor(novelty=True)`. "
                "هذا هو سيناريو مراقبة الإنتاج.",
                "green",
            ),
            (
                "Influential point (leverage)",
                "نقطة **تغيّر نتائج النموذج** بشكل كبير إن حُذفت — وقد لا تكون "
                "شاذة بصريًا. تُقاس بـ Cook's distance و leverage و DFFITS. "
                "شاذّة ≠ مؤثّرة، والعكس صحيح.",
                "violet",
            ),
        ],
        n_cols=3,
    )
    st.latex(r"D_i = \frac{\sum_{j=1}^{n} (\hat{y}_j - \hat{y}_{j(i)})^2}{p \cdot MSE}")
    st.caption(
        "مسافة كوك $D_i$: كم تتغيّر كل التنبّؤات لو حذفنا المشاهدة $i$. "
        "عتبة شائعة: $D_i > 4/n$ تستحق الفحص."
    )

    ui.beginner_note(
        """
تخيّل فصلًا دراسيًا: طالب طوله ٢.١٠ م هو **شاذ عام** (أطول من الجميع).
طالب طوله ١.٧٥ م في صف أطفال خمس سنوات هو **شاذ سياقي**. وطالب طوله ١.٥٠ م
ووزنه ١٢٠ كغ **شاذ متعدّد المتغيّرات** — كل رقم وحده معقول، لكن التركيبة لا.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. أول فحص بصري وعددي", divider="blue")
    ui.code(
        """
df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T

# أكبر وأصغر 10 قيم — أسرع طريقة لاكتشاف أخطاء الإدخال
print(df["income"].nlargest(10).to_list())
print(df["income"].nsmallest(10).to_list())

df["income"].plot(kind="box")
df["income"].plot(kind="hist", bins=60)
"""
    )

    st.subheader("٢. الشاذ السياقي: قارن داخل المجموعة", divider="green")
    ui.code(
        """
# z-score داخل كل مدينة بدل الجدول كاملًا
g = df.groupby("city")["spending"]
df["z_in_city"] = (df["spending"] - g.transform("mean")) / g.transform("std")
df[df["z_in_city"].abs() > 3]

# نسخة أمتن: باستعمال الوسيط والـ MAD داخل المجموعة
med = g.transform("median")
mad = g.transform(lambda s: (s - s.median()).abs().median())
df["robust_z_in_city"] = 0.6745 * (df["spending"] - med) / mad
"""
    )

    st.subheader("٣. الشاذ متعدّد المتغيّرات: اختبار منطقي سريع", divider="violet")
    ui.code(
        """
# قواعد المجال تسبق أي إحصاء
impossible = df.query("experience_years > age - 14 or spending > income * 2 or age > 110")
print(len(impossible), "صفّ مستحيل منطقيًا")

# نسب مشتقّة تكشف ما لا يكشفه العمود وحده
df["spend_ratio"] = df["spending"] / df["income"]
df[df["spend_ratio"] > 1.5]
"""
    )

    st.subheader("٤. النقاط المؤثّرة في الانحدار", divider="red")
    ui.code(
        """
import statsmodels.api as sm

X = sm.add_constant(df[["age", "experience_years"]])
res = sm.OLS(df["income"], X).fit()
infl = res.get_influence()

cooks = infl.cooks_distance[0]
leverage = infl.hat_matrix_diag
student = infl.resid_studentized_external

flagged = df[(cooks > 4 / len(df))]
print(len(flagged), "مشاهدة مؤثّرة تستحق الفحص")
""",
        caption="النقطة المؤثّرة تستحق **فحصًا**، لا حذفًا آليًا.",
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("شاهد الأنواع الأربعة على نفس البيانات", divider="violet")
    st.markdown(
        "في كل إطار نزرع نوعًا مختلفًا من الشواذ في سحابة نقاط طبيعية "
        "(`y ≈ 2x`) ونرى كيف يبدو — ولماذا تفشل بعض طرق الكشف."
    )

    KINDS = [
        (
            "global",
            "Global outliers",
            "نقاط بعيدة في كلا البعدين. **يكشفها** z-score و IQR بسهولة.",
        ),
        (
            "contextual",
            "Contextual outliers",
            "قيمة `x` طبيعية و `y` طبيعية **كلٌّ على حدة**، لكن التركيبة تخالف "
            "العلاقة السائدة. **يفشل** z-score الأحادي تمامًا.",
        ),
        (
            "collective",
            "Collective outliers",
            "عنقود صغير متماسك بعيد عن النمط. كل نقطة فيه قد تبدو مقبولة، "
            "لكن **وجودها معًا** هو الشذوذ. يكشفه DBSCAN و LOF.",
        ),
        (
            "multivariate",
            "Multivariate outliers",
            "داخل المدى الطبيعي لكل متغيّر، لكن خارج **سحابة الارتباط**. "
            "يحتاج Mahalanobis أو LOF أو Isolation Forest.",
        ),
    ]

    def _render_kind(frame: int) -> None:
        kind, title, note = KINDS[frame]
        d = outlier_demo(kind)
        base = d[d["type"] == "normal"]

        left, right = st.columns([3, 2])
        with left:
            sc = charts.scatter(d, "x", "y", color="type", height=330)
            st.altair_chart(sc + charts.reg_line(base, "x", "y"), width="stretch")
        with right:
            st.markdown("###### هل يكشفها z-score على `y` وحده؟")
            z = (d["y"] - d["y"].mean()) / d["y"].std()
            caught = (z.abs() > 3) & (d["type"] != "normal")
            total = (d["type"] != "normal").sum()
            st.metric(
                "نقاط شاذة كشفها z-score",
                f"{int(caught.sum())} / {int(total)}",
                delta="كشف ناقص" if caught.sum() < total else "كشف كامل",
                delta_color="inverse" if caught.sum() < total else "normal",
            )
            hist = charts.hist(d.assign(z=z), "z", bins=40, height=220)
            st.altair_chart(hist + charts.vrule(3) + charts.vrule(-3), width="stretch")

        st.markdown(f"**:violet-badge[{title}]** {note}")

    play_animation(
        "outlier_kinds",
        len(KINDS),
        _render_kind,
        fps=0.5,
        label="النوع",
        frame_labels=[k[1] for k in KINDS],
    )

# ============================================================== القرار
with t_decide:
    st.subheader("النوع ⇐ طريقة الكشف", divider="orange")
    ui.decision_table(
        [
            ("Global", "z-score / IQR / المئينات", "الانحراف واضح في بعد واحد"),
            ("Contextual", "تقييس داخل المجموعة (groupby) أو نموذج شرطي", "«الطبيعي» يختلف باختلاف السياق"),
            ("Collective", "DBSCAN / LOF / تحليل نوافذ زمنية", "الشذوذ خاصية مجموعة لا نقطة"),
            ("Multivariate", "Mahalanobis (MCD) / LOF / IsolationForest", "يعتمد على بنية الارتباط بين الأعمدة"),
            ("Influential", "Cook's distance / leverage / DFFITS", "يقيس الأثر على النموذج لا البعد عن المركز"),
            ("Novelty في الإنتاج", "OneClassSVM / LOF(novelty=True)", "التدريب على «الطبيعي» فقط"),
        ],
        headers=("نوع الشاذ", "أداة الكشف", "السبب"),
    )

    ui.danger(
        "لا تطبّق كشفًا آليًا على **كل** الأعمدة ثم تحذف كل صفّ مُعلَّم. "
        "مع 20 عمودًا وقاعدة ٣ انحرافات، تتوقّع أن يُعلَّم نحو **٥٪ من الصفوف** "
        "بالصدفة وحدها حتى لو كانت البيانات مثالية."
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("Hawkins (1980) — Identification of Outliers", "https://doi.org/10.1007/978-94-015-3994-4"),
            ("scikit-learn — Novelty and Outlier Detection", "https://scikit-learn.org/stable/modules/outlier_detection.html"),
            ("statsmodels — Outlier and influence measures", "https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.OLSInfluence.html"),
            ("Aggarwal — Outlier Analysis (2nd ed.)", "https://doi.org/10.1007/978-3-319-47578-3"),
            ("Chandola, Banerjee & Kumar (2009) — Anomaly Detection: A Survey", "https://doi.org/10.1145/1541880.1541882"),
        ]
    )
    ui.terms(
        [
            ("Outlier", "قيمة شاذة"),
            ("Anomaly", "شذوذ (غالبًا يُستعمل مرادفًا)"),
            ("Novelty detection", "كشف الجديد/غير المألوف"),
            ("Contextual outlier", "شاذ سياقي"),
            ("Collective outlier", "شاذ جماعي"),
            ("Leverage", "رافعة المشاهدة في الانحدار"),
            ("Cook's distance", "مسافة كوك لقياس التأثير"),
        ]
    )
