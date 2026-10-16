"""أنواع القيم المفقودة: MCAR / MAR / MNAR."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import charts, ui
from core.anim import play_animation
from core.datasets import missing_demo

ui.page_header(
    "أنواع القيم المفقودة",
    "الفقد ليس نوعًا واحدًا. **آلية الفقد (missingness mechanism)** هي التي "
    "تحدّد إن كان الحذف آمنًا، وإن كان التعويض بالمتوسط كارثة، وإن كانت "
    "نتائجك متحيّزة أصلًا. هذا التصنيف يعود إلى *Rubin (1976)* وهو أساس كل "
    "أدبيات التعامل مع الفقد.",
    icon=":material/help_center:",
    badges=[":blue-badge[MCAR]", ":orange-badge[MAR]", ":red-badge[MNAR]", ":violet-badge[Rubin 1976]"],
)

t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. التعريف الرياضي في سطر واحد", divider="blue")
    st.markdown(
        "لتكن $Y$ البيانات، $Y_{obs}$ المرصودة، $Y_{mis}$ المفقودة، و $R$ "
        "مؤشّر الفقد (1 = مفقود). نسأل: **على ماذا يعتمد احتمال الفقد؟**"
    )
    st.latex(r"\textbf{MCAR:}\quad P(R \mid Y_{obs}, Y_{mis}) = P(R)")
    st.latex(r"\textbf{MAR:}\quad P(R \mid Y_{obs}, Y_{mis}) = P(R \mid Y_{obs})")
    st.latex(r"	extbf{MNAR:}\quad P(R \mid Y_{obs}, Y_{mis}) 
eq P(R \mid Y_{obs})")
    st.caption(
        "أي أن احتمال الفقد يبقى معتمدًا على القيمة المفقودة نفسها حتى بعد "
        "التحكّم في كل ما هو مرصود."
    )

    st.subheader("٢. الأنواع الثلاثة بالأمثلة", divider="green")
    ui.concept_grid(
        [
            (
                "MCAR — Missing Completely At Random",
                "**التعريف:** احتمال الفقد لا يعتمد على أي شيء، لا على المتغيّر "
                "نفسه ولا على غيره — صدفة بحتة.\n\n"
                "**أمثلة واقعية:** أنبوب عيّنة انكسر في المخبر؛ انقطاع كهرباء "
                "أثناء تسجيل قراءة حسّاس؛ استبيانات ضاعت في البريد.\n\n"
                "**الأثر:** الحذف لا يُحدث تحيّزًا — فقط يُضعف القوّة الإحصائية "
                "(statistical power). هذه هي الحالة **الوحيدة** المريحة.\n\n"
                ":blue-badge[نادر جدًا في الواقع]",
                "blue",
            ),
            (
                "MAR — Missing At Random",
                "**التعريف:** احتمال الفقد يعتمد على متغيّرات **مرصودة** أخرى، "
                "وليس على القيمة المفقودة نفسها بعد التحكّم فيها.\n\n"
                "**أمثلة:** الشباب يرفضون كشف الدخل أكثر من كبار السن (الفقد "
                "يعتمد على `age` المرصود)؛ جهاز قياس يعطّل في درجات حرارة "
                "منخفضة (`temperature` مرصود).\n\n"
                "**الأثر:** الحذف **يُحدث تحيّزًا**، لكن التعويض الصحيح "
                "(Multiple Imputation / Iterative) يستعيد التقديرات غير "
                "المتحيّزة — شرط إدخال المتغيّرات المفسِّرة للفقد.\n\n"
                ":orange-badge[الافتراض العملي الأكثر استعمالًا]",
                "orange",
            ),
            (
                "MNAR — Missing Not At Random",
                "**التعريف:** احتمال الفقد يعتمد على **القيمة المفقودة نفسها**.\n\n"
                "**أمثلة:** أصحاب الدخول المرتفعة جدًا يرفضون التصريح بدخلهم؛ "
                "المرضى الأشدّ مرضًا ينسحبون من التجربة السريرية؛ حسّاس يتشبّع "
                "فوق حدّ معيّن فلا يسجّل القيم العالية (censoring).\n\n"
                "**الأثر:** **لا توجد معالجة إحصائية بحتة تحلّها.** كل الطرق "
                "تبقى متحيّزة. الحل: نمذجة آلية الفقد صراحةً (selection models، "
                "pattern-mixture models) + تحليل حساسية (sensitivity analysis).\n\n"
                ":red-badge[الأخطر والأكثر شيوعًا في البيانات البشرية]",
                "red",
            ),
            (
                "نوع رابع عملي: Structurally missing",
                "**التعريف:** القيمة غير موجودة **منطقيًا**، لا مفقودة: "
                "`تاريخ_الطلاق` لشخص غير متزوّج، `عدد_الأطفال` لحساب شركة، "
                "`آخر_شراء` لعميل جديد.\n\n"
                "**الأثر:** تعويضها **خطأ منهجي** — أنت تخترع واقعًا. "
                "الصواب: فئة صريحة (`Not applicable`) أو عمود مؤشّر، أو فصل "
                "النمذجة إلى مجموعتين.",
                "violet",
            ),
        ],
        n_cols=2,
    )

    st.subheader("٣. لماذا هذا التصنيف هو أهم شيء في الدرس؟", divider="red")
    st.markdown(
        """
| الآلية | الحذف (listwise) | التعويض بالمتوسط | التعويض المتعدّد (MI) |
|---|---|---|---|
| **MCAR** | :green[غير متحيّز] (لكن يفقد قوّة) | :orange[يقلّل التباين بشدّة] | :green[صحيح] |
| **MAR** | :red[متحيّز] | :red[متحيّز ويقلّل التباين] | :green[صحيح إن أُدرجت المتغيّرات المفسِّرة] |
| **MNAR** | :red[متحيّز] | :red[متحيّز] | :orange[يبقى متحيّزًا — يحتاج نمذجة صريحة] |

الخلاصة العملية: **السؤال ليس «بماذا أعوّض؟» بل «لماذا فُقدت؟»**
"""
    )

    ui.beginner_note(
        """
- **MCAR**: الفقد صدفة، كأن ورقة طارت من الملف.
- **MAR**: الفقد له سبب، والسبب **مكتوب في جدولك** (مثلًا العمر).
- **MNAR**: الفقد له سبب، والسبب هو **القيمة الغائبة نفسها** — ولا تستطيع
  رؤيتها، لذلك لا تستطيع تصحيحها بالكامل.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. محاكاة الآليات الثلاث (لفهمها عمليًا)", divider="blue")
    ui.code(
        """
import numpy as np, pandas as pd
rng = np.random.default_rng(7)

# MCAR: احتمال ثابت للجميع
mask = rng.random(len(df)) < 0.25
df.loc[mask, "income"] = np.nan

# MAR: الاحتمال يعتمد على عمود مرصود (العمر)
z = (df["age"] - df["age"].mean()) / df["age"].std()
p = 1 / (1 + np.exp(-(2.2 * z - 0.6)))          # الشباب أكثر عرضة للفقد
df.loc[rng.random(len(df)) < p, "income"] = np.nan

# MNAR: الاحتمال يعتمد على قيمة الدخل نفسها
z = (df["income"] - df["income"].mean()) / df["income"].std()
p = 1 / (1 + np.exp(-(2.6 * z - 0.8)))          # الدخول العالية تختفي
df.loc[rng.random(len(df)) < p, "income"] = np.nan
"""
    )

    st.subheader("٢. أول ما تفعله: مؤشّر الفقد", divider="green")
    ui.code(
        """
# عمود يحفظ «واقعة» الفقد نفسها — غالبًا متغيّر تنبؤي قوي
df["income_was_missing"] = df["income"].isna().astype("int8")

# في scikit-learn: مؤشّر آلي لكل عمود فيه فقد
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="median", add_indicator=True)
""",
        caption="في MNAR خصوصًا، المؤشّر يلتقط جزءًا من المعلومة التي ضاعت.",
    )

    st.subheader("٣. كشف رموز الفقد المخفيّة", divider="orange")
    ui.code(
        """
SENTINELS = [-99, -999, -1, 9999, "?", "N/A", "NA", "", " ", "unknown", "none"]

for c in df.columns:
    hits = df[c].isin(SENTINELS).sum()
    if hits:
        print(f"{c}: {hits} قيمة مشبوهة")

df = df.replace(SENTINELS, pd.NA)       # وحّدها قبل أي حساب
""",
        caption="قيمة `-99` تدخل في المتوسط وتجرّه إلى الأسفل بصمت. هذا من أكثر "
        "الأخطاء شيوعًا في البيانات المسحية.",
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("شاهد التحيّز يكبر مع نسبة الفقد", divider="violet")
    st.markdown(
        "نزرع فقدًا في عمود `income` بآلية تختارها، ثم نقارن **المتوسط الحقيقي** "
        "(نعرفه لأن البيانات مُصنّعة) بالمتوسط المحسوب من القيم المرصودة فقط."
    )

    mech = st.segmented_control(
        "آلية الفقد",
        ["MCAR", "MAR", "MNAR"],
        default="MNAR",
        key="mt_mech",
        help="غيّر الآلية ثم شغّل الأنيميشن لترى الفرق",
    ) or "MNAR"

    RATES = [0.05, 0.12, 0.2, 0.3, 0.4, 0.5, 0.6]

    @st.cache_data(ttl=1800, max_entries=64)
    def _bias_curve(mechanism: str) -> pd.DataFrame:
        rows = []
        for r in RATES:
            d = missing_demo(mechanism, r)
            true_mean = float(d["income_true"].mean())
            obs_mean = float(d["income"].mean())
            rows.append(
                {
                    "rate": round(r * 100),
                    "true_mean": true_mean,
                    "observed_mean": obs_mean,
                    "bias_pct": (obs_mean - true_mean) / true_mean * 100,
                    "n_missing": int(d["income"].isna().sum()),
                }
            )
        return pd.DataFrame(rows)

    curve = _bias_curve(mech)

    def _render_bias(frame: int) -> None:
        row = curve.iloc[frame]
        d = missing_demo(mech, RATES[frame])
        d = d.assign(status=np.where(d["income"].isna(), "missing", "observed"))

        c1, c2, c3 = st.columns(3)
        c1.metric("نسبة الفقد", f"{row['rate']:.0f}%", help=f"{int(row['n_missing'])} قيمة")
        c2.metric("المتوسط الحقيقي", f"{row['true_mean']:,.0f}")
        c3.metric(
            "المتوسط المرصود",
            f"{row['observed_mean']:,.0f}",
            delta=f"{row['bias_pct']:+.1f}%",
            delta_color="inverse",
        )

        left, right = st.columns(2)
        with left:
            st.markdown("###### أين تقع القيم المفقودة؟")
            sc = charts.scatter(
                d.assign(income_plot=d["income_true"]),
                "age",
                "income_plot",
                color="status",
                tooltip=["age", "income_plot", "status"],
                height=300,
            )
            st.altair_chart(sc, width="stretch")
        with right:
            st.markdown("###### تحيّز المتوسط مع ارتفاع نسبة الفقد")
            sofar = curve.iloc[: frame + 1]
            line = (
                alt.Chart(sofar)
                .mark_line(point=True, strokeWidth=3)
                .encode(
                    x=alt.X("rate:Q", title="missing rate (%)", scale=alt.Scale(domain=[0, 65])),
                    y=alt.Y("bias_pct:Q", title="bias in mean (%)", scale=alt.Scale(domain=[-30, 30])),
                    tooltip=["rate", "bias_pct"],
                )
                .properties(height=300)
            )
            zero = charts.hrule(0, color="#5C6B84", dash=[4, 4])
            st.altair_chart(line + zero, width="stretch")

        notes = {
            "MCAR": ":blue-badge[MCAR] القيم المفقودة موزّعة عشوائيًا على كامل المدى؛ "
            "المتوسط المرصود يبقى قريبًا من الحقيقي مهما ارتفعت النسبة — فقط يصبح أقل دقّة.",
            "MAR": ":orange-badge[MAR] الفقد يتركّز عند الأعمار الصغيرة؛ وبما أن العمر "
            "مرتبط بالدخل، ينزاح المتوسط. لكن العمر **مرصود**، لذا يمكن تصحيح التحيّز بالنمذجة.",
            "MNAR": ":red-badge[MNAR] الدخول العالية هي التي تختفي؛ المتوسط المرصود يُقلّل "
            "تقدير الحقيقة بشكل منهجي، ولا يوجد عمود في الجدول يفسّر الفقد ⇒ لا تصحيح كاملًا.",
        }
        st.markdown(notes[mech])

    play_animation(
        f"missing_bias_{mech}",
        len(RATES),
        _render_bias,
        fps=0.7,
        label="نسبة الفقد",
    )

# ============================================================== القرار
with t_decide:
    st.subheader("الآلية ⇐ ما هو مسموح؟", divider="orange")
    ui.decision_table(
        [
            ("MCAR، فقد < 5٪", "الحذف (listwise) مقبول", "لا تحيّز، والخسارة في القوّة ضئيلة"),
            ("MCAR، فقد مرتفع", "تعويض (median/iterative)", "الحذف يهدر بيانات كثيرة بلا داعٍ"),
            ("MAR", "تعويض متعدّد أو Iterative + إدراج المتغيّرات المفسِّرة للفقد", "هي الطريقة الوحيدة لاستعادة تقدير غير متحيّز"),
            ("MAR مع نموذج شجري", "التعويض + مؤشّر فقد", "الأشجار تستغل المؤشّر لتقسيم مفيد"),
            ("MNAR", "نمذجة صريحة + تحليل حساسية + مؤشّر فقد", "لا طريقة آلية تكفي؛ أبلغ عن الافتراض في ورقتك"),
            ("Structurally missing", "فئة `Not applicable` أو نموذجان منفصلان", "التعويض هنا اختراع لبيانات غير موجودة"),
            ("لا تعرف الآلية", "افترض MAR وطبّق MI، وصرّح بالافتراض", "الممارسة المعيارية في الأدبيات"),
        ],
        headers=("الآلية / الحالة", "الإجراء", "السبب"),
    )

    ui.danger(
        "**لا يمكن إثبات MCAR أو MAR أو MNAR من البيانات وحدها.** اختبار "
        "Little يختبر MCAR فقط (وبقدرة محدودة)، والتمييز بين MAR و MNAR "
        "**غير قابل للاختبار إحصائيًا** لأنه يتطلّب معرفة القيم المفقودة. "
        "القرار يُبنى على فهم كيفية جمع البيانات — اسأل من جمعها."
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("Rubin (1976) — Inference and Missing Data", "https://doi.org/10.1093/biomet/63.3.581"),
            ("Little & Rubin — Statistical Analysis with Missing Data (3rd ed.)", "https://doi.org/10.1002/9781119482260"),
            ("van Buuren — Flexible Imputation of Missing Data (كتاب مجاني)", "https://stefvanbuuren.name/fimd/"),
            ("scikit-learn — Imputation of missing values", "https://scikit-learn.org/stable/modules/impute.html"),
            ("pandas — Working with missing data", "https://pandas.pydata.org/docs/user_guide/missing_data.html"),
            ("statsmodels — MICE", "https://www.statsmodels.org/stable/imputation.html"),
        ]
    )
    ui.terms(
        [
            ("MCAR", "مفقود عشوائيًا تمامًا"),
            ("MAR", "مفقود عشوائيًا مشروطًا بمتغيّرات مرصودة"),
            ("MNAR / NMAR", "مفقود غير عشوائي (يعتمد على القيمة نفسها)"),
            ("Missingness mechanism", "آلية الفقد"),
            ("Listwise deletion", "حذف الصف كاملًا عند وجود أي فقد"),
            ("Multiple Imputation (MI)", "التعويض المتعدّد"),
            ("Missing indicator", "عمود ثنائي يسجّل واقعة الفقد"),
            ("Sensitivity analysis", "تحليل حساسية النتائج لتغيّر الافتراضات"),
        ]
    )
