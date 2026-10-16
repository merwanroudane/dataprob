"""حذف القيم المفقودة: متى يكون صحيحًا ومتى يكون كارثة."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import charts, ui
from core.anim import play_animation
from core.datasets import missing_demo

ui.page_header(
    "الحذف: متى ولماذا",
    "الحذف هو أسهل حل وأكثره استعمالًا — وأكثره إساءة استعمال. هنا نرى "
    "**أنواع الحذف**، وتكلفته الخفيّة التي لا يراها المبتدئ: انهيار حجم "
    "العيّنة أسّيًا مع عدد الأعمدة.",
    icon=":material/delete_sweep:",
    badges=[":blue-badge[Listwise]", ":green-badge[Pairwise]", ":orange-badge[Column drop]", ":red-badge[Power loss]"],
)

t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. أنواع الحذف", divider="blue")
    ui.concept_grid(
        [
            (
                "Listwise deletion (Complete-case analysis)",
                "احذف الصف كاملًا إن كان فيه أي قيمة مفقودة.\n\n"
                "**مزايا:** بسيط، ومتّسق (كل التحليلات على نفس العيّنة).\n\n"
                "**عيوب:** يهدر بيانات هائلة؛ **متحيّز** إن لم يكن الفقد MCAR؛ "
                "ويقلّل القوّة الإحصائية.\n\n"
                "`df.dropna()`",
                "blue",
            ),
            (
                "Pairwise deletion (Available-case)",
                "استعمل كل زوج من الأعمدة بالصفوف المتاحة له فقط (شائع في "
                "حساب مصفوفة الارتباط).\n\n"
                "**عيوب خطيرة:** كل معامل محسوب على عيّنة مختلفة ⇒ مصفوفة ارتباط "
                "قد تكون **غير معرّفة موجبة** (non-positive-definite) وتكسر "
                "PCA والانحدار.\n\n"
                "`df.corr()` تفعل هذا افتراضيًا — انتبه.",
                "orange",
            ),
            (
                "Column (feature) deletion",
                "احذف العمود كلّه. مبرّر حين: الفقد ساحق (> 60-80٪) **و** العمود "
                "ليس حاسمًا، أو حين يكون العمود مكرّرًا مع عمود آخر مكتمل.\n\n"
                "**تحذير:** عمود بفقد 70٪ لكنه السبب الرئيسي للظاهرة أنفع من "
                "عشرة أعمدة مكتملة بلا معنى.",
                "violet",
            ),
            (
                "Threshold deletion",
                "احذف الصفوف/الأعمدة التي تتجاوز عتبة فقد معيّنة.\n\n"
                "`df.dropna(thresh=int(0.7 * df.shape[1]))` — أبقِ الصفوف التي "
                "فيها ٧٠٪ من الأعمدة على الأقل.\n\n"
                "عملي، لكن العتبة قرار تعسّفي يجب تبريره وتوثيقه.",
                "green",
            ),
        ],
        n_cols=2,
    )

    st.subheader("٢. التكلفة التي لا يراها أحد: انهيار العيّنة", divider="red")
    st.markdown(
        """
افترض **فقدًا مستقلًا بنسبة ٥٪ فقط** في كل عمود. احتمال أن يكون صف ما
مكتملًا في $k$ عمودًا هو:
"""
    )
    st.latex(r"P(\text{complete row}) = (1 - 0.05)^{k}")
    st.markdown(
        """
| عدد الأعمدة | الصفوف الباقية |
|---|---|
| 5 | 77٪ |
| 10 | 60٪ |
| 20 | **36٪** |
| 40 | **13٪** |
| 60 | **4.6٪** |

خسرت **ثلثي** بياناتك بسبب فقد ٥٪ فقط لكل عمود. وهذا في أفضل الحالات
(MCAR)؛ إن كان الفقد MAR فأنت لم تخسر حجمًا فقط، بل **غيّرت تركيبة عيّنتك**.
"""
    )

    st.subheader("٣. متى يكون الحذف هو القرار الصحيح فعلًا؟", divider="green")
    st.markdown(
        """
1. **الفقد في الهدف `y`** — احذف الصف. التعويض في الهدف يعني اختراع الإجابة
   التي تريد التنبّؤ بها، ويضخّم الأداء كذبًا.
2. **فقد ضئيل (< 5٪) + دلائل MCAR** — الخسارة في القوّة أقل من تعقيد التعويض.
3. **صف فارغ تقريبًا** — صف بـ 2 قيمة من 30 عمودًا لا يحمل معلومة.
4. **عمود شبه فارغ وغير جوهري** — لكن افحص أولًا: هل وجود القيمة نفسه مؤشّر؟
5. **سجلّ غير صالح أصلًا** — صف يمثّل اختبارًا داخليًا أو خطأ تسجيل.
"""
    )

    ui.danger(
        "الخطأ الأشيع: `df.dropna()` في أول سطر من الـ notebook، قبل حتى "
        "النظر إلى سبب الفقد. إن فعلتها على بيانات MAR أو MNAR، فكل نتيجة "
        "بعدها متحيّزة — ولن يظهر ذلك في أي مقياس أداء."
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. صيغ الحذف في pandas", divider="blue")
    ui.code(
        """
df.dropna()                                   # listwise: أي فقد ⇒ احذف الصف
df.dropna(how="all")                          # الصفوف الفارغة تمامًا فقط
df.dropna(subset=["income", "age"])           # فقد في أعمدة محدّدة فقط
df.dropna(thresh=int(0.7 * df.shape[1]))      # أبقِ الصفوف بـ ≥70% قيم
df.dropna(axis=1, thresh=int(0.5 * len(df)))  # احذف الأعمدة بفقد > 50%

# فقد في الهدف ⇒ حذف دائمًا
df = df[df["churn"].notna()]
"""
    )

    st.subheader("٢. قِس التكلفة قبل أن تحذف", divider="orange")
    ui.code(
        '''
before = len(df)
after = len(df.dropna())
print(f"الصفوف: {before:,} ← {after:,} ({after/before:.1%} باقية)")

# أي الأعمدة هي سبب الحذف؟
culprits = df.isna().sum().sort_values(ascending=False)
print(culprits[culprits > 0])

# كم صفًا سأربح لو حذفت أسوأ عمود بدل الصفوف؟
worst = culprits.index[0]
print("بدون هذا العمود:", len(df.drop(columns=[worst]).dropna()))
''',
        caption="غالبًا يتبيّن أن عمودًا واحدًا هو سبب فقدان ٦٠٪ من الصفوف — "
        "حذف العمود أرخص بكثير من حذف الصفوف.",
    )

    st.subheader("٣. قارن المجموعتين قبل الحذف (فحص التحيّز)", divider="red")
    ui.code(
        """
kept = df.dropna()
dropped = df[df.isna().any(axis=1)]

compare = pd.DataFrame({
    "kept_mean": kept.select_dtypes("number").mean(),
    "dropped_mean": dropped.select_dtypes("number").mean(),
})
compare["rel_diff_%"] = ((compare.dropped_mean - compare.kept_mean)
                         / compare.kept_mean * 100).round(1)
print(compare)      # فروق كبيرة ⇒ الحذف سيغيّر تركيبة عيّنتك
"""
    )

    st.subheader("٤. الفرق بين `corr()` الافتراضي والحذف الكامل", divider="violet")
    ui.code(
        """
df.corr(numeric_only=True)              # pairwise — كل زوج بعيّنته
df.dropna().corr(numeric_only=True)     # listwise — عيّنة واحدة متّسقة

# الفرق بينهما يكشف حجم أثر الفقد على بنية الارتباط
"""
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("انهيار العيّنة مع عدد الأعمدة", divider="violet")
    st.markdown(
        "اختر نسبة الفقد لكل عمود، ثم شغّل الأنيميشن لترى كم صفًا يبقى بعد "
        "`dropna()` كلّما أضفت عمودًا."
    )

    rate = st.select_slider(
        "نسبة الفقد في كل عمود (مستقلّة)",
        options=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
        value=0.05,
        format_func=lambda v: f"{v:.0%}",
        key="del_rate",
    )

    MAXK = 40
    ks = np.arange(1, MAXK + 1)
    survive = (1 - rate) ** ks
    curve = pd.DataFrame({"k_columns": ks, "pct_complete": survive * 100})

    def _render_collapse(frame: int) -> None:
        k = int(ks[frame])
        pct = float(survive[frame] * 100)
        n0 = 10_000
        c1, c2, c3 = st.columns(3)
        c1.metric("عدد الأعمدة", k)
        c2.metric("نسبة الصفوف الباقية", f"{pct:.1f}%")
        c3.metric(
            "من 10,000 صف",
            f"{int(n0 * pct / 100):,}",
            delta=f"-{int(n0 - n0 * pct / 100):,}",
            delta_color="inverse",
        )

        sofar = curve.iloc[: frame + 1]
        line = (
            alt.Chart(sofar)
            .mark_area(opacity=0.35, line={"strokeWidth": 3})
            .encode(
                x=alt.X("k_columns:Q", title="number of columns", scale=alt.Scale(domain=[1, MAXK])),
                y=alt.Y("pct_complete:Q", title="% complete rows", scale=alt.Scale(domain=[0, 100])),
                tooltip=["k_columns", "pct_complete"],
            )
            .properties(height=320)
        )
        st.altair_chart(line + charts.hrule(50, color="#E8590C"), width="stretch")

        if pct < 50:
            st.markdown(
                f":red-badge[تحذير] بـ {k} عمودًا ونسبة فقد {rate:.0%} لكل عمود، "
                f"فقدت أكثر من نصف بياناتك — وكل ذلك قبل أن تدرّب نموذجًا واحدًا."
            )
        else:
            st.markdown(f":green-badge[مقبول حتى الآن] ما زال {pct:.0f}% من الصفوف مكتملًا.")

    play_animation("delete_collapse", MAXK, _render_collapse, fps=3.0, label="عدد الأعمدة")

    st.subheader("وهل الحذف يغيّر النتائج؟ جرّب", divider="orange")
    mech = st.segmented_control(
        "آلية الفقد", ["MCAR", "MAR", "MNAR"], default="MNAR", key="del_mech"
    ) or "MNAR"
    d = missing_demo(mech, 0.35)
    full_mean = float(d["income_true"].mean())
    cc_mean = float(d.dropna(subset=["income"])["income"].mean())
    cc_age = float(d.dropna(subset=["income"])["age"].mean())
    all_age = float(d["age"].mean())

    m1, m2, m3 = st.columns(3)
    m1.metric("متوسط الدخل الحقيقي", f"{full_mean:,.0f}")
    m2.metric(
        "بعد الحذف (complete-case)",
        f"{cc_mean:,.0f}",
        delta=f"{(cc_mean - full_mean) / full_mean:+.1%}",
        delta_color="inverse",
    )
    m3.metric(
        "متوسط العمر بعد الحذف",
        f"{cc_age:.1f}",
        delta=f"{cc_age - all_age:+.1f} سنة",
        delta_color="off",
        help="تغيّر متوسط العمر دليل على أن تركيبة العيّنة نفسها تغيّرت",
    )
    st.caption(
        "في MCAR تتوقّع انحرافًا ضئيلًا عشوائيًا. في MAR/MNAR سترى انزياحًا "
        "منهجيًا — وهذا هو التحيّز الذي لا يظهر في أي مقياس أداء."
    )

# ============================================================== القرار
with t_decide:
    st.subheader("شجرة قرار الحذف", divider="orange")
    ui.decision_table(
        [
            ("الفقد في الهدف `y`", ":green-badge[احذف الصف]", "التعويض في الهدف تسريب واختراع للإجابة"),
            ("فقد < 5٪ ودلائل MCAR", ":green-badge[حذف مقبول]", "خسارة قوّة طفيفة، لا تحيّز"),
            ("فقد 5–30٪ وMAR", ":red-badge[لا تحذف — عوّض]", "الحذف يغيّر تركيبة العيّنة ويُحدث تحيّزًا"),
            ("فقد > 60٪ في عمود غير جوهري", ":orange-badge[احذف العمود]", "المعلومة الباقية ضئيلة والضجيج عالٍ"),
            ("فقد > 60٪ لكن العمود جوهري", ":violet-badge[أبقه + مؤشّر فقد]", "حتى الفقد نفسه معلومة؛ والنماذج الشجرية تستفيد"),
            ("صف فارغ بنسبة > 70٪", ":green-badge[احذف الصف]", "لا يحمل معلومة كافية"),
            ("حجم العيّنة صغير أصلًا", ":red-badge[تجنّب الحذف]", "القوّة الإحصائية ستنهار"),
            ("ستحسب مصفوفة ارتباط", ":orange-badge[قرّر listwise صراحةً]", "pairwise قد تنتج مصفوفة غير صالحة لـ PCA/الانحدار"),
            ("بيانات الاختبار (test set)", ":red-badge[لا تحذف صفوفًا]", "الاختبار يجب أن يمثّل الواقع؛ عوّض بنفس معاملات التدريب"),
        ]
    )

    ui.tip(
        "قاعدة عملية: **احذف الصفوف لأسباب منطقية (سجلّ غير صالح، هدف مفقود)، "
        "ولا تحذفها لأسباب إحصائية (راحة في الحساب).**"
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("pandas — DataFrame.dropna", "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html"),
            ("pandas — Working with missing data", "https://pandas.pydata.org/docs/user_guide/missing_data.html"),
            ("van Buuren — FIMD §1.3: Ad-hoc solutions", "https://stefvanbuuren.name/fimd/sec-simplesolutions.html"),
            ("Little & Rubin — Statistical Analysis with Missing Data", "https://doi.org/10.1002/9781119482260"),
            ("scikit-learn — Imputation of missing values", "https://scikit-learn.org/stable/modules/impute.html"),
        ]
    )
    ui.terms(
        [
            ("Listwise deletion", "حذف الصف كاملًا"),
            ("Complete-case analysis", "التحليل على الصفوف المكتملة فقط"),
            ("Pairwise deletion", "استعمال الصفوف المتاحة لكل زوج"),
            ("Statistical power", "القوّة الإحصائية (القدرة على كشف أثر حقيقي)"),
            ("Positive-definite matrix", "مصفوفة معرّفة موجبة (شرط لعمل PCA/الانحدار)"),
        ]
    )
