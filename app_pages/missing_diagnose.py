"""تشخيص الفقد: الأنماط والاختبارات."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from core import charts, ui
from core.anim import play_animation
from core.datasets import clean_customers, missing_demo

ui.page_header(
    "تشخيص الفقد واختباراته",
    "كيف تنتقل من «يوجد فقد» إلى «أعرف لماذا فُقد»؟ ثلاث أدوات: "
    "**خريطة الأنماط**، **ارتباط الفقد بين الأعمدة**، و**مقارنة المجموعتين** "
    "(من فُقدت قيمته مقابل من لم تُفقد).",
    icon=":material/stethoscope:",
    badges=[":blue-badge[Patterns]", ":green-badge[Nullity correlation]", ":orange-badge[Group tests]", ":red-badge[Little's MCAR]"],
)

t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()


# ---------------------------------------------------------------- أدوات
@st.cache_data(ttl=1800, max_entries=16)
def pattern_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """أنماط الفقد: كل تركيبة (مفقود/مرصود) وعدد الصفوف الموافقة."""
    miss = df[cols].isna()
    pat = miss.astype(int).astype(str).agg("".join, axis=1)
    out = (
        pd.DataFrame({"pattern": pat})
        .value_counts()
        .reset_index(name="n_rows")
        .sort_values("n_rows", ascending=False)
    )
    out["n_missing_cols"] = out["pattern"].str.count("1")
    out["pct"] = (out["n_rows"] / len(df) * 100).round(1)
    out["readable"] = out["pattern"].map(
        lambda p: " + ".join(c for c, f in zip(cols, p) if f == "1") or "لا فقد"
    )
    return out.reset_index(drop=True)


@st.cache_data(ttl=1800, max_entries=16)
def group_tests(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """مقارنة بقيّة الأعمدة بين مجموعتي «مفقود» و«مرصود» في `target_col`."""
    is_miss = df[target_col].isna()
    rows = []
    for c in df.columns:
        if c == target_col or c.endswith("_true"):
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            a = s[is_miss].dropna()
            b = s[~is_miss].dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            t, p = stats.ttest_ind(a, b, equal_var=False)
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or 1e-9
            rows.append(
                {
                    "variable": c,
                    "test": "Welch t-test",
                    "mean_if_missing": round(float(a.mean()), 2),
                    "mean_if_observed": round(float(b.mean()), 2),
                    "effect_size_d": round(float((a.mean() - b.mean()) / pooled), 3),
                    "p_value": float(p),
                }
            )
        else:
            tab = pd.crosstab(s.fillna("__NA__"), is_miss)
            if tab.shape[0] < 2 or tab.shape[1] < 2 or tab.to_numpy().min() < 1:
                continue
            chi2, p, _, _ = stats.chi2_contingency(tab)
            rows.append(
                {
                    "variable": c,
                    "test": "Chi-square",
                    "mean_if_missing": None,
                    "mean_if_observed": None,
                    "effect_size_d": round(float(np.sqrt(chi2 / tab.to_numpy().sum())), 3),
                    "p_value": float(p),
                }
            )
    out = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    if len(out):
        out["significant"] = np.where(out["p_value"] < 0.05, "نعم ⇒ دليل ضدّ MCAR", "لا")
        out["p_value"] = out["p_value"].map(lambda v: f"{v:.2e}" if v < 1e-3 else f"{v:.4f}")
    return out


# ============================================================== النظرية
with t_theory:
    st.subheader("١. ثلاث أدوات تشخيص، بهذا الترتيب", divider="blue")
    ui.concept_grid(
        [
            (
                "خريطة الأنماط — Missingness patterns",
                "لا تنظر إلى كل عمود وحده، بل إلى **تركيبات** الفقد: هل الأعمدة "
                "الثلاثة تُفقد معًا دائمًا؟ هذا يكشف سببًا بنيويًا (قسم من "
                "الاستبيان لم يُملأ، جدول لم يُدمج، حسّاس تعطّل).",
                "blue",
            ),
            (
                "ارتباط الفقد — Nullity correlation",
                "ارتباط بين **مؤشّرات** الفقد لا بين القيم. ارتباط ≈ +1 يعني "
                "«يُفقدان معًا»، و ≈ -1 يعني «أحدهما بديل عن الآخر» "
                "(مثلاً: `phone` أو `email`، أحدهما يُملأ).",
                "green",
            ),
            (
                "مقارنة المجموعتين — Group comparison",
                "قسّم الصفوف إلى «فُقدت قيمة X» و«لم تُفقد»، ثم قارن بقيّة "
                "الأعمدة بين المجموعتين (t-test / Mann-Whitney / Chi²). "
                "فرق معنوي ⇒ الفقد **ليس** MCAR، ويعتمد على متغيّر مرصود "
                "⇒ دليل لصالح MAR.",
                "orange",
            ),
            (
                "اختبار Little للـ MCAR",
                "اختبار شامل (χ²) لفرضية «الفقد MCAR». رفض الفرضية ⇒ ليس MCAR. "
                "**عدم** الرفض **لا يثبت** MCAR (قد تكون القوّة ضعيفة). "
                "متوفّر في R عبر `naniar::mcar_test`؛ في بايثون يُستعاض عنه "
                "غالبًا بمجموعة اختبارات المجموعتين أعلاه.",
                "red",
            ),
        ],
        n_cols=2,
    )

    st.subheader("٢. حدود التشخيص — صريحة", divider="red")
    st.markdown(
        """
- تستطيع البيانات أن تقول لك: **«هذا ليس MCAR»**.
- لا تستطيع أن تقول لك: **«هذا MAR وليس MNAR»** — لأن التمييز بينهما يتطلّب
  معرفة القيم المفقودة نفسها، وهي غائبة بالتعريف.
- لذلك: التشخيص الإحصائي **يضيّق** الاحتمالات، والمعرفة بالمجال
  (domain knowledge) هي التي تحسم. اسأل: *من جمع هذه البيانات؟ كيف؟ متى
  يُترك الحقل فارغًا؟*
"""
    )

    ui.beginner_note(
        """
تخيّل أن ٣٠٪ من الناس لم يكتبوا دخلهم. اسأل نفسك: **هل الذين لم يكتبوا
يختلفون عن الذين كتبوا؟** إن كانوا أصغر سنًا، أو من مدينة معيّنة، أو أقل
رضًا — فالفقد له سبب، وليس صدفة. هذا بالضبط ما تقيسه «مقارنة المجموعتين».
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. الصورة السريعة", divider="blue")
    ui.code(
        """
df.isna().sum().sort_values(ascending=False)
df.isna().mean().mul(100).round(1)
df.isna().sum(axis=1).value_counts().sort_index()   # كم عمودًا مفقودًا لكل صف؟
"""
    )

    st.subheader("٢. أنماط الفقد", divider="green")
    ui.code(
        """
cols = ["income", "satisfaction", "spending"]
pattern = (df[cols].isna().astype(int).astype(str).agg("".join, axis=1)
           .value_counts().rename_axis("pattern").reset_index(name="n_rows"))
print(pattern)        # 000 = لا فقد، 101 = الأول والثالث مفقودان
"""
    )

    st.subheader("٣. ارتباط الفقد بين الأعمدة", divider="orange")
    ui.code(
        """
nullity = df.isna().astype(int)
nullity = nullity.loc[:, nullity.sum() > 0]      # الأعمدة التي فيها فقد فقط
print(nullity.corr().round(2))

# مكتبة مخصّصة للتصوير (اختيارية)
# pip install missingno
import missingno as msno
msno.matrix(df); msno.heatmap(df); msno.dendrogram(df)
"""
    )

    st.subheader("٤. اختبار الفرق بين المجموعتين", divider="violet")
    ui.code(
        '''
from scipy import stats

is_missing = df["income"].isna()

for c in df.select_dtypes("number").columns.drop("income"):
    a, b = df.loc[is_missing, c].dropna(), df.loc[~is_missing, c].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)       # Welch
    if p < 0.05:
        print(f"{c}: فرق معنوي (p={p:.4g}) ⇒ الفقد مرتبط بـ {c} ⇒ ليس MCAR")

for c in df.select_dtypes(exclude="number").columns:
    tab = pd.crosstab(df[c], is_missing)
    chi2, p, dof, exp = stats.chi2_contingency(tab)
    if p < 0.05:
        print(f"{c}: ارتباط فئوي معنوي (p={p:.4g})")
''',
        caption="عند تعدّد الاختبارات، صحّح مستوى المعنوية (Bonferroni أو FDR) "
        "حتى لا تتضخّم الأخطاء من النوع الأول.",
    )

    st.subheader("٥. على بياناتنا — جرّب الآليات الثلاث", divider="blue")
    mech = st.segmented_control(
        "آلية الفقد المزروعة في `income`",
        ["MCAR", "MAR", "MNAR"],
        default="MAR",
        key="md_mech",
    ) or "MAR"
    d = missing_demo(mech, 0.3).drop(columns=["income_true"])
    res = group_tests(d, "income")
    st.dataframe(res, width="stretch", hide_index=True)
    st.caption(
        "في **MCAR** تتوقّع ألّا يكون أي فرق معنويًا. في **MAR** سترى `age` "
        "(المتغيّر الذي بُنيت عليه آلية الفقد) معنويًا بوضوح. في **MNAR** قد "
        "تظهر فروق في المتغيّرات المرتبطة بالدخل (مثل `spending`) لأنها "
        "**انعكاس** للقيمة المفقودة، لا سببها."
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("بناء خريطة الفقد أمامك", divider="violet")
    st.markdown(
        "نعرض مصفوفة الفقد (nullity matrix): كل صف = مُشاهدة، كل عمود = متغيّر. "
        "الأزرق = قيمة موجودة، الأحمر = مفقودة. الأنيميشن يضيف الأعمدة تدريجيًا."
    )

    base = clean_customers(220).copy()
    rng = np.random.default_rng(3)
    base["income"] = np.where(rng.random(len(base)) < 0.18, np.nan, base["income"])
    z = (base["age"] - base["age"].mean()) / base["age"].std()
    p_sat = 1 / (1 + np.exp(-(2.5 * z - 1.0)))
    base["satisfaction"] = np.where(rng.random(len(base)) < p_sat, np.nan, base["satisfaction"])
    base["spending"] = np.where(base["income"].isna() & (rng.random(len(base)) < 0.75), np.nan, base["spending"])
    base["education"] = base["education"].where(rng.random(len(base)) > 0.06)

    SHOW = ["age", "income", "satisfaction", "spending", "education"]
    NOTES = [
        "`age` — لا فقد إطلاقًا: عمود مرجعي سنقارن به.",
        "`income` — فقد موزّع عشوائيًا على كل الصفوف (يشبه MCAR).",
        "`satisfaction` — لاحظ تجمّع الفقد في منطقة معيّنة: مرتبط بـ `age` ⇒ MAR.",
        "`spending` — يُفقد غالبًا **مع** `income`: نمط مشترك ⇒ سبب بنيوي واحد.",
        "`education` — فقد خفيف ومتناثر؛ فئوي، يُعالج بفئة `Missing` صريحة.",
    ]

    def _render_matrix(frame: int) -> None:
        cols = SHOW[: frame + 1]
        m = base[cols].isna()
        long = (
            m.reset_index(names="row")
            .melt(id_vars="row", var_name="variable", value_name="is_missing")
        )
        long["state"] = np.where(long["is_missing"], "missing", "observed")
        chart = (
            alt.Chart(long)
            .mark_rect()
            .encode(
                x=alt.X("variable:N", title=None, sort=cols),
                y=alt.Y("row:O", title="rows", axis=None),
                color=alt.Color(
                    "state:N",
                    scale=alt.Scale(
                        domain=["observed", "missing"], range=["#4C6EF5", "#E8590C"]
                    ),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=["variable", "state"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, width="stretch")

        st.markdown(f"**:blue-badge[{SHOW[frame]}]** {NOTES[frame]}")
        if frame >= 1:
            nullity = base[cols].isna().astype(int)
            nullity = nullity.loc[:, nullity.sum() > 0]
            if nullity.shape[1] >= 2:
                corr = nullity.corr().round(2).reset_index(names="a").melt(
                    id_vars="a", var_name="b", value_name="corr"
                )
                st.markdown("###### ارتباط الفقد (nullity correlation)")
                st.altair_chart(charts.heat(corr, "a", "b", "corr", height=220), width="stretch")

    play_animation(
        "missing_matrix",
        len(SHOW),
        _render_matrix,
        fps=0.6,
        label="المتغيّر",
    )

# ============================================================== القرار
with t_decide:
    st.subheader("ماذا تفعل بنتيجة التشخيص؟", divider="orange")
    ui.decision_table(
        [
            ("لا فروق معنوية بين المجموعتين", "افترض MCAR مبدئيًا", "لكن تذكّر أن عدم الرفض ليس إثباتًا"),
            ("فرق معنوي في متغيّر مرصود", "MAR على الأرجح", "أدرج هذا المتغيّر في نموذج التعويض — هذا شرط صحّة MI"),
            ("نمط فقد كتلي (أعمدة تُفقد معًا)", "ابحث عن سبب بنيوي", "قسم استبيان، جدول غير مدموج، حسّاس معطّل"),
            ("الفقد مرتبط بالزمن", "افحص تغيّر طريقة الجمع", "قد يكون العمود أُضيف لاحقًا ⇒ فقد بنيوي لا إحصائي"),
            ("الفقد مرتبط بالهدف نفسه", "تحذير شديد", "تعويضه بنموذج يستعمل الهدف = تسريب؛ والمؤشّر يصبح ضروريًا"),
            ("فقد في الهدف `y`", "احذف تلك الصفوف", "التعويض في الهدف يخترع إجابات ويضخّم الأداء كذبًا"),
            ("لا تستطيع الحسم", "افترض MAR + حلّل الحساسية", "وصرّح بالافتراض في منهجية البحث"),
        ]
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("van Buuren — FIMD, Ch. 2: Multiple imputation", "https://stefvanbuuren.name/fimd/sec-MCAR.html"),
            ("Little (1988) — A Test of MCAR for Multivariate Data", "https://doi.org/10.1080/01621459.1988.10478722"),
            ("naniar (R) — mcar_test", "https://naniar.njtierney.com/reference/mcar_test.html"),
            ("missingno — تصوير أنماط الفقد في Python", "https://github.com/ResidentMario/missingno"),
            ("SciPy — statistical tests", "https://docs.scipy.org/doc/scipy/reference/stats.html"),
            ("pandas — DataFrame.isna", "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html"),
        ]
    )
    ui.terms(
        [
            ("Nullity matrix", "مصفوفة تُظهر مواضع الفقد"),
            ("Nullity correlation", "ارتباط بين مؤشّرات الفقد"),
            ("Missingness pattern", "تركيبة الأعمدة المفقودة في الصف"),
            ("Welch t-test", "اختبار t بدون افتراض تساوي التباينات"),
            ("Chi-square test", "اختبار استقلال بين متغيّرين فئويين"),
            ("Multiple testing correction", "تصحيح تعدّد الاختبارات"),
        ]
    )
