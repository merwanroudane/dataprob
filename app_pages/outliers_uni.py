"""الكشف أحادي المتغيّر عن القيم الشاذة."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import charts, ui
from core.anim import play_animation
from core.datasets import clean_customers

ui.page_header(
    "الكشف أحادي المتغيّر (Univariate detection)",
    "طرق تفحص عمودًا واحدًا في كل مرّة. سريعة ومفهومة — لكن لها عيب قاتل "
    "يجب أن تعرفه: **المقاييس التي تستعملها (المتوسط والانحراف المعياري) "
    "ملوّثة بالشواذ نفسها**.",
    icon=":material/show_chart:",
    badges=[":blue-badge[z-score]", ":green-badge[MAD]", ":orange-badge[IQR]", ":red-badge[Masking]"],
)


@st.cache_data(ttl=1800, max_entries=8)
def demo_series(n: int = 400, n_out: int = 8) -> pd.DataFrame:
    """دخل واقعي (منحرف يمينًا) + شواذ مزروعة نعرف مواقعها."""
    df = clean_customers(n).copy()
    rng = np.random.default_rng(99)
    idx = rng.choice(n, n_out, replace=False)
    v = df["income"].to_numpy(dtype=float).copy()
    v[idx[: n_out // 2]] = v.max() * rng.uniform(3, 7, n_out // 2)
    v[idx[n_out // 2 :]] = v.min() * rng.uniform(0.05, 0.2, n_out - n_out // 2)
    return pd.DataFrame(
        {"income": v, "planted": np.isin(np.arange(n), idx)}
    )


def flags(v: np.ndarray, k: float) -> dict[str, np.ndarray]:
    """أعلام الشذوذ حسب كل طريقة عند معامل حساسية k."""
    mean, std = v.mean(), v.std(ddof=1)
    z = (v - mean) / std

    med = np.median(v)
    mad = np.median(np.abs(v - med))
    rz = 0.6745 * (v - med) / (mad if mad else 1e-9)

    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr

    p_lo, p_hi = np.percentile(v, [1, 99])

    return {
        "z-score": np.abs(z) > k,
        "modified z (MAD)": np.abs(rz) > k + 0.5,
        f"IQR ±{k:.1f}×": (v < lo) | (v > hi),
        "percentile 1–99%": (v < p_lo) | (v > p_hi),
    }


t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. z-score — والفخّ الذي فيه", divider="blue")
    st.latex(r"z_i = \frac{x_i - \bar{x}}{s}, \qquad |z_i| > 3 \;\Rightarrow\; \text{outlier}")
    st.caption("العتبة المعتادة ثلاثة انحرافات معيارية — وسنرى بعد قليل لماذا هي هشّة.")
    st.markdown(
        """
**الافتراض:** التوزيع طبيعي تقريبًا. تحت هذا الافتراض، 99.7٪ من القيم تقع
داخل ±3 انحرافات.

**المشكلة الأولى — التلوّث (contamination):** $\\bar{x}$ و $s$ يُحسبان من
بيانات **فيها الشواذ**. قيمة واحدة متطرّفة ترفع $s$ بشدّة، فتصبح القيم
المتطرّفة الأخرى «داخل الحدود».

- **Masking (التقنيع):** شاذّة تخفي شاذّة أخرى بتضخيم $s$.
- **Swamping (الإغراق):** مجموعة شواذ تزيح $\\bar{x}$ فتبدو قيم سليمة شاذّة.

**المشكلة الثانية — الحدّ الرياضي:** في عيّنة حجمها $n$، أقصى z ممكن هو:
"""
    )
    st.latex(r"|z|_{\max} \le \frac{n-1}{\sqrt{n}}")
    st.markdown(
        "فمع $n = 10$ لا يمكن لأي قيمة أن تتجاوز $|z| = 2.85$ — "
        "**قاعدة «z > 3» عديمة الفائدة تمامًا في العيّنات الصغيرة.**"
    )

    st.subheader("٢. البديل الأمتن: modified z-score بالـ MAD", divider="green")
    st.latex(r"\text{MAD} = \text{median}\left(|x_i - \tilde{x}|\right), \qquad M_i = \frac{0.6745\,(x_i - \tilde{x})}{\text{MAD}}")
    st.markdown(
        """
- الوسيط والـ MAD **مقياسان أمتن (robust)**: نقطة انهيارهما (breakdown point)
  = 50٪، أي يحتملان تلويث نصف البيانات قبل أن ينهارا. المتوسط نقطة انهياره
  = 0٪ (قيمة واحدة تكفي).
- الثابت `0.6745` يجعل MAD مكافئًا للانحراف المعياري في التوزيع الطبيعي.
- العتبة المقترحة في *Iglewicz & Hoaglin (1993)*: $|M_i| > 3.5$.
- :orange-badge[تحذير] إذا كانت أكثر من 50٪ من القيم متطابقة، يصبح
  `MAD = 0` وتنفجر القسمة. استعمل عندها `scipy.stats.median_abs_deviation`
  مع احتياط، أو انتقل إلى طريقة المئينات.
"""
    )

    st.subheader("٣. قاعدة Tukey (IQR) — بلا افتراض توزيع", divider="orange")
    st.latex(r"[\,Q_1 - k \cdot IQR,\;\; Q_3 + k \cdot IQR\,], \qquad IQR = Q_3 - Q_1")
    st.markdown(
        """
- $k = 1.5$ ⇒ «شاذ خفيف» (mild)، $k = 3.0$ ⇒ «شاذ متطرّف» (extreme).
- لا تفترض التوزيع الطبيعي، لكنها **تفترض التماثل**. في التوزيعات المنحرفة
  (كالدخل) تعلّم كثيرًا من القيم السليمة في الذيل الطويل.
- الحلّ للتوزيعات المنحرفة: **adjusted boxplot** الذي يعدّل الحدود بمعامل
  الالتواء medcouple، أو **حوّل التوزيع أولًا** (`log`) ثم طبّق القاعدة.
"""
    )

    st.subheader("٤. الاختبارات الإحصائية الرسمية", divider="violet")
    st.markdown(
        """
| الاختبار | الفرضية | الاستعمال | القيد |
|---|---|---|---|
| **Grubbs** | «لا توجد شاذّة» مقابل «توجد واحدة» | عيّنة صغيرة، شاذّة واحدة | يتطلّب **توزيعًا طبيعيًا**؛ يعاني masking |
| **Generalized ESD (Rosner)** | حتى $r$ شواذ | عدّة شواذ محتملة | يتطلّب تحديد $r$ مسبقًا وتوزيعًا طبيعيًا |
| **Dixon's Q** | شاذّة واحدة | $n$ صغير جدًا (3–30) | حسّاس للافتراضات |
| **Tietjen–Moore** | عدد معلوم من الشواذ | عدد محدّد سلفًا | نفس القيود |

:red-badge[تنبيه منهجي] هذه اختبارات **تأكيد**، لا استكشاف. تطبيقها
تكراريًا حتى «تنظيف» البيانات هو انتقاء نتائج (p-hacking) ويجب تجنّبه.
"""
    )

    ui.beginner_note(
        """
الفرق في مثال: راتبا الجميع بين 30 و 90 ألف، وشخص واحد يكسب 5 ملايين.
- **المتوسط والانحراف** يقفزان بسببه ⇒ يصبح هو نفسه «غير شاذ» بمقياس z.
- **الوسيط والـ MAD** لا يتأثّران إطلاقًا ⇒ يبقى شاذًا بوضوح.
ولهذا نفضّل المقاييس الأمتن في الكشف.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. الطرق الأربع في دالة واحدة", divider="blue")
    ui.code(
        '''
import numpy as np
import pandas as pd

def detect_univariate(s: pd.Series, method: str = "iqr", k: float = 1.5) -> pd.Series:
    """يرجع قناعًا منطقيًا: True = مشتبه به."""
    v = s.dropna().astype(float)

    if method == "zscore":
        z = (s - v.mean()) / v.std(ddof=1)
        return z.abs() > k                      # k = 3 عادة

    if method == "modified_z":                  # أمتن — مُستحسن
        med = v.median()
        mad = (v - med).abs().median()
        if mad == 0:
            return pd.Series(False, index=s.index)
        return (0.6745 * (s - med) / mad).abs() > k    # k = 3.5

    if method == "iqr":                         # بلا افتراض توزيع
        q1, q3 = v.quantile([0.25, 0.75])
        iqr = q3 - q1
        return (s < q1 - k * iqr) | (s > q3 + k * iqr)

    if method == "percentile":                  # بسيط وشفّاف
        lo, hi = v.quantile([0.01, 0.99])
        return (s < lo) | (s > hi)

    raise ValueError(method)

mask = detect_univariate(df["income"], "modified_z", 3.5)
print(f"{mask.sum()} قيمة مشتبه بها ({mask.mean():.1%})")
df.loc[mask, ["customer_id", "income"]].head(20)   # افحصها بعينك قبل أي قرار
'''
    )

    st.subheader("٢. اختبار Grubbs (عيّنات صغيرة وتوزيع طبيعي)", divider="green")
    ui.code(
        '''
import numpy as np
from scipy import stats

def grubbs_test(x, alpha=0.05):
    x = np.asarray(x, dtype=float)
    n = len(x)
    g = np.abs(x - x.mean()).max() / x.std(ddof=1)
    t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    critical = (n - 1) / np.sqrt(n) * np.sqrt(t**2 / (n - 2 + t**2))
    return g, critical, g > critical

g, crit, is_outlier = grubbs_test(df["income"].dropna())
print(f"G={g:.3f} vs critical={crit:.3f} ⇒ {'شاذّة مؤكّدة' if is_outlier else 'لا دليل'}")

# تحقّق من شرط الطبيعية أولًا، وإلّا فالاختبار بلا معنى
print(stats.shapiro(df["income"].dropna().sample(500, random_state=0)))
'''
    )

    st.subheader("٣. التوزيعات المنحرفة: حوّل ثم اكشف", divider="orange")
    ui.code(
        """
import numpy as np

# الدخل منحرف يمينًا ⇒ IQR الخام يعلّم ذيله الطبيعي كله
log_income = np.log1p(df["income"])
q1, q3 = log_income.quantile([.25, .75]); iqr = q3 - q1
mask = (log_income < q1 - 1.5*iqr) | (log_income > q3 + 1.5*iqr)
print("بعد التحويل:", mask.sum(), "مقابل الخام:", detect_univariate(df['income']).sum())
""",
        caption="قاعدة عملية: **عالج الانحراف قبل أن تحكم على الشواذ** — وإلّا "
        "ستحذف الذيل الطبيعي للتوزيع وتسمّيه تنظيفًا.",
    )

    st.subheader("٤. الكشف داخل السياق (شاذ سياقي)", divider="violet")
    ui.code(
        """
# بدل عتبة عامة واحدة: عتبة لكل مجموعة
def robust_z_by_group(df, col, by):
    g = df.groupby(by)[col]
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median())
    return 0.6745 * (df[col] - med) / mad.replace(0, np.nan)

df["rz_city"] = robust_z_by_group(df, "spending", "city")
df[df["rz_city"].abs() > 3.5]
"""
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("حرّك العتبة وشاهد من يُعلَّم", divider="violet")
    st.markdown(
        "بيانات دخل حقيقية الشكل (منحرفة يمينًا) زرعنا فيها **٨ شواذ نعرفها**. "
        "الأنيميشن يرفع معامل الحساسية $k$ تدريجيًا ويقارن الطرق: كم شاذّة "
        "حقيقية أمسكت؟ وكم قيمة سليمة اتُّهمت ظلمًا؟"
    )

    d = demo_series()
    v = d["income"].to_numpy(dtype=float)
    planted = d["planted"].to_numpy()
    KS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

    def _render_threshold(frame: int) -> None:
        k = KS[frame]
        f = flags(v, k)

        rows = []
        for name, m in f.items():
            tp = int((m & planted).sum())
            fp = int((m & ~planted).sum())
            rows.append(
                {
                    "method": name,
                    "flagged": int(m.sum()),
                    "true_outliers_caught": tp,
                    "false_alarms": fp,
                    "recall": tp / planted.sum(),
                }
            )
        summary = pd.DataFrame(rows)

        st.markdown(f"###### معامل الحساسية الحالي: **k = {k}**")
        left, right = st.columns([3, 2])
        with left:
            dd = d.assign(
                z_flag=f["z-score"],
                mad_flag=f["modified z (MAD)"],
                label=np.where(
                    planted, "planted outlier",
                    np.where(f["modified z (MAD)"], "flagged by MAD", "clean"),
                ),
                idx=np.arange(len(d)),
            )
            ch = (
                alt.Chart(dd)
                .mark_circle(size=70, opacity=0.8)
                .encode(
                    x=alt.X("idx:Q", title="observation index"),
                    y=alt.Y("income:Q", title="income", scale=alt.Scale(type="symlog")),
                    color=alt.Color(
                        "label:N",
                        legend=alt.Legend(orient="top", title=None),
                        scale=alt.Scale(
                            domain=["clean", "flagged by MAD", "planted outlier"],
                            range=["#4C6EF5", "#F08C00", "#E8590C"],
                        ),
                    ),
                    tooltip=["income:Q", "label:N"],
                )
                .properties(height=330)
            )
            st.altair_chart(ch, width="stretch")
        with right:
            st.altair_chart(
                charts.bar(
                    summary[["method", "flagged"]], "method", "flagged", horizontal=True, height=150
                ),
                width="stretch",
            )
            st.dataframe(
                summary.assign(recall=summary["recall"].map("{:.0%}".format)),
                width="stretch",
                hide_index=True,
            )

        if k <= 2:
            st.markdown(
                ":red-badge[عتبة متساهلة] عدد كبير من القيم السليمة يُعلَّم — "
                "هذا **الإغراق (swamping)**."
            )
        elif k >= 4:
            st.markdown(
                ":orange-badge[عتبة متشدّدة] z-score يبدأ بفقد شواذ حقيقية لأن "
                "الانحراف المعياري نفسه تضخّم بسببها — هذا **التقنيع (masking)**."
            )
        else:
            st.markdown(
                ":green-badge[منطقة متوازنة] لاحظ أن **modified z (MAD)** يمسك "
                "شواذ أكثر عند نفس العتبة، لأنه لا يتلوّث بها."
            )

    play_animation("uni_threshold", len(KS), _render_threshold, fps=0.6, label="k")

# ============================================================== القرار
with t_decide:
    st.subheader("أي طريقة أستعمل؟", divider="orange")
    ui.decision_table(
        [
            ("توزيع طبيعي تقريبًا، تلوّث قليل", "z-score بعتبة 3", "بسيط وشائع ومفهوم للمُراجعين"),
            ("توزيع غير طبيعي أو تلوّث ملموس", "modified z (MAD) بعتبة 3.5", "أمتن؛ نقطة انهيار 50٪"),
            ("لا أعرف التوزيع، أريد قاعدة بصرية", "IQR مع k=1.5 (أو 3 للمتطرّف)", "بلا افتراض توزيع، ومتوافق مع boxplot"),
            ("توزيع منحرف بشدّة (دخل، أسعار)", "حوّل (`log1p`) أولًا، أو adjusted boxplot", "القاعدة الخام تعاقب الذيل الطبيعي"),
            ("عيّنة صغيرة (n < 30)", "Grubbs / Dixon مع فحص الطبيعية", "قاعدة z > 3 مستحيلة رياضيًا في n الصغيرة"),
            ("أريد نسبة قصّ ثابتة", "Winsorizing عند 1٪ و99٪", "شفّاف وسهل التوثيق، لكنه تعسّفي"),
            ("الشذوذ سياقي", "تقييس داخل المجموعة (groupby)", "العتبة العامة تخلط بين سياقات مختلفة"),
            ("أعمدة كثيرة", "لا تجمع الأعلام آليًا", "تضخّم الإيجابيات الكاذبة مع كل عمود إضافي"),
        ],
        headers=("الحالة", "الطريقة", "السبب"),
    )

    ui.tip(
        "لا تعتمد على طريقة واحدة. طبّق اثنتين أو ثلاثًا وانظر إلى **التقاطع**: "
        "القيم التي تعلّمها كل الطرق هي الأجدر بالفحص اليدوي."
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("Iglewicz & Hoaglin (1993) — How to Detect and Handle Outliers", "https://asq.org/quality-press/display-item?item=H0613"),
            ("NIST/SEMATECH e-Handbook — Grubbs' Test", "https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm"),
            ("NIST/SEMATECH e-Handbook — Generalized ESD Test", "https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm"),
            ("SciPy — median_abs_deviation", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html"),
            ("Hubert & Vandervieren (2008) — An adjusted boxplot for skewed distributions", "https://doi.org/10.1016/j.csda.2007.11.008"),
            ("Shiffler (1988) — Maximum Z Scores and Outliers", "https://doi.org/10.1080/00031305.1988.10475530"),
        ]
    )
    ui.terms(
        [
            ("z-score", "الدرجة المعيارية"),
            ("MAD", "متوسط الانحراف المطلق عن الوسيط"),
            ("Robust statistic", "إحصاءة أمتن (لا تتأثّر بالشواذ)"),
            ("Breakdown point", "نقطة الانهيار (نسبة التلوّث المحتملة)"),
            ("Masking", "تقنيع: شاذّة تخفي أخرى"),
            ("Swamping", "إغراق: اتّهام قيم سليمة"),
            ("Tukey fences", "حدود تيوكي القائمة على IQR"),
        ]
    )
