"""فحص جودة البيانات — أول ١٥ دقيقة مع أي جدول."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import charts, ui
from core.anim import play_animation
from core.datasets import messy_customers

ui.page_header(
    "فحص جودة البيانات (Data quality audit)",
    "لا تبدأ بالمعالجة؛ ابدأ بالتشخيص. هذه الصفحة تعطيك **تقريرًا آليًا** "
    "يرفع الأعلام الحمراء في أي جدول، وشرحًا لما يعنيه كل علم.",
    icon=":material/fact_check:",
    badges=[
        ":blue-badge[Profiling]",
        ":green-badge[Completeness]",
        ":orange-badge[Validity]",
        ":red-badge[Red flags]",
    ],
)

HIDDEN_NA = {-99, -999, -1, 9999, "?", "N/A", "NA", "na", "n/a", "none", "None", "null", "NULL", "", " ", "-", "unknown", "Unknown", "missing"}


@st.cache_data(ttl=1800, max_entries=8)
def audit(df: pd.DataFrame) -> pd.DataFrame:
    """تقرير جودة عمودي: نوع، فقد، تفرّد، شذوذ، وأعلام حمراء."""
    n = len(df)
    rows = []
    for c in df.columns:
        s = df[c]
        nun = int(s.nunique(dropna=True))
        flags: list[str] = []

        pct_na = float(s.isna().mean() * 100)
        if pct_na > 0:
            flags.append("missing")
        if pct_na > 40:
            flags.append("missing>40%")
        if nun <= 1:
            flags.append("constant")
        if nun == n and n > 0:
            flags.append("id-like")

        hidden = int(s.isin(list(HIDDEN_NA)).sum()) if s.dtype != "datetime64[ns]" else 0
        if hidden > 0:
            flags.append(f"hidden-NA({hidden})")

        n_out = 0
        skew = np.nan
        if pd.api.types.is_numeric_dtype(s):
            v = s.dropna().astype(float)
            if len(v) > 5:
                q1, q3 = v.quantile([0.25, 0.75])
                iqr = q3 - q1
                n_out = int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum())
                skew = float(v.skew())
                if n_out > 0:
                    flags.append(f"outliers({n_out})")
                if abs(skew) > 1:
                    flags.append("skewed")
        else:
            # نص يبدو رقميًا؟
            sample = s.dropna().astype(str).head(50)
            numericish = pd.to_numeric(
                sample.str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce"
            ).notna().mean()
            if len(sample) and numericish > 0.9:
                flags.append("numeric-stored-as-text")
            if len(sample) and sample.str.strip().str.lower().nunique() < sample.nunique():
                flags.append("case/space inconsistency")
            if 1 < nun <= 50:
                pass
            elif nun > 50 and nun < n:
                flags.append("high-cardinality")

        rows.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "pct_missing": round(pct_na, 1),
                "n_unique": nun,
                "skew": round(skew, 2) if pd.notna(skew) else None,
                "n_outliers_iqr": n_out,
                "flags": ", ".join(flags) if flags else "ok",
            }
        )
    return pd.DataFrame(rows)


t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. أبعاد جودة البيانات السبعة", divider="blue")
    ui.concept_grid(
        [
            ("Completeness — الاكتمال", "كم نسبة الفقد في كل عمود وصف؟ وهل الفقد مُرمَّز بـ `NaN` فعلًا أم مخبّأ خلف `-99` و «غير متوفّر»؟", "blue"),
            ("Uniqueness — التفرّد", "تكرار تام أو جزئي للصفوف؟ هل المفتاح (`customer_id`) فريد حقًا؟", "green"),
            ("Validity — الصلاحية", "هل القيم داخل المدى المنطقي؟ عمر = 199، نسبة = 1.4، تاريخ ميلاد في المستقبل.", "orange"),
            ("Consistency — الاتساق", "`Algiers` و `algiers` و `ALGIERS`؛ وحدات مختلطة (دينار/يورو، كلغ/رطل)؛ تاريخان متناقضان.", "violet"),
            ("Accuracy — الدقّة", "هل القيمة تطابق الواقع؟ لا يُكتشف من داخل الجدول — يحتاج مصدرًا مرجعيًا أو خبيرًا.", "red"),
            ("Timeliness — الحداثة", "متى جُمعت البيانات؟ هل توزيعها ما زال يمثّل اليوم (data drift)؟", "blue"),
            ("Representativeness — التمثيل", "هل العيّنة تمثّل المجتمع المستهدف؟ تحيّز الاختيار (selection bias) لا يُصلحه أي تنظيف.", "green"),
            ("Target sanity — سلامة الهدف", "توزيع الهدف، نسبة الفئة النادرة، وهل يوجد عمود يرتبط به ارتباطًا شبه تام (= تسريب).", "red"),
        ],
        n_cols=2,
    )

    st.subheader("٢. الأعلام الحمراء التي يجب أن تبحث عنها دائمًا", divider="red")
    st.markdown(
        """
| العلم | ماذا يعني غالبًا | أول ردّ فعل |
|---|---|---|
| عمود بقيمة واحدة | عمود مُهمل أو فلترة سابقة | احذفه |
| عمود بعدد فئات = عدد الصفوف | معرّف أو نص حر | أخرجه من المتغيّرات |
| قيمة `-99` / `0` متكرّرة بشكل غير طبيعي | رمز فقد مخفي | حوّله إلى `NaN` قبل أي حساب |
| `dtype = object` لعمود رقمي | فاصلة أو رمز عملة أو مسافة | نظّف ثم `to_numeric` |
| انحراف (skew) > 2 | ذيل طويل / قيم شاذة | افحص الذيل قبل التحويل |
| ارتباط متغيّر بالهدف > 0.95 | تسريب شبه مؤكّد | تحقّق من كيفية توليد العمود |
| فئة واحدة تغطّي > 99٪ | تباين شبه صفري | غالبًا بلا فائدة تنبؤية |
| صفوف مكرّرة تمامًا | خطأ في الاستخراج / دمج | احذف، وافهم السبب |

:orange-badge[ملاحظة] العلم الأحمر ليس حكمًا نهائيًا — هو **سؤال** يجب أن
تجيب عنه قبل المضي قدمًا.
"""
    )

    ui.beginner_note(
        """
فكّر في الأمر كفحص طبي: لا يصف الطبيب دواءً قبل التحاليل. هذه الصفحة هي
«التحاليل»: ترفع لك قائمة بما هو غير طبيعي، وأنت تقرّر العلاج في الدروس
التالية.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. الفحص السريع (٥ أسطر)", divider="blue")
    ui.code(
        """
df.shape
df.info(memory_usage="deep")
df.isna().mean().sort_values(ascending=False).head(15)   # نسب الفقد
df.duplicated().sum()                                     # تكرار تام
df.describe(include="all").T
"""
    )

    st.subheader("٢. دالة تقرير جاهزة للنسخ", divider="green")
    ui.code(
        '''
import numpy as np
import pandas as pd

HIDDEN_NA = [-99, -999, "?", "N/A", "NA", "", " ", "unknown", "none", "null"]

def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    out = []
    for c in df.columns:
        s = df[c]
        flags = []
        pct_na = s.isna().mean() * 100
        nun = s.nunique(dropna=True)

        if pct_na > 40:                      flags.append("missing>40%")
        if nun <= 1:                         flags.append("constant")
        if nun == n:                         flags.append("id-like")
        if s.isin(HIDDEN_NA).any():          flags.append("hidden-NA")

        n_out = 0
        if pd.api.types.is_numeric_dtype(s):
            v = s.dropna()
            q1, q3 = v.quantile([.25, .75]); iqr = q3 - q1
            n_out = ((v < q1 - 1.5*iqr) | (v > q3 + 1.5*iqr)).sum()
            if abs(v.skew()) > 1:            flags.append("skewed")
        elif nun > 50 and nun < n:           flags.append("high-cardinality")

        out.append(dict(column=c, dtype=str(s.dtype), pct_missing=round(pct_na, 1),
                        n_unique=nun, n_outliers=int(n_out),
                        flags=", ".join(flags) or "ok"))
    return pd.DataFrame(out).sort_values("pct_missing", ascending=False)

report = quality_report(df)
print(report.to_string(index=False))
''',
        caption="شغّلها على كل جدول جديد قبل أي شيء آخر، واحفظ مخرجاتها في ملف المشروع.",
    )

    st.subheader("٣. فحوص إضافية مهمّة", divider="orange")
    ui.code(
        """
# تكرار جزئي: نفس العميل بمعرّف مختلف
df.duplicated(subset=["name", "birth_date", "city"], keep=False).sum()

# صلاحية المدى (domain validity)
bad_age   = df.query("age < 0 or age > 110")
bad_ratio = df.query("ratio < 0 or ratio > 1")
future    = df[df["signup_date"] > pd.Timestamp.today()]

# اتساق منطقي بين عمودين
df.query("end_date < start_date")
df.query("spending > income * 2")

# تسريب محتمل: ارتباط شبه تام بالهدف
corr = df.select_dtypes("number").corrwith(df["target"]).abs().sort_values(ascending=False)
print(corr[corr > 0.95])

# توزيع الهدف
df["target"].value_counts(normalize=True)
"""
    )

    st.subheader("٤. التقرير على بياناتنا «الوسخة»", divider="violet")
    df = messy_customers()
    rep = audit(df)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("الصفوف", f"{len(df):,}")
    k2.metric("الأعمدة", f"{df.shape[1]}")
    k3.metric("صفوف مكرّرة تمامًا", f"{int(df.duplicated().sum())}")
    k4.metric("خلايا مفقودة", f"{int(df.isna().sum().sum()):,}")

    only_flagged = st.toggle("أظهر الأعمدة التي رُفع لها علم فقط", value=True)
    view = rep[rep["flags"] != "ok"] if only_flagged else rep
    st.dataframe(
        view,
        width="stretch",
        hide_index=True,
        column_config={
            "pct_missing": st.column_config.ProgressColumn(
                "pct_missing", format="%.1f%%", min_value=0, max_value=100
            ),
            "flags": st.column_config.TextColumn("flags", width="large"),
        },
    )

    st.markdown("###### نسب الفقد لكل عمود")
    miss = rep[rep["pct_missing"] > 0][["column", "pct_missing"]]
    if len(miss):
        st.altair_chart(
            charts.bar(miss, "column", "pct_missing", horizontal=True, height=260),
            width="stretch",
        )
    else:
        st.caption("لا فقد في هذه النسخة.")

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("فحص آلي عمودًا عمودًا", divider="violet")
    st.markdown("شغّل العرض لترى الماسح يمرّ على الأعمدة ويرفع الأعلام واحدًا واحدًا.")

    df_a = messy_customers()
    rep_a = audit(df_a).reset_index(drop=True)

    def _render_scan(frame: int) -> None:
        row = rep_a.iloc[frame]
        done = rep_a.iloc[: frame + 1]
        n_flag = int((done["flags"] != "ok").sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("العمود قيد الفحص", str(row["column"]))
        c2.metric("أعمدة مفحوصة", f"{frame + 1}/{len(rep_a)}")
        c3.metric("أعلام مرفوعة", n_flag)

        with st.container(border=True):
            ok = row["flags"] == "ok"
            badge = ":green-badge[سليم]" if ok else ":red-badge[يحتاج انتباهًا]"
            st.markdown(f"##### `{row['column']}` — {badge}")
            st.markdown(
                f"- **dtype:** `{row['dtype']}`\n"
                f"- **نسبة الفقد:** {row['pct_missing']}%\n"
                f"- **قيم فريدة:** {row['n_unique']}\n"
                f"- **الانحراف (skew):** {row['skew']}\n"
                f"- **شواذ بقاعدة IQR:** {row['n_outliers_iqr']}\n"
                f"- **الأعلام:** `{row['flags']}`"
            )
        st.progress((frame + 1) / len(rep_a))

    play_animation(
        "audit_scan",
        len(rep_a),
        _render_scan,
        fps=1.2,
        label="العمود",
    )

# ============================================================== القرار
with t_decide:
    st.subheader("من العلم إلى الفعل", divider="orange")
    ui.decision_table(
        [
            ("`constant`", "احذف العمود", "تباين صفري ⇒ لا معلومة، لكنه يضيف تكلفة"),
            ("`id-like`", "أخرجه من `X` واحتفظ به للربط", "يسبّب حفظًا (memorisation) وتسريبًا"),
            ("`hidden-NA`", "حوّله إلى `NaN` **قبل** أي إحصاء", "وإلا دخل `-99` في المتوسط وأفسد كل شيء"),
            ("`numeric-stored-as-text`", "نظّف ثم `to_numeric(errors='coerce')`", "ثم افحص كم قيمة فشلت في التحويل"),
            ("`case/space inconsistency`", "`.str.strip().str.title()` + قاموس مرادفات", "فئة واحدة مقسّمة إلى عدّة فئات تُضعف الترميز"),
            ("`high-cardinality`", "Target/Frequency encoding أو تجميع النادر", "One-Hot سيفجّر الأبعاد"),
            ("`skewed`", "افحص الذيل: شواذ أم توزيع طبيعي منحرف؟", "القرار بين الحذف والتحويل يختلف جذريًا"),
            ("`outliers(k)`", "لا تحذف قبل الفهم", "قد تكون أهم صفوف الجدول"),
            ("`missing>40%`", "قرار مركّب: آلية الفقد + أهمية العمود", "النسبة وحدها ليست معيارًا"),
            ("ارتباط بالهدف > 0.95", "تحقّق من مصدر العمود فورًا", "غالبًا تسريب (عمود يُحسب بعد معرفة الهدف)"),
        ],
        headers=("العلم المرفوع", "الإجراء", "السبب"),
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("pandas — Working with missing data", "https://pandas.pydata.org/docs/user_guide/missing_data.html"),
            ("pandas — DataFrame.describe", "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html"),
            ("pandas — DataFrame.duplicated", "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.duplicated.html"),
            ("ydata-profiling — تقرير استكشافي آلي", "https://docs.profiling.ydata.ai/"),
            ("Great Expectations — توثيق اختبارات جودة البيانات", "https://docs.greatexpectations.io/"),
            ("scikit-learn — Common pitfalls", "https://scikit-learn.org/stable/common_pitfalls.html"),
        ]
    )
    ui.terms(
        [
            ("Data profiling", "استكشاف آلي لخصائص البيانات"),
            ("Completeness", "نسبة القيم غير المفقودة"),
            ("Cardinality", "عدد القيم الفريدة"),
            ("Sentinel value", "قيمة تُستعمل كرمز للفقد مثل -99"),
            ("Data drift", "تغيّر توزيع البيانات مع الزمن"),
            ("Selection bias", "تحيّز في طريقة اختيار العيّنة"),
        ]
    )
