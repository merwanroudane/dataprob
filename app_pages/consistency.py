"""توحيد الصيغ والأنواع والوحدات."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import charts, ui
from core.anim import play_animation
from core.datasets import messy_customers

ui.page_header(
    "توحيد الصيغ والأنواع (Consistency)",
    "«Algiers» و «algiers» و «ALGIERS» و « Alger » أربع فئات مختلفة في نظر "
    "الكمبيوتر، وفئة واحدة في نظرك. هذا الدرس عن إغلاق الفجوة بين الاثنين — "
    "وهو **تنظيف بنيوي يسبق التقسيم**.",
    icon=":material/format_align_right:",
    badges=[
        ":blue-badge[Case / whitespace]",
        ":green-badge[Units]",
        ":orange-badge[Dates]",
        ":violet-badge[Encoding]",
        ":red-badge[Cross-field rules]",
    ],
)

t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. سبع جبهات لعدم الاتساق", divider="blue")
    ui.concept_grid(
        [
            (
                "١. الحالة والمسافات",
                "`\" Oran \"` ≠ `\"Oran\"` ≠ `\"oran\"`. المسافات غير المرئية "
                "(tab، مسافة غير فاصلة `\\xa0`) هي الأكثر خداعًا لأنها لا تُرى "
                "في العرض.",
                "blue",
            ),
            (
                "٢. المرادفات والاختصارات",
                "`USA` / `U.S.A.` / `United States`؛ `DZ` / `Algérie` / "
                "`الجزائر`. يحتاج **قاموس تعيين** تبنيه يدويًا — لا توجد قاعدة "
                "آلية.",
                "green",
            ),
            (
                "٣. الوحدات والمقاييس",
                "دينار مقابل أورو؛ كلغ مقابل رطل؛ متر مقابل قدم. "
                "أخطر نوع لأنه **لا يُكشف إحصائيًا**: القيم تبدو معقولة، "
                "والنموذج يتعلّم خلطًا.",
                "orange",
            ),
            (
                "٤. التواريخ والمناطق الزمنية",
                "`05/01/2023`: الخامس من يناير أم الأول من مايو؟ "
                "`dayfirst` قرار يغيّر البيانات كلّها. وتوقيت UTC مقابل محلي "
                "يزيح الأحداث ساعات.",
                "violet",
            ),
            (
                "٥. الصيغ الرقمية",
                "`1.234,56` (أوروبي) مقابل `1,234.56` (أنجلوسكسوني)؛ "
                "رموز العملة؛ النسب المكتوبة `45%`. القراءة الخاطئة تضرب "
                "القيمة في 1000.",
                "red",
            ),
            (
                "٦. الترميز النصّي (Encoding)",
                "UTF-8 مقابل Latin-1 مقابل Windows-1256. الأعراض: `Ø§Ù„Ø¬` أو "
                "`Ã©` بدل `é`. يجب إصلاحه عند **القراءة**، لا بعدها.",
                "blue",
            ),
            (
                "٧. التناقض بين الحقول",
                "`end_date < start_date`؛ `experience_years > age - 14`؛ "
                "`spending > income * 3`. تُكتشف بقواعد مجال، لا بإحصاء.",
                "green",
            ),
            (
                "٨. الثنائيات",
                "`Yes/No`, `Y/N`, `1/0`, `True/False`, `نعم/لا`, `TRUE`, "
                "`oui`. في نفس العمود أحيانًا.",
                "orange",
            ),
        ],
        n_cols=4,
    )

    st.subheader("٢. لماذا يسبق هذا التقسيم؟", divider="green")
    st.markdown(
        """
كل ما في هذا الدرس **تصحيح ترميز حتمي**: قاعدة ثابتة تُطبَّق على الصف وحده
دون النظر إلى بقيّة الصفوف. لذلك:

- :green-badge[آمن قبل التقسيم] `.str.strip()`، `.str.lower()`، تحويل التاريخ،
  تصحيح الوحدات، قاموس المرادفات المبني على معرفة المجال.
- :red-badge[ليس آمنًا] تجميع الفئات النادرة في `Other` بناءً على **تكرارها
  في البيانات** — هذا إحصاء مُتعلَّم، مكانه داخل `Pipeline` بعد التقسيم
  (`OneHotEncoder(min_frequency=...)`).

الفرق دقيق لكنه جوهري: القاعدة الأولى تعرفها قبل أن ترى البيانات؛ الثانية
تستخرجها منها.
"""
    )

    ui.danger(
        "أثر عدم الاتساق على الترميز: فئة واحدة مكتوبة بخمس صيغ تنتج **خمسة "
        "أعمدة** بعد One-Hot، كلٌّ منها بعدد مشاهدات ضئيل ⇒ تقديرات غير "
        "مستقرّة، وفئات نادرة مزيّفة، وأبعاد منتفخة بلا سبب."
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. توحيد النصوص", divider="blue")
    ui.code(
        '''
import unicodedata
import pandas as pd

def clean_text(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
         .str.normalize("NFKC")          # يوحّد الأشكال المتكافئة للمحارف
         .str.replace(r"\\s+", " ", regex=True)   # مسافات متعدّدة ← واحدة
         .str.strip()
         .str.title()                     # أو .str.lower() حسب الاتفاق
    )

df["city"] = clean_text(df["city"])

# إزالة التشكيل/اللكنات اللاتينية (é ← e)
def strip_accents(s):
    return (s.str.normalize("NFKD")
             .str.encode("ascii", "ignore").str.decode("utf-8"))

# قاموس المرادفات — يُبنى يدويًا من `value_counts()`
SYNONYMS = {"Alger": "Algiers", "Dz": "Algeria", "U.S.A.": "USA", "United States": "USA"}
df["city"] = df["city"].replace(SYNONYMS)

print(df["city"].value_counts())   # راجع النتيجة دائمًا بعينك
'''
    )

    st.subheader("٢. الأرقام المخبّأة في نصوص", divider="green")
    ui.code(
        '''
# "25 000 DA" / "1,234.56" / "1.234,56" / "45%"
s = df["income_text"].astype("string")

# صيغة أنجلوسكسونية
num = pd.to_numeric(s.str.replace(r"[^\\d.\\-]", "", regex=True), errors="coerce")

# صيغة أوروبية (النقطة للآلاف، الفاصلة للعشور)
eu = pd.to_numeric(
    s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
    errors="coerce",
)

# النسب
pct = pd.to_numeric(s.str.rstrip("%"), errors="coerce") / 100

failed = num.isna() & s.notna()
print(f"{failed.sum()} قيمة فشل تحويلها:", s[failed].unique()[:10])
''',
        caption="افحص **دائمًا** ما فشل تحويله — غالبًا يكشف صيغة لم تتوقّعها.",
    )

    st.subheader("٣. التواريخ", divider="orange")
    ui.code(
        """
# حدّد الصيغة صراحةً إن عرفتها — أسرع وأأمن
df["signup_date"] = pd.to_datetime(df["signup_date"], format="%d/%m/%Y", errors="coerce")

# صيغ مختلطة
df["d"] = pd.to_datetime(df["d"], format="mixed", dayfirst=True, errors="coerce")

print("فشل التحويل في:", df["signup_date"].isna().sum(), "صفًا")

# المناطق الزمنية: وحّد على UTC ثم حوّل للعرض فقط
df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Africa/Algiers")

# فحوص منطقية
print((df["signup_date"] > pd.Timestamp.today()).sum(), "تاريخًا في المستقبل")
print((df["end"] < df["start"]).sum(), "صفًا ينتهي قبل أن يبدأ")
"""
    )

    st.subheader("٤. الثنائيات والوحدات", divider="violet")
    ui.code(
        """
BOOLS = {"yes": True, "y": True, "1": True, "true": True, "نعم": True,
         "no": False, "n": False, "0": False, "false": False, "لا": False}
df["subscribed"] = df["subscribed"].astype("string").str.strip().str.lower().map(BOOLS).astype("boolean")

# توحيد الوحدات عبر عمود يحدّدها
to_kg = {"kg": 1.0, "g": 0.001, "lb": 0.453592}
df["weight_kg"] = df["weight"] * df["unit"].map(to_kg)

# توحيد العملة بسعر صرف مؤرّخ (لا بسعر اليوم!)
df["amount_dzd"] = df["amount"] * df.merge(fx, on=["currency", "date"])["rate"]
""",
        caption=":red-badge[انتبه] استعمال سعر صرف اليوم على معاملات قديمة "
        "تسريب من المستقبل في أي مهمّة تنبّؤية.",
    )

    st.subheader("٥. قواعد التناسق بين الحقول", divider="red")
    ui.code(
        '''
RULES = {
    "age_range":        "age.between(16, 110)",
    "exp_le_age":       "experience_years <= age - 14",
    "spend_le_income":  "spending <= income * 2",
    "ratio_unit":       "ratio.between(0, 1)",
}

report = {}
for name, expr in RULES.items():
    ok = df.eval(expr)
    report[name] = int((~ok).sum())
print(report)      # صفر = القاعدة محترمة؛ غير ذلك يحتاج قرارًا
''',
        caption="أطر جاهزة لأتمتة هذه القواعد: `pandera` و `Great Expectations`.",
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("توحيد عمود الفئات خطوة بخطوة", divider="violet")
    st.markdown(
        "نبدأ من عمود `city` كما وصل — بكل صيغه — ونطبّق خطوة تنظيف في كل "
        "إطار. راقب **عدد الفئات** ينهار إلى العدد الحقيقي."
    )

    raw = messy_customers()["city"].astype("string")

    STEPS = [
        ("الخام كما وصل", lambda s: s, "خمس صيغ لمدينة واحدة — كل واحدة ستصبح عمودًا مستقلًا بعد One-Hot."),
        ("`.str.normalize('NFKC')`", lambda s: s.str.normalize("NFKC"), "توحيد الأشكال المتكافئة للمحارف (مهم جدًا في العربية والفرنسية)."),
        ("`.str.replace(r'\\s+', ' ')` + `.str.strip()`", lambda s: s.str.replace(r"\s+", " ", regex=True).str.strip(), "إزالة المسافات الزائدة وغير المرئية."),
        ("`.str.title()`", lambda s: s.str.replace(r"\s+", " ", regex=True).str.strip().str.title(), "توحيد حالة الأحرف: `ALGIERS` و `algiers` ← `Algiers`."),
        ("قاموس المرادفات", lambda s: s.str.replace(r"\s+", " ", regex=True).str.strip().str.title().replace({"Alger": "Algiers"}), "`Alger` ← `Algiers`: خطوة تحتاج **معرفتك بالمجال**، لا الكود."),
    ]

    def _render_clean(frame: int) -> None:
        title, fn, note = STEPS[frame]
        s = fn(raw)
        vc = s.value_counts(dropna=False).reset_index()
        vc.columns = ["city", "count"]
        vc["city"] = vc["city"].astype(str)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("عدد الفئات", int(s.nunique(dropna=True)), delta=None)
            st.metric(
                "أعمدة بعد One-Hot",
                int(s.nunique(dropna=True)),
                help="كل فئة زائفة = عمود زائد وتقدير أضعف",
            )
            st.markdown(f"**الخطوة:** {title}")
            st.markdown(note)
        with c2:
            st.altair_chart(
                charts.bar(vc.head(12), "city", "count", horizontal=True, height=300),
                width="stretch",
            )

    play_animation("consistency_clean", len(STEPS), _render_clean, fps=0.55, label="الخطوة")

    st.subheader("فحص التناسق على بياناتنا", divider="orange")
    dfm = messy_customers()
    checks = {
        "age خارج [16, 110]": int((~dfm["age"].between(16, 110)).sum()),
        "satisfaction سالبة (رمز -99)": int((dfm["satisfaction"] < 0).sum()),
        "spending > 2 × income": int((dfm["spending"] > dfm["income"] * 2).sum()),
        "experience_years > age - 14": int((dfm["experience_years"] > dfm["age"] - 14).sum()),
        "income مخزّن كنص": int(dfm["income_text"].notna().sum()),
    }
    res = pd.DataFrame({"check": list(checks), "violations": list(checks.values())})
    st.dataframe(res, width="stretch", hide_index=True)
    st.caption(
        "لاحظ `satisfaction` السالبة: ليست «قيمة شاذّة» بل **رمز فقد مقنّع** "
        "(`-99`). لو حسبت المتوسط قبل اكتشافه لحصلت على رقم خاطئ تمامًا."
    )

# ============================================================== القرار
with t_decide:
    st.subheader("ماذا أفعل بكل حالة؟", divider="orange")
    ui.decision_table(
        [
            ("اختلاف حالة/مسافات", ":green-badge[وحّد قبل التقسيم]", "قاعدة حتمية لا تتعلّم من البيانات"),
            ("مرادفات معروفة", ":green-badge[قاموس تعيين يدوي]", "معرفة مجال، لا إحصاء"),
            ("تشابه نصّي غير مؤكّد", ":orange-badge[تطابق تقريبي + مراجعة بشرية]", "الدمج الآلي ينتج أخطاء صامتة"),
            ("وحدات مختلطة", ":red-badge[أصلح من المصدر إن أمكن]", "لا يُكشف إحصائيًا وأثره كارثي"),
            ("صيغ تاريخ مختلطة", ":blue-badge[`format='mixed'` + `dayfirst` صريح + فحص]", "الغموض بين يوم وشهر يغيّر البيانات بصمت"),
            ("رموز فقد مقنّعة (-99, 'N/A')", ":green-badge[حوّلها إلى NaN فورًا]", "وإلا دخلت في كل الإحصاءات"),
            ("تناقض بين حقول", ":violet-badge[قاعدة مجال صريحة + قرار موثّق]", "قد يكون خطأ إدخال أو حالة حقيقية نادرة"),
            ("تجميع الفئات النادرة في Other", ":red-badge[بعد التقسيم، داخل Pipeline]", "«النادر» يُحدَّد من تكرار التدريب ⇒ معامل مُتعلَّم"),
            ("ترميز ملف خاطئ", ":green-badge[أصلحه عند القراءة]", "`pd.read_csv(..., encoding='utf-8-sig')`"),
        ]
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("pandas — Working with text data (.str)", "https://pandas.pydata.org/docs/user_guide/text.html"),
            ("pandas — Time series: to_datetime", "https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html"),
            ("pandas — IO tools (encoding, parsing)", "https://pandas.pydata.org/docs/user_guide/io.html"),
            ("pandera — تحقّق تصريحي من صحّة الجداول", "https://pandera.readthedocs.io/"),
            ("Great Expectations — اختبارات جودة البيانات", "https://docs.greatexpectations.io/"),
            ("Unicode — Normalization Forms (UAX #15)", "https://unicode.org/reports/tr15/"),
            ("scikit-learn — OneHotEncoder (min_frequency)", "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html"),
        ]
    )
    ui.terms(
        [
            ("Normalization (Unicode)", "توحيد تمثيل المحارف"),
            ("Canonical form", "الصيغة المرجعية المعتمدة"),
            ("Mapping dictionary", "قاموس تعيين المرادفات"),
            ("Cross-field validation", "تحقّق بين عدّة حقول"),
            ("Encoding (text)", "ترميز المحارف (UTF-8 …)"),
            ("Deterministic cleaning", "تنظيف حتمي لا يتعلّم معاملات"),
        ]
    )
