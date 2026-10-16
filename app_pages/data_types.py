"""أنواع البيانات ومستويات القياس."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core import ui
from core.anim import play_animation
from core.datasets import messy_customers

ui.page_header(
    "أنواع البيانات ومستويات القياس",
    "قبل أن تعالج أي مشكلة، عليك أن تعرف **نوع** العمود. النوع هو الذي يحدّد: "
    "بماذا تعوّض الفقد؟ كيف ترمّز؟ هل تُقيّس؟ أي مقياس مسافة يصلح؟ أي اختبار "
    "إحصائي؟ خطأ في تحديد النوع = سلسلة أخطاء بعده.",
    icon=":material/category:",
    badges=[
        ":blue-badge[Nominal]",
        ":green-badge[Ordinal]",
        ":orange-badge[Interval]",
        ":violet-badge[Ratio]",
        ":red-badge[dtype]",
    ],
)

t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. مستويات القياس (Levels of measurement)", divider="blue")
    st.markdown(
        "تصنيف *ستيفنز* الكلاسيكي (1946) ما زال الأساس الذي تُبنى عليه كل "
        "قرارات المعالجة:"
    )
    ui.concept_grid(
        [
            (
                "Nominal — اسمي",
                "فئات بلا ترتيب: `city`, `blood_type`, `browser`. \n\n"
                "**مسموح:** العدّ، المنوال (mode)، التكرار. \n"
                "**ممنوع:** المتوسط، الترتيب، الطرح. \n"
                "**الترميز:** One-Hot أو Target، لا Ordinal.",
                "blue",
            ),
            (
                "Ordinal — ترتيبي",
                "فئات **مرتّبة** والمسافات بينها غير معروفة: "
                "`Low < Medium < High`, `Secondary < Bachelor < Master < PhD`. \n\n"
                "**مسموح:** الترتيب، الوسيط، المئينات. \n"
                "**حذر:** المتوسط مُضلّل. \n"
                "**الترميز:** OrdinalEncoder بترتيب تُحدّده **أنت**.",
                "green",
            ),
            (
                "Interval — فتري",
                "رقمي، المسافات متساوية، لكن **الصفر اعتباطي**: درجة الحرارة "
                "بالمئوي، التاريخ. \n\n"
                "**مسموح:** الجمع والطرح والمتوسط. \n"
                "**ممنوع:** النسب («٢٠° ضعف ١٠°» خطأ).",
                "orange",
            ),
            (
                "Ratio — نسبي",
                "رقمي بصفر حقيقي: `income`, `age`, `distance`, `count`. \n\n"
                "**مسموح:** كل شيء بما فيه النسب والتحويل اللوغاريتمي "
                "(شرط أن تكون القيم موجبة).",
                "violet",
            ),
        ],
        n_cols=2,
    )

    st.subheader("٢. التصنيف العملي في التعلّم الآلي", divider="green")
    st.markdown(
        """
| النوع العملي | أمثلة | المعالجة النموذجية |
|---|---|---|
| **Continuous numeric** | الدخل، الطول، الزمن | تعويض بالوسيط، تقييس، معالجة الانحراف |
| **Discrete / count** | عدد الطلبات، عدد الأجهزة | تعويض بالوسيط، أحيانًا `log1p`، نماذج عدّية |
| **Binary** | نعم/لا، 0/1 | لا تقييس، تعويض بالمنوال أو فئة «مجهول» |
| **Nominal categorical** | المدينة، نوع المنتج | One-Hot / Target / Frequency encoding |
| **Ordinal categorical** | المستوى التعليمي، شدّة المرض | Ordinal encoding بترتيب يدوي |
| **Datetime** | تاريخ التسجيل | استخراج سمات (سنة، شهر، يوم الأسبوع، فارق زمني) |
| **Text** | وصف المنتج، التعليقات | TF-IDF / Embeddings |
| **Identifier** | رقم العميل، البريد | :red-badge[يُحذف] أو يُستعمل للربط فقط |
| **Geospatial** | خط الطول والعرض | مسافات، تجميع مكاني |

:red-badge[تحذير] العمود الذي يبدو رقميًا ليس بالضرورة رقميًا:
`postal_code` و `phone` و `customer_id` أرقام **اسمية** — متوسطها بلا معنى،
وتقييسها يضلّل النموذج.
"""
    )

    st.subheader("٣. أنواع pandas (dtypes) ولماذا تهم", divider="orange")
    st.markdown(
        """
| dtype | متى | ملاحظة حرجة |
|---|---|---|
| `int64` / `float64` | أرقام | `int64` **لا يقبل** `NaN`؛ أي فقد يحوّل العمود إلى `float64` |
| `Int64` (بحرف كبير) | أرقام صحيحة مع فقد | نوع nullable يحتفظ بـ `pd.NA` ويبقى صحيحًا |
| `object` | نصوص مختلطة | الأبطأ والأثقل في الذاكرة؛ غالبًا علامة على عمود لم يُنظَّف |
| `string` | نصوص صريحة | أوضح من `object` ويدعم `pd.NA` |
| `category` | فئات محدودة متكرّرة | يوفّر ذاكرة هائلة ويحمل **ترتيبًا** اختياريًا |
| `bool` / `boolean` | ثنائي | `boolean` يقبل الفقد |
| `datetime64[ns]` | تواريخ | يفتح كل عمليات `.dt` |

لماذا هذا مهم عمليًا؟ لأن `df.describe()` و `select_dtypes` و
`ColumnTransformer` كلها تعتمد على الـ dtype. عمود تواريخ مخزّن كنص سيُعامل
كفئة نصّية ذات ألف فئة، وعمود فئوي مخزّن كرقم سيُقيَّس ويُحسب متوسطه.
"""
    )

    ui.beginner_note(
        """
الفكرة في جملة: **الكمبيوتر لا يعرف أن `3` في عمود «المدينة» هي «وهران»،
ولا يعرف أن `"25/12/2023"` تاريخ.** أنت من يخبره — وإن لم تفعل، سيتعامل
معها بالطريقة الخاطئة بصمت، دون أي رسالة خطأ.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. أول نظرة على الأنواع", divider="blue")
    ui.code(
        """
import pandas as pd

df.info(memory_usage="deep")      # الأنواع + الذاكرة الحقيقية
df.dtypes.value_counts()
df.head(10)
df.describe(include="all").T      # رقمي وفئوي معًا
df.nunique().sort_values()        # يكشف الأعمدة الثابتة والمعرّفات
"""
    )

    st.subheader("٢. تصحيح الأنواع", divider="green")
    ui.code(
        '''
# نص يحمل أرقامًا ("25 000 DA") ← رقم
df["income"] = (
    df["income_text"]
    .str.replace(r"[^0-9.\\-]", "", regex=True)
    .replace("", pd.NA)
    .astype("Float64")
)

# تاريخ بصيغة يوم/شهر/سنة
df["signup_date"] = pd.to_datetime(df["signup_date"], format="%d/%m/%Y", errors="coerce")

# فئوي ترتيبي: الترتيب تُحدّده أنت، لا الأبجدية
edu = pd.CategoricalDtype(["Secondary", "Bachelor", "Master", "PhD"], ordered=True)
df["education"] = df["education"].astype(edu)
df["education"] > "Bachelor"      # يعمل الآن لأن النوع مرتّب

# فئوي اسمي: توفير ذاكرة
df["city"] = df["city"].astype("category")

# صحيح مع فقد
df["n_devices"] = df["n_devices"].astype("Int64")
''',
        caption="`errors=\"coerce\"` يحوّل ما يتعذّر تفسيره إلى `NaT`/`NaN` بدل رمي خطأ — "
        "ثم تفحص كم قيمة فشلت: `df['signup_date'].isna().sum()`.",
    )

    st.subheader("٣. فصل الأعمدة آليًا للـ Pipeline", divider="orange")
    ui.code(
        """
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
dt_cols  = df.select_dtypes(include="datetime").columns.tolist()

# أزل ما ليس متغيّرًا تنبؤيًا
drop_like_id = [c for c in df.columns if df[c].nunique() == len(df)]
constant     = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
print("ID-like:", drop_like_id, "| Constant:", constant)
"""
    )

    st.subheader("٤. جرّب بنفسك على بيانات «وسخة»", divider="violet")
    df = messy_customers()
    profile = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
            "pct_missing": (df.isna().mean() * 100).round(1).to_numpy(),
            "sample": [str(df[c].dropna().iloc[0])[:28] if df[c].notna().any() else "—" for c in df.columns],
        }
    )
    st.dataframe(profile, width="stretch", hide_index=True)
    st.caption(
        "لاحظ: `income_text` و `signup_date` مخزّنان كنص، `country` ثابت، "
        "`email` و `customer_id` شبه معرّفات."
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("ماذا يحدث حين نخطئ في النوع؟", divider="violet")

    CASES = [
        (
            "postal_code كـ int64",
            "النموذج يحسب: «الرمز 31000 أكبر من 16000 بمقدار 15000» — "
            "وهذا بلا معنى. المسافات بين الأكواد البريدية لا تمثّل شيئًا.",
            "الصواب: `astype('category')` ثم Target/Frequency encoding.",
            "red",
        ),
        (
            "education كـ One-Hot",
            "فقدت المعلومة الترتيبية: النموذج لا يعرف أن Master > Bachelor. "
            "يحتاج بيانات أكثر ليعيد اكتشاف ما كنت تعرفه أصلًا.",
            "الصواب: `OrdinalEncoder(categories=[['Secondary','Bachelor','Master','PhD']])`.",
            "orange",
        ),
        (
            "city كـ OrdinalEncoder",
            "فرضت ترتيبًا وهميًا: Algiers=0 < Annaba=1 < Oran=2. "
            "النماذج الخطّية ستفترض علاقة خطّية بين «رقم المدينة» والهدف.",
            "الصواب: One-Hot (أو Target encoding عند ارتفاع عدد الفئات).",
            "red",
        ),
        (
            "signup_date كـ object",
            "يُعامَل كنصّ فريد لكل صف ⇒ ألف فئة بلا فائدة، وذاكرة ضائعة، "
            "ومعلومة الزمن (الموسمية، الأقدمية) مهدورة تمامًا.",
            "الصواب: `pd.to_datetime` ثم استخراج `year`, `month`, `dayofweek`, `tenure_days`.",
            "orange",
        ),
        (
            "customer_id كمتغيّر",
            "إن كانت المعرّفات مُولّدة بترتيب زمني، سيلتقط النموذج ارتباطًا "
            "وهميًا بين «رقم العميل» والهدف — وينهار في الإنتاج.",
            "الصواب: احذفه من `X` تمامًا، واحتفظ به للربط فقط.",
            "red",
        ),
        (
            "country الثابت",
            "عمود بقيمة واحدة: تباين صفري ⇒ صفر معلومة، لكنه يضيف عمودًا "
            "بعد الترميز ويبطئ التدريب.",
            "الصواب: احذفه، أو استعمل `VarianceThreshold(threshold=0.0)`.",
            "green",
        ),
    ]

    def _render_case(frame: int) -> None:
        name, problem, fix, color = CASES[frame]
        with st.container(border=True):
            st.markdown(f"##### :{color}-badge[{name}]")
            st.markdown(f"**ماذا يحدث؟** {problem}")
            st.markdown(f"**:green[الصواب:]** {fix}")

    play_animation(
        "types_cases",
        len(CASES),
        _render_case,
        fps=0.55,
        label="الحالة",
        frame_labels=[c[0] for c in CASES],
    )

# ============================================================== القرار
with t_decide:
    st.subheader("النوع ⇐ المعالجة", divider="orange")
    ui.decision_table(
        [
            ("رقمي متّصل بانحراف قوي", "وسيط للتعويض + `log1p` أو Yeo-Johnson + تقييس", "المتوسط والانحراف حسّاسان للذيل الطويل"),
            ("رقمي بعدد قليل من القيم (عدّي)", "وسيط، وربّما تحويله إلى فئات (binning)", "قد تكون العلاقة غير خطّية بين العدد والهدف"),
            ("فئوي اسمي، فئات قليلة (< 15)", "One-Hot + فئة `Missing` صريحة", "لا يفرض ترتيبًا، وعدد الأعمدة مقبول"),
            ("فئوي اسمي، فئات كثيرة (> 50)", "Target / Frequency / Hashing encoding", "One-Hot يفجّر الأبعاد ويضعف النماذج"),
            ("فئوي ترتيبي", "Ordinal بترتيب يدوي صريح", "الترتيب معلومة مجّانية لا تُهدر"),
            ("ثنائي", "0/1 بلا تقييس", "التقييس لا يضيف شيئًا هنا"),
            ("تاريخ", "استخراج سمات + حساب الأقدمية", "القيمة الخام للتاريخ غير قابلة للتعميم على المستقبل"),
            ("نص حر", "TF-IDF أو embeddings في فرع منفصل", "ليس متغيّرًا فئويًا"),
            ("معرّف فريد", "احذفه من المتغيّرات", "خطر حفظ (memorisation) وتسريب"),
            ("عمود ثابت", "احذفه", "تباين صفري = صفر معلومة"),
        ]
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("pandas — Essential basic functionality: dtypes", "https://pandas.pydata.org/docs/user_guide/basics.html#dtypes"),
            ("pandas — Categorical data", "https://pandas.pydata.org/docs/user_guide/categorical.html"),
            ("pandas — Nullable integer data type", "https://pandas.pydata.org/docs/user_guide/integer_na.html"),
            ("pandas — Time series / date functionality", "https://pandas.pydata.org/docs/user_guide/timeseries.html"),
            ("scikit-learn — Encoding categorical features", "https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features"),
            ("Stevens (1946) — On the Theory of Scales of Measurement", "https://www.science.org/doi/10.1126/science.103.2684.677"),
        ]
    )
    ui.terms(
        [
            ("Level of measurement", "مستوى القياس (اسمي/ترتيبي/فتري/نسبي)"),
            ("dtype", "نوع العمود في pandas/NumPy"),
            ("Cardinality", "عدد الفئات المختلفة في عمود فئوي"),
            ("Nullable dtype", "نوع يقبل القيمة المفقودة `pd.NA`"),
            ("Identifier column", "عمود معرّف فريد لا يُستعمل كمتغيّر"),
        ]
    )
