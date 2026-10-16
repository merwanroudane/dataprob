"""خريطة الطريق الكاملة لتحضير البيانات."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from core import ui
from core.anim import play_animation

ui.page_header(
    "خريطة الطريق الكاملة",
    "قبل أي تفصيل: ما هو **الترتيب الصحيح** للعمليات، ولماذا تغيير الترتيب "
    "وحده كفيل بإفساد نتائجك حتى لو كان كل كود كتبته صحيحًا؟",
    icon=":material/map:",
    badges=[
        ":blue-badge[Workflow]",
        ":green-badge[fit / transform]",
        ":red-badge[Data leakage]",
        ":violet-badge[Pipeline]",
    ],
)

t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. المشكلة التي يحلّها الترتيب", divider="blue")
    st.markdown(
        """
تخيّل أنك تريد قياس قدرة طالب على اجتياز امتحان. إذا أطلعته على ورقة
الامتحان أثناء المراجعة، ستكون نتيجته ممتازة — لكنها **لا تقيس شيئًا**.
هذا بالضبط ما يحدث حين تحسب متوسطًا أو انحرافًا معياريًا أو قائمة فئات
من **كامل** البيانات ثم تقسّمها إلى تدريب واختبار: جزء من معلومات
الاختبار تسرّب إلى التدريب.

المصطلح: **Data leakage** (تسريب البيانات). وهو السبب الأول لانهيار
النماذج التي «كانت ممتازة» في الورقة البحثية أو في الـ notebook.
"""
    )

    ui.beginner_note(
        """
- **`fit`** = «تعلّم» المعاملات (مثلاً: المتوسط والانحراف المعياري لعمود).
- **`transform`** = «طبّق» هذه المعاملات على بيانات.
- القاعدة: `fit` على **التدريب فقط**، ثم `transform` على التدريب والاختبار
  والبيانات الجديدة.
- `fit_transform` تجمع الاثنين — ولذلك لا تُستعمل أبدًا على بيانات الاختبار.
"""
    )

    st.subheader("٢. الخريطة الكاملة", divider="green")
    st.markdown(
        """
```mermaid
flowchart TD
    A[Raw data] --> B[Understand: schema, units, target]
    B --> C{Is the row/feature even valid?}
    C -->|No| D[Structural cleaning: duplicates, types, encodings of NA]
    C -->|Yes| D
    D --> E[TRAIN / TEST SPLIT]
    E --> F[Train set only]
    E --> G[Test set - locked away]
    F --> H[Missing values: diagnose and impute]
    H --> I[Outliers: detect, decide, treat]
    I --> J[Transform: scale, skew, encode]
    J --> K[Feature engineering]
    K --> L[Feature selection]
    L --> M[Resampling for imbalance - train folds only]
    M --> N[Model fit]
    N --> O[Apply SAME fitted steps to test]
    G --> O
    O --> P[Honest evaluation]
```
"""
    )
    st.caption(
        "لاحظ أن الاختبار (Test set) لا يلمسه شيء حتى اللحظة الأخيرة — "
        "ويُعامل بنفس المعاملات المُتعلَّمة من التدريب، لا بمعاملاته الخاصة."
    )

    st.subheader("٣. ما الذي يُنظَّف قبل التقسيم؟", divider="orange")
    st.markdown(
        """
ليس كل شيء يأتي بعد التقسيم. التمييز الحاسم:

- **تنظيف بنيوي (structural / deterministic)** — لا يتعلّم أي معامل من
  البيانات: حذف صفوف مكرّرة حرفيًا، تصحيح نوع العمود من نص إلى رقم،
  تحويل `"N/A"` و `-99` إلى `NaN`، توحيد `"ALGIERS"` و `"algiers"`،
  تصحيح وحدات القياس. **هذا يسبق التقسيم بأمان.**
- **تنظيف إحصائي (statistical / learned)** — يستخرج رقمًا من البيانات:
  المتوسط للتعويض، الحدود للشواذ، معاملات التقييس، قائمة الفئات، أهمية
  المتغيّرات. **هذا يأتي بعد التقسيم، ويُحسب من التدريب فقط.**

قاعدة الفصل في سؤال واحد: *«هل هذه الخطوة تحتاج أن تنظر إلى بقيّة الصفوف
لتقرّر ماذا تفعل بهذا الصف؟»* — إن كان الجواب نعم، فهي خطوة مُتعلَّمة ويجب
أن تكون داخل `Pipeline` بعد التقسيم.
"""
    )

    st.subheader("٤. أين يقع الخطأ عمليًا؟", divider="red")
    ui.concept_grid(
        [
            (
                "التقييس قبل التقسيم",
                "`StandardScaler().fit_transform(X)` ثم التقسيم. المتوسط حُسب من "
                "صفوف الاختبار أيضًا ⇒ تسريب **خفيف لكنه منهجي**.",
                "red",
            ),
            (
                "التعويض قبل التقسيم",
                "`df.fillna(df.mean())` على الجدول كاملًا. نفس المشكلة، وأثرها "
                "أكبر كلما ارتفعت نسبة الفقد.",
                "red",
            ),
            (
                "اختيار المتغيّرات قبل التقسيم",
                "حساب الارتباط مع الهدف على كامل البيانات ثم الاحتفاظ بأفضل k. "
                "هذا **أخطر أنواع التسريب** وقد يرفع الدقّة عشرات النقاط كذبًا.",
                "red",
            ),
            (
                "SMOTE قبل التقسيم",
                "توليد عيّنات اصطناعية ثم التقسيم ⇒ نسخ شبه مطابقة من نفس الصف "
                "تقع في التدريب والاختبار معًا.",
                "red",
            ),
            (
                "حذف الشواذ من الاختبار",
                "الاختبار يجب أن يمثّل الواقع كما هو. حذف شواذه = اختبار على "
                "عالم وردي غير موجود.",
                "orange",
            ),
            (
                "تنظيف بنيوي قبل التقسيم",
                "حذف التكرار الحرفي وتصحيح الأنواع ورموز الفقد — :green-badge[هذا "
                "صحيح ومطلوب] لأنه لا يتعلّم أي معامل.",
                "green",
            ),
        ],
        n_cols=3,
    )

    ui.takeaways(
        [
            "الترتيب جزء من المنهجية، لا تفصيل تقني.",
            "كل خطوة «تتعلّم» رقمًا من البيانات تأتي **بعد** التقسيم.",
            "`Pipeline` ليس رفاهية تنظيمية: هو الضمان الآلي لعدم التسريب.",
            "الاختبار يُفتح مرّة واحدة في نهاية العمل.",
        ]
    )

# ============================================================== الكود
with t_code:
    st.subheader("الهيكل العام الذي ستبني عليه كل شيء", divider="blue")
    ui.code(
        '''
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------ 1) تنظيف بنيوي (قبل التقسيم)
df = pd.read_csv("data.csv")
df = df.drop_duplicates()                       # تكرار حرفي
df["income"] = pd.to_numeric(df["income"], errors="coerce")   # نص ← رقم
df = df.replace({-99: pd.NA, "N/A": pd.NA, "": pd.NA})        # رموز فقد مخفيّة
df["city"] = df["city"].str.strip().str.title()               # توحيد الفئات

y = df["churn"]
X = df.drop(columns=["churn", "customer_id"])   # المعرّف ليس متغيّرًا تنبؤيًا

# ------------------------------------------------------- 2) التقسيم (مبكّرًا!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------- 3) كل ما يُتعلَّم يوضع داخل Pipeline
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric = Pipeline([
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("scale", StandardScaler()),
])

categorical = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                             min_frequency=0.01, sparse_output=False)),
])

pre = ColumnTransformer([
    ("num", numeric, num_cols),
    ("cat", categorical, cat_cols),
], remainder="drop")

model = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
])

# ------------------------------------------------ 4) تقييم نزيه ثم تدريب نهائي
cv = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
print(f"CV ROC-AUC = {cv.mean():.3f} ± {cv.std():.3f}")

model.fit(X_train, y_train)                 # fit على التدريب فقط
print("Test:", model.score(X_test, y_test)) # الاختبار يُفتح مرّة واحدة
''',
        caption="هذا الهيكل هو العمود الفقري لكل ما سيأتي في بقيّة الدروس.",
    )

    st.subheader("الخطأ مقابل الصواب في سطرين", divider="red")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("###### :red-badge[خطأ — تسريب]")
        ui.code(
            """
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)      # رأى الاختبار!
X_tr, X_te = train_test_split(X_scaled)
"""
        )
    with c2:
        st.markdown("###### :green-badge[صواب]")
        ui.code(
            """
X_tr, X_te = train_test_split(X)
scaler = StandardScaler().fit(X_tr)     # تعلّم من التدريب
X_tr = scaler.transform(X_tr)
X_te = scaler.transform(X_te)           # طبّق فقط
"""
        )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("شاهد: أين تذهب المعلومة في كل خطوة؟", divider="violet")
    st.markdown(
        "المستطيل الأزرق = بيانات التدريب، الأحمر = بيانات الاختبار. "
        "في كل إطار نُظهر أي جزء **تقرأه** الخطوة الحالية."
    )

    STEPS = [
        ("Raw data", "both", "البيانات الخام كما وصلت. لا شيء مُقسّم بعد."),
        ("Structural cleaning", "both", "تكرار، أنواع، رموز فقد — آمن على الكل لأنه لا يتعلّم شيئًا."),
        ("TRAIN / TEST SPLIT", "split", "اللحظة الحاسمة: من الآن فصاعدًا الاختبار مُقفل."),
        ("Impute (fit)", "train", "المتوسط/الوسيط يُحسب من التدريب فقط."),
        ("Outliers (fit)", "train", "حدود القطع تُقدّر من التدريب فقط."),
        ("Scale + encode (fit)", "train", "المتوسطات والفئات تُتعلَّم من التدريب فقط."),
        ("Feature selection (fit)", "train", "أهمّية المتغيّرات تُقاس على التدريب فقط."),
        ("Model fit", "train", "التدريب الفعلي."),
        ("transform(test)", "test", "الاختبار يُمرّر عبر نفس المعاملات المُتعلَّمة."),
        ("Evaluate", "test", "تقييم نزيه — وهذه هي النتيجة التي تُنشر."),
    ]

    def _render_split(frame: int) -> None:
        name, scope, note = STEPS[frame]
        blocks = []
        for i in range(10):
            part = "test" if i >= 8 else "train"
            if scope == "both":
                active = True
            elif scope == "split":
                active = True
            else:
                active = part == scope
            blocks.append(
                {
                    "i": i,
                    "part": "Test" if part == "test" else "Train",
                    "state": "تُقرأ الآن" if active else "غير مستعملة",
                    "opacity": 1.0 if active else 0.18,
                }
            )
        d = pd.DataFrame(blocks)
        chart = (
            alt.Chart(d)
            .mark_rect(stroke="white", strokeWidth=3, cornerRadius=6)
            .encode(
                x=alt.X("i:O", title="rows (80% train / 20% test)", axis=None),
                color=alt.Color(
                    "part:N",
                    scale=alt.Scale(domain=["Train", "Test"], range=["#4C6EF5", "#E8590C"]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                opacity=alt.Opacity("opacity:Q", legend=None, scale=alt.Scale(domain=[0, 1])),
                tooltip=["part:N", "state:N"],
            )
            .properties(height=90)
        )
        st.altair_chart(chart, width="stretch")
        color = {"both": "blue", "split": "violet", "train": "blue", "test": "orange"}[scope]
        st.markdown(f"**:{color}-badge[{name}]** — {note}")

    play_animation(
        "roadmap_flow",
        len(STEPS),
        _render_split,
        fps=0.8,
        label="الخطوة",
        frame_labels=[s[0] for s in STEPS],
    )

# ============================================================== القرار
with t_decide:
    st.subheader("هل هذه الخطوة قبل التقسيم أم بعده؟", divider="orange")
    ui.decision_table(
        [
            ("حذف صفوف مكرّرة حرفيًا", ":green-badge[قبل التقسيم]", "عملية حتمية، لا تتعلّم معاملًا — وتركها يسبّب تسريبًا بالتكرار"),
            ("تحويل نوع العمود (نص ← رقم/تاريخ)", ":green-badge[قبل التقسيم]", "تصحيح بنيوي بحت"),
            ("استبدال رموز الفقد (-99، «N/A»)", ":green-badge[قبل التقسيم]", "تصحيح ترميز، لا استدلال"),
            ("توحيد كتابة الفئات (case/trim)", ":green-badge[قبل التقسيم]", "قاعدة نصّية ثابتة"),
            ("حذف أعمدة بلا معنى (ID، ثابتة)", ":green-badge[قبل التقسيم]", "قرار قائم على المعرفة لا على إحصاء"),
            ("التعويض بالمتوسط/الوسيط/KNN", ":red-badge[بعد التقسيم]", "يتعلّم قيمة من توزيع البيانات"),
            ("كشف/قص الشواذ بحدود إحصائية", ":red-badge[بعد التقسيم]", "الحدود نفسها معاملات مُتعلَّمة"),
            ("StandardScaler / MinMaxScaler", ":red-badge[بعد التقسيم]", "المتوسط والانحراف معاملات"),
            ("OneHot / Target encoding", ":red-badge[بعد التقسيم]", "قائمة الفئات ومتوسطات الهدف مُتعلَّمة"),
            ("PCA / اختيار المتغيّرات", ":red-badge[بعد التقسيم]", "تعتمد على مصفوفة التباين أو على الهدف"),
            ("SMOTE / إعادة المعاينة", ":red-badge[بعد التقسيم، وداخل الطيّات فقط]", "وإلا تتسرّب نسخ مولّدة إلى التحقّق"),
        ]
    )

    ui.danger(
        "اختبار ذاتي سريع: لو وصلك **صف واحد جديد غدًا**، هل تستطيع تنفيذ هذه "
        "الخطوة عليه وحده؟ إن كنت تحتاج بقيّة الصفوف ⇒ الخطوة مُتعلَّمة ومكانها "
        "داخل `Pipeline`."
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("scikit-learn — Common pitfalls and recommended practices (Data leakage)", "https://scikit-learn.org/stable/common_pitfalls.html"),
            ("scikit-learn — Pipelines and composite estimators", "https://scikit-learn.org/stable/modules/compose.html"),
            ("scikit-learn — Dataset transformations (User Guide)", "https://scikit-learn.org/stable/data_transforms.html"),
            ("scikit-learn — Cross-validation: evaluating estimator performance", "https://scikit-learn.org/stable/modules/cross_validation.html"),
            ("pandas — Working with missing data", "https://pandas.pydata.org/docs/user_guide/missing_data.html"),
            ("Kaufman, Rosset & Perlich (2012) — Leakage in Data Mining", "https://dl.acm.org/doi/10.1145/2020408.2020496"),
        ]
    )
    ui.terms(
        [
            ("Data leakage", "تسرّب معلومة من الاختبار (أو من المستقبل) إلى التدريب"),
            ("fit / transform", "تعلّم المعاملات / تطبيقها"),
            ("Pipeline", "سلسلة خطوات تُعامَل كنموذج واحد"),
            ("ColumnTransformer", "تطبيق معالجات مختلفة على مجموعات أعمدة مختلفة"),
            ("Train / Test split", "تقسيم البيانات إلى تدريب واختبار"),
            ("Cross-validation", "تقييم بتقسيمات متعدّدة متداخلة"),
        ]
    )
