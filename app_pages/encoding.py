"""ترميز المتغيّرات الفئوية."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

from core import charts, ui
from core.anim import play_animation
from core.datasets import clean_customers

ui.page_header(
    "ترميز المتغيّرات الفئوية (Encoding)",
    "النماذج تفهم الأرقام فقط. لكن تحويل `\"Oran\"` إلى رقم ليس عملية "
    "محايدة: كل طريقة ترميز **تزرع افتراضًا** في البيانات — والافتراض "
    "الخاطئ يكلّف دقّة ومعنى.",
    icon=":material/translate:",
    badges=[
        ":blue-badge[One-Hot]",
        ":green-badge[Ordinal]",
        ":orange-badge[Target]",
        ":violet-badge[Frequency]",
        ":red-badge[Leakage risk]",
    ],
)

t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. الطرق الأساسية", divider="blue")
    ui.concept_grid(
        [
            (
                "One-Hot Encoding",
                "عمود ثنائي لكل فئة. **لا يفرض أي ترتيب** ⇒ الخيار الافتراضي "
                "للفئات الاسمية.\n\n"
                ":green[مزايا] محايد، مفهوم، يناسب النماذج الخطّية.\n"
                ":red[عيوب] انفجار الأبعاد مع الفئات الكثيرة؛ أعمدة متناثرة؛ "
                "يُضعف الأشجار (كل تقسيم يرى فئة واحدة فقط).",
                "blue",
            ),
            (
                "Ordinal Encoding",
                "رقم لكل فئة **بترتيب تحدّده أنت**: `Low=0 < Medium=1 < High=2`.\n\n"
                ":green[مزايا] عمود واحد فقط؛ يحفظ معلومة الترتيب.\n"
                ":red[خطر] استعماله على فئات اسمية يزرع ترتيبًا وهميًا يتعلّمه "
                "النموذج الخطّي حرفيًا.",
                "green",
            ),
            (
                "Target / Mean Encoding",
                "استبدال الفئة بـ **متوسط الهدف** فيها (مع تنعيم وتنظيم).\n\n"
                ":green[مزايا] عمود واحد مهما كان عدد الفئات؛ قوي جدًا مع "
                "الكاردينالية العالية.\n"
                ":red[خطر] **تسريب مباشر للهدف** إن لم يُحسب داخل تحقّق "
                "متقاطع — وهو أشهر سبب لنموذج «ممتاز» ينهار في الإنتاج.",
                "orange",
            ),
            (
                "Frequency / Count Encoding",
                "استبدال الفئة بعدد (أو نسبة) ظهورها.\n\n"
                ":green[مزايا] عمود واحد، لا يستعمل الهدف ⇒ لا تسريب.\n"
                ":red[عيوب] فئتان بنفس التكرار تحصلان على نفس القيمة.",
                "violet",
            ),
            (
                "Binary / Hashing Encoding",
                "تمثيل الفئة بأرقام ثنائية، أو تمريرها عبر دالة تجزئة إلى "
                "$k$ عمودًا ثابتًا.\n\n"
                ":green[مزايا] أبعاد مضبوطة مع ملايين الفئات؛ يعمل على التدفّق "
                "(streaming).\n"
                ":red[عيوب] تصادمات (collisions) وفقدان قابلية التفسير.",
                "red",
            ),
            (
                "Embeddings",
                "متّجه كثيف مُتعلَّم لكل فئة (شائع في الشبكات العصبية والأنظمة "
                "التوصوية).\n\n"
                ":green[مزايا] يلتقط تشابهًا بين الفئات.\n"
                ":red[عيوب] يحتاج بيانات كبيرة وتدريبًا إضافيًا.",
                "blue",
            ),
        ],
        n_cols=3,
    )

    st.subheader("٢. `drop='first'` — ومتى تحتاجه", divider="green")
    st.markdown(
        """
مع One-Hot، مجموع الأعمدة الناتجة = 1 دائمًا. فإن كان في النموذج حدّ ثابت
(intercept) نشأ **تعدّد خطّي تام** (dummy variable trap): مصفوفة التصميم
تصبح شاذّة.

- **الانحدار الخطّي بلا تنظيم / الاستدلال الإحصائي:** `drop="first"` ضروري.
- **النماذج المنظّمة (Ridge/Lasso) والأشجار:** غير ضروري، بل قد يكون ضارًّا —
  الحذف يجعل «الفئة المرجعية» غير قابلة للتعبير عنها مباشرة ويشوّه التنظيم.
- `drop="if_binary"` حلّ وسط عملي: يحذف عمودًا واحدًا فقط للمتغيّرات الثنائية.
"""
    )

    st.subheader("٣. المشكلتان اللتان تواجهانك دائمًا", divider="red")
    st.markdown(
        """
**أ) الفئة غير المرئية (unseen category):** مدينة ظهرت في الاختبار ولم
تظهر في التدريب. `OneHotEncoder(handle_unknown="error")` سيتعطّل — وهذا هو
السلوك الافتراضي القديم.

الحلول: `handle_unknown="ignore"` (صفر في كل الأعمدة) أو
`handle_unknown="infrequent_if_exist"` (توجيهها إلى فئة «نادر» المُجمَّعة).

**ب) الفئات النادرة:** فئة بـ 3 مشاهدات لا تعطي تقديرًا مستقرًّا وتزيد
الأبعاد. الحل: `min_frequency` أو `max_categories` — وهما يجمعان النادر في
`infrequent_sklearn` تلقائيًا **بحدود مُتعلَّمة من التدريب**.
"""
    )

    st.subheader("٤. Target Encoding: كيف يُسرّب وكيف نمنعه؟", divider="orange")
    st.markdown(
        """
لو حسبت متوسط الهدف لكل فئة من **كامل** البيانات، فإن قيمة كل صف تحتوي
جزءًا من هدف ذلك الصف نفسه. مع فئة بمشاهدة واحدة، تصبح القيمة **هي الهدف
حرفيًا** ⇒ النموذج يحفظ ولا يتعلّم.

الحلول الثلاثة (كلها مطبّقة داخل `sklearn.preprocessing.TargetEncoder`):
1. **تحقّق متقاطع داخلي (internal CV):** قيمة كل صف تُحسب من طيّات **لا
   تحتويه**.
2. **تنعيم (smoothing):** مزج متوسط الفئة بالمتوسط العام حسب حجم الفئة:
"""
    )
    st.latex(r"\hat{y}_c = \frac{n_c \bar{y}_c + m \bar{y}}{n_c + m}")
    st.markdown(
        "حيث $n_c$ حجم الفئة و $m$ معامل التنعيم. الفئات الصغيرة تنجذب نحو "
        "المتوسط العام فلا تنفجر.\n\n"
        "3. **ضوضاء عشوائية** مضافة أثناء التدريب (أسلوب شائع في المسابقات)."
    )

    ui.beginner_note(
        """
القاعدة البسيطة:
- فئات **مرتّبة** (صغير/متوسط/كبير) ⇒ `OrdinalEncoder` بترتيبك أنت.
- فئات **غير مرتّبة وقليلة** ⇒ `OneHotEncoder`.
- فئات **غير مرتّبة وكثيرة** ⇒ `TargetEncoder` (داخل Pipeline!) أو Frequency.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. One-Hot بشكل آمن", divider="blue")
    ui.code(
        """
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(
    handle_unknown="infrequent_if_exist",  # فئة جديدة ⇒ تُعامَل كنادرة
    min_frequency=0.01,                    # أقل من 1% ⇒ تُجمَّع
    max_categories=20,                     # سقف صارم للأبعاد
    sparse_output=False,                   # أسهل في التصحيح
    drop=None,                             # "first" فقط مع نموذج خطّي غير منظّم
)
X_enc = ohe.fit_transform(X_train[cat_cols])
print(ohe.get_feature_names_out()[:10])
print(ohe.infrequent_categories_)          # ما جُمّع فعلًا
"""
    )

    st.subheader("٢. Ordinal بترتيب صريح", divider="green")
    ui.code(
        """
from sklearn.preprocessing import OrdinalEncoder

edu_order = ["Secondary", "Bachelor", "Master", "PhD"]
oe = OrdinalEncoder(
    categories=[edu_order],                 # الترتيب قرارك، لا الأبجدية
    handle_unknown="use_encoded_value",
    unknown_value=-1,                       # علامة واضحة للفئة غير المرئية
)
X_train[["education"]] = oe.fit_transform(X_train[["education"]])
""",
        caption=":red-badge[خطأ شائع] `OrdinalEncoder()` بلا `categories` يرتّب "
        "أبجديًا: `Bachelor < Master < PhD < Secondary` — ترتيب بلا معنى.",
    )

    st.subheader("٣. Target Encoding بلا تسريب", divider="orange")
    ui.code(
        """
from sklearn.preprocessing import TargetEncoder

te = TargetEncoder(
    smooth="auto",            # تنعيم تلقائي حسب حجم الفئة
    cv=5,                     # تحقّق متقاطع داخلي ⇒ لا تسريب
    target_type="continuous", # صرّح بالنوع: "auto" قد يخطئ مع أهداف عددية صحيحة
    random_state=0,
)

# داخل Pipeline دائمًا
pipe = Pipeline([
    ("enc", ColumnTransformer([("te", te, high_card_cols),
                               ("oh", OneHotEncoder(handle_unknown="ignore"), low_card_cols)])),
    ("model", Ridge()),
])
cross_val_score(pipe, X_train, y_train, cv=5)
""",
        caption="`TargetEncoder` في scikit-learn يستعمل `fit_transform` خاصًّا "
        "يطبّق CV داخليًا على التدريب، بينما `transform` يستعمل التعيينات "
        "الكاملة — وهذا بالضبط ما يمنع التسريب.",
    )

    st.subheader("٤. Frequency و Hashing", divider="violet")
    ui.code(
        """
# Frequency encoding (يدويًا، ويجب تعلّمه من التدريب فقط)
freq = X_train["city"].value_counts(normalize=True)
X_train["city_freq"] = X_train["city"].map(freq)
X_test["city_freq"] = X_test["city"].map(freq).fillna(0)   # الفئة الجديدة ⇒ 0

# Hashing: أبعاد ثابتة مهما بلغ عدد الفئات
from sklearn.feature_extraction import FeatureHasher
fh = FeatureHasher(n_features=32, input_type="string")
X_hashed = fh.transform(X_train["city"].astype(str).map(lambda v: [v]))

# مكتبة متخصّصة بعشرات الطرق
# pip install category_encoders
"""
    )

    st.subheader("٥. الترميز الدوري (Cyclical) للزمن", divider="red")
    ui.code(
        """
import numpy as np

# الساعة 23 والساعة 0 متجاورتان — الترميز الرقمي المباشر يجعلهما أبعد ما يكون
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# بديل من scikit-learn
from sklearn.preprocessing import SplineTransformer
SplineTransformer(n_knots=12, extrapolation="periodic")
"""
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("أ. انفجار الأبعاد مع One-Hot", divider="violet")
    st.markdown(
        "نرفع عدد الفئات في عمود واحد تدريجيًا ونقارن عدد الأعمدة الناتجة: "
        "One-Hot الخام مقابل One-Hot مع `min_frequency` مقابل Target/Frequency."
    )

    CARDS = [3, 5, 10, 25, 60, 150, 400, 1000]

    def _render_card(frame: int) -> None:
        k = CARDS[frame]
        rows = pd.DataFrame(
            {
                "encoder": [
                    "One-Hot (raw)",
                    "One-Hot (max_categories=20)",
                    "Target encoding",
                    "Frequency encoding",
                    "Hashing (n=32)",
                ],
                "n_columns": [k, min(k, 20) + (1 if k > 20 else 0), 1, 1, 32],
            }
        )
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("عدد الفئات في العمود", f"{k:,}")
            st.metric("أعمدة One-Hot الخام", f"{k:,}", delta_color="inverse")
            st.metric("أعمدة Target encoding", "1")
        with c2:
            st.altair_chart(
                charts.bar(rows, "encoder", "n_columns", horizontal=True, height=260),
                width="stretch",
            )
        if k >= 60:
            st.markdown(
                ":red-badge[منطقة الخطر] مع هذا العدد تصبح معظم أعمدة One-Hot "
                "أصفارًا شبه دائمة: تقديرات غير مستقرّة، ذاكرة مهدورة، وأشجار "
                "تضعف لأن كل تقسيم يرى فئة واحدة فقط."
            )
        else:
            st.markdown(":green-badge[مقبول] One-Hot ما زال خيارًا سليمًا هنا.")

    play_animation("card_explosion", len(CARDS), _render_card, fps=0.8, label="عدد الفئات")

    st.subheader("ب. Target Encoding: مع التسريب وبدونه", divider="red")
    st.markdown(
        "نصنع عمودًا فئويًا **عالي الكاردينالية بلا أي علاقة حقيقية بالهدف** "
        "(فئات عشوائية تمامًا). الجواب الصحيح معروف سلفًا: "
        "$R^2 = 0$. فلنرَ ماذا يقول كل أسلوب ترميز."
    )

    @st.cache_data(ttl=1800, max_entries=4)
    def leakage_demo() -> pd.DataFrame:
        rng = np.random.default_rng(0)
        n, n_future = 600, 400
        y = rng.lognormal(10.6, 0.45, n)
        cat = pd.Series([f"cat_{i}" for i in rng.integers(0, 300, n)])
        X = pd.DataFrame({"random_cat": cat})
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

        # بيانات «الغد»: نفس التوزيع، نفس الفئات، أهداف جديدة مستقلّة
        y_fut = rng.lognormal(10.6, 0.45, n_future)
        cat_fut = pd.Series([f"cat_{i}" for i in rng.integers(0, 300, n_future)])
        X_fut = pd.DataFrame({"random_cat": cat_fut})

        # (١) الطريقة الخاطئة: متوسط الهدف محسوب من **كل** البيانات
        full_map = pd.Series(y).groupby(cat.to_numpy()).mean()
        gmean = float(np.mean(y))

        def enc(frame: pd.DataFrame) -> np.ndarray:
            return frame["random_cat"].map(full_map).fillna(gmean).to_numpy().reshape(-1, 1)

        m1 = Ridge().fit(enc(Xtr), ytr)
        leak = {
            "train": m1.score(enc(Xtr), ytr),
            "test": m1.score(enc(Xte), yte),
            "future": m1.score(enc(X_fut), y_fut),
        }

        # (٢) الطريقة الصحيحة: TargetEncoder بتحقّق متقاطع داخلي
        pipe = Pipeline(
            [
                ("te", TargetEncoder(cv=5, random_state=0, target_type="continuous")),
                ("m", Ridge()),
            ]
        )
        pipe.fit(Xtr, ytr)
        ok = {
            "train": pipe.score(Xtr, ytr),
            "test": pipe.score(Xte, yte),
            "future": pipe.score(X_fut, y_fut),
        }

        rows = []
        for method, d in (("تسريب: المتوسط من كل البيانات", leak), ("سليم: TargetEncoder (cv=5)", ok)):
            for split, v in d.items():
                rows.append({"method": method, "split": split, "r2": float(v)})
        return pd.DataFrame(rows)

    res = leakage_demo()
    st.altair_chart(
        alt.Chart(res)
        .mark_bar()
        .encode(
            y=alt.Y("method:N", title=None),
            x=alt.X("r2:Q", title="R²  (الحقيقة = 0 لأن العمود عشوائي)"),
            color=alt.Color(
                "split:N",
                scale=alt.Scale(
                    domain=["train", "test", "future"],
                    range=["#4C6EF5", "#F08C00", "#E8590C"],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            yOffset="split:N",
            tooltip=["method", "split", "r2"],
        )
        .properties(height=240),
        width="stretch",
    )
    st.dataframe(res.assign(r2=res["r2"].round(3)), width="stretch", hide_index=True)
    st.markdown(
        """
**كيف تقرأ هذا الرسم؟**

- :orange-badge[الطريقة المُسرِّبة] تعطي $R^2$ مرتفعًا على التدريب **وعلى
  الاختبار معًا**. لماذا على الاختبار أيضًا؟ لأن خريطة المتوسطات حُسبت من
  كامل البيانات، فصفوف الاختبار شاركت في بناء ترميزها هي نفسها. **اختبارك
  لم يعد اختبارًا.**
- العمود الأحمر (`future`) هو الحقيقة: بيانات جديدة لم تشارك في بناء
  الخريطة. هناك ينهار الأداء إلى الصفر أو دونه — وهذا ما سيحدث في أول يوم
  إنتاج.
- :green-badge[الطريقة السليمة] تعطي ≈ 0 في الحالات الثلاث. تبدو «مخيّبة»،
  لكنها **الإجابة الصحيحة**: العمود عشوائي ولا يحمل أي معلومة.

:red-badge[الدرس] التسريب لا يُكتشف بالنظر إلى الفجوة بين التدريب والاختبار،
لأنه يرفع الاثنين معًا. يُكتشف بـ **مراجعة كيفية حساب كل متغيّر**.
"""
    )

# ============================================================== القرار
with t_decide:
    st.subheader("أي ترميز أستعمل؟", divider="orange")
    ui.decision_table(
        [
            ("فئوي ترتيبي", "`OrdinalEncoder(categories=[...])`", "الترتيب معلومة حقيقية لا تُهدر"),
            ("اسمي، < 15 فئة، نموذج خطّي", "One-Hot (+ `drop='first'`)", "محايد ويتجنّب فخّ المتغيّرات الصورية"),
            ("اسمي، < 15 فئة، نموذج شجري", "One-Hot أو فئوي أصلي (LightGBM/CatBoost)", "الأشجار تتعامل مع الفئات الأصلية بكفاءة أعلى"),
            ("اسمي، 15–50 فئة", "One-Hot + `min_frequency`", "يضبط الأبعاد ويجمع النادر"),
            ("اسمي، > 50 فئة", "`TargetEncoder` أو Frequency", "One-Hot يفجّر الأبعاد ويفكّك الإشارة"),
            ("كاردينالية ضخمة (معرّفات، IP)", "Hashing أو Embeddings", "أبعاد ثابتة؛ وتذكّر أن المعرّف قد لا يكون متغيّرًا أصلًا"),
            ("فئة قد تظهر لأول مرّة في الإنتاج", "`handle_unknown='infrequent_if_exist'`", "يمنع التعطّل ويعامل الجديد كنادر"),
            ("فئات نادرة جدًا", "`min_frequency` أو تجميع يدوي في `Other`", "تقديرات الفئات الصغيرة غير مستقرّة"),
            ("متغيّر دوري (ساعة، شهر)", "`sin/cos` أو SplineTransformer دوري", "يحفظ تجاور 23 و 0"),
            ("فقد في عمود فئوي", "فئة `Missing` صريحة ثم ترميز", "واقعة الفقد معلومة تنبؤية"),
        ],
        headers=("الحالة", "الترميز", "السبب"),
    )

    ui.danger(
        "**أي ترميز يستعمل `y` (Target، WOE، Leave-one-out) يجب أن يكون داخل "
        "`Pipeline` ومع تحقّق متقاطع داخلي.** حسابه على كامل البيانات هو "
        "أسرع طريق إلى نموذج يبدو مثاليًا وينهار في أول يوم إنتاج."
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("scikit-learn — Encoding categorical features", "https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features"),
            ("scikit-learn — TargetEncoder", "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html"),
            ("scikit-learn — Target Encoder's Internal Cross fitting", "https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html"),
            ("scikit-learn — Comparing Target Encoder with Other Encoders", "https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder.html"),
            ("scikit-learn — Time-related feature engineering (cyclical)", "https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html"),
            ("category_encoders — مكتبة ترميز شاملة", "https://contrib.scikit-learn.org/category_encoders/"),
            ("Micci-Barreca (2001) — A Preprocessing Scheme for High-Cardinality Categorical Attributes", "https://doi.org/10.1145/507533.507538"),
        ]
    )
    ui.terms(
        [
            ("One-Hot encoding", "ترميز ثنائي لكل فئة"),
            ("Ordinal encoding", "ترميز ترتيبي"),
            ("Target / mean encoding", "ترميز بمتوسط الهدف"),
            ("Smoothing", "تنعيم التقدير نحو المتوسط العام"),
            ("High cardinality", "كاردينالية عالية (فئات كثيرة)"),
            ("Dummy variable trap", "فخّ المتغيّرات الصورية"),
            ("Unseen category", "فئة لم تظهر في التدريب"),
            ("Cyclical encoding", "ترميز دوري"),
        ]
    )
