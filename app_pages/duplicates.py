"""القيم المكرّرة: الأنواع، الخطر، والقرار."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from core import charts, ui
from core.anim import play_animation
from core.datasets import clean_customers, messy_customers

ui.page_header(
    "القيم المكرّرة (Duplicates)",
    "التكرار ليس مجرّد «صفوف زائدة». إنه **أخطر مصدر تسريب** في التعلّم "
    "الآلي: صفّ موجود في التدريب والاختبار معًا يجعل النموذج يبدو عبقريًا "
    "بينما هو يحفظ فقط.",
    icon=":material/content_copy:",
    badges=[
        ":blue-badge[Exact]",
        ":green-badge[Key-based]",
        ":orange-badge[Fuzzy]",
        ":red-badge[Leakage]",
        ":violet-badge[Legitimate repeats]",
    ],
)

t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. ستة أنواع من التكرار", divider="blue")
    ui.concept_grid(
        [
            (
                "١. تكرار تام (Exact)",
                "صفّان متطابقان في **كل** عمود. سببه غالبًا: استيراد مكرّر، "
                "`JOIN` خاطئ، أو تشغيل سكربت الجمع مرّتين.\n\n"
                "`df.duplicated()` يكشفه مباشرة.",
                "blue",
            ),
            (
                "٢. تكرار بالمفتاح (Key-based)",
                "نفس `customer_id` في صفّين بقيم مختلفة قليلًا (تحديث السجل). "
                "الجدول يفترض أن المفتاح فريد — وهو ليس كذلك.\n\n"
                "`df.duplicated(subset=['customer_id'])`",
                "green",
            ),
            (
                "٣. تكرار جزئي / غامض (Fuzzy)",
                "«محمد العربي» و «Mohamed El Arbi»، أو فرق حرف في البريد. "
                "نفس الكيان، كتابتان مختلفتان.\n\n"
                "يحتاج تطابقًا نصّيًا تقريبيًا (`rapidfuzz`) أو ربط سجلات "
                "(`recordlinkage`).",
                "orange",
            ),
            (
                "٤. تكرار بنيوي (Structural)",
                "نفس المعلومة بتمثيل مختلف: `1,5 كغ` و `1500 غ`؛ "
                "`2023-01-05` و `05/01/2023`.\n\n"
                "يُكشف بعد توحيد الصيغ والوحدات، لا قبله.",
                "violet",
            ),
            (
                "٥. أعمدة مكرّرة",
                "عمودان بنفس القيم تمامًا (`price` و `price_eur` بعد تحويل "
                "خاطئ)، أو ارتباط = 1.0.\n\n"
                "يسبّب **تعدّد خطّي تام** يكسر الانحدار الخطّي.",
                "red",
            ),
            (
                "٦. تكرار مشروع (Legitimate)",
                "**ليس خطأ**: معاملتان بنفس المبلغ في نفس اليوم؛ بيانات بانل "
                "(نفس الفرد عبر سنوات)؛ قياسات متكرّرة في تجربة.\n\n"
                ":red-badge[لا تحذفها] — ستفقد معلومة حقيقية.",
                "green",
            ),
        ],
        n_cols=3,
    )

    st.subheader("٢. لماذا التكرار خطر بالتحديد؟", divider="red")
    st.markdown(
        """
1. **تسريب عبر التقسيم:** إن كان الصف `A` موجودًا مرّتين وقسّمت عشوائيًا،
   فقد تقع نسخة في التدريب وأخرى في الاختبار. النموذج «رأى» الإجابة.
   النتيجة: دقّة اختبار متفائلة كذبًا — وهذا يحدث **صامتًا**.
2. **ترجيح غير مقصود:** الصف المكرّر 5 مرّات يساهم في دالة الخسارة 5 أضعاف،
   فيميل النموذج نحوه كأنه أهم.
3. **كسر فرضية الاستقلال:** كل الاختبارات الإحصائية (قيم p، فترات الثقة)
   تفترض مشاهدات مستقلة. التكرار يضخّم حجم العيّنة الظاهري فيضيّق فترات
   الثقة كذبًا.
4. **تشويه الإحصاءات الوصفية:** المتوسط والتباين والارتباطات كلّها منحازة
   نحو الصفوف المكرّرة.
5. **تسريب في التجميع (clustering):** عناقيد وهمية كثيفة حول النقاط المكرّرة.
"""
    )

    ui.danger(
        "القاعدة: **احذف التكرار التام قبل التقسيم.** هذه إحدى الحالات القليلة "
        "التي يكون فيها التنظيف قبل التقسيم صحيحًا — لأن الحذف عملية حتمية "
        "لا تتعلّم أي معامل، ولأن تأجيله يعني تسريبًا مؤكّدًا."
    )

    st.subheader("٣. متى يكون «التكرار» معلومة؟", divider="green")
    st.markdown(
        """
- **بيانات المعاملات:** عميل اشترى نفس المنتج بنفس السعر مرّتين — حدثان
  منفصلان. الحل: أضف `transaction_id` و `timestamp` قبل أن تقرّر.
- **بيانات البانل (Panel / longitudinal):** نفس الفرد عبر سنوات. المفتاح
  مركّب `(id, year)` لا `id` وحده.
- **القياسات المتكرّرة:** ثلاث قراءات لنفس العيّنة في المخبر — تنوّعها هو
  تقدير خطأ القياس.
- **بعد SMOTE أو oversampling:** التكرار مقصود، لكن **يجب** أن يبقى داخل
  طيّات التدريب فقط.
"""
    )

    ui.beginner_note(
        """
تخيّل امتحانًا فيه ١٠٠ سؤال، لكن ٣٠ سؤالًا منها مكرّرة من ورقة المراجعة
التي حفظها الطالب. درجته ستكون عالية، لكنها **لا تقيس فهمه**. حذف التكرار
قبل تقسيم «المراجعة» و«الامتحان» هو ما يجعل الدرجة صادقة.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. الكشف الأساسي", divider="blue")
    ui.code(
        """
df.duplicated().sum()                       # عدد الصفوف المكرّرة تمامًا
df[df.duplicated(keep=False)].sort_values(list(df.columns))   # اعرضها كلها

# تكرار بمفتاح العمل (business key)
key = ["first_name", "last_name", "birth_date"]
dups = df[df.duplicated(subset=key, keep=False)].sort_values(key)
print(f"{df.duplicated(subset=key).sum()} سجلًا مكرّرًا بالمفتاح")

# أي الأعمدة تختلف بين النسخ؟
grp = df[df.duplicated(subset=key, keep=False)].groupby(key)
print(grp.nunique().max())      # الأعمدة بقيمة > 1 هي مصدر الاختلاف
"""
    )

    st.subheader("٢. الحذف الذكي: أي نسخة نُبقي؟", divider="green")
    ui.code(
        """
# الأحدث: رتّب أولًا ثم أبقِ الأخير
df = df.sort_values("updated_at").drop_duplicates(subset=["customer_id"], keep="last")

# الأكمل: أبقِ الصف الأقل فقدًا
df["_n_missing"] = df.isna().sum(axis=1)
df = (df.sort_values("_n_missing")
        .drop_duplicates(subset=["customer_id"], keep="first")
        .drop(columns="_n_missing"))

# الدمج بدل الحذف: اجمع أفضل ما في كل نسخة
merged = df.groupby("customer_id").agg(
    name=("name", "first"),
    income=("income", "max"),          # أو median
    last_seen=("updated_at", "max"),
    n_records=("customer_id", "size"), # عدد النسخ = متغيّر مفيد أحيانًا
).reset_index()
""",
        caption="`keep='first'` بدون `sort_values` يعني «أبقِ ما صادفته أولًا» — "
        "قرار عشوائي لا يُدافَع عنه. رتّب دائمًا أولًا.",
    )

    st.subheader("٣. التكرار الغامض (Fuzzy matching)", divider="orange")
    ui.code(
        """
# pip install rapidfuzz
from rapidfuzz import process, fuzz

names = df["company"].dropna().unique().tolist()
for i, n in enumerate(names):
    close = process.extract(n, names[i+1:], scorer=fuzz.token_sort_ratio, limit=3)
    for match, score, _ in close:
        if score > 92:
            print(f"محتمل تطابق: {n!r} ≈ {match!r}  ({score})")

# توحيد نصّي قبل أي مقارنة
import unicodedata
def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()

df["_key"] = df["name"].map(norm) + "|" + df["birth_date"].astype(str)
df.duplicated(subset="_key").sum()
""",
        caption="لربط سجلات على نطاق واسع: مكتبة `recordlinkage` أو "
        "`dedupe` (تعلّم نشط).",
    )

    st.subheader("٤. الأعمدة المكرّرة والمتطابقة", divider="violet")
    ui.code(
        """
# أعمدة متطابقة تمامًا
dupe_cols = df.T.duplicated()
print("أعمدة مكرّرة:", df.columns[dupe_cols].tolist())

# أعمدة بارتباط ≈ 1 (تعدّد خطّي تام)
corr = df.select_dtypes("number").corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
print([(a, b) for a in upper.index for b in upper.columns if upper.loc[a, b] > 0.99])
"""
    )

    st.subheader("٥. على بياناتنا", divider="blue")
    dfm = messy_customers()
    exact = int(dfm.duplicated().sum())
    by_id = int(dfm.duplicated(subset=["customer_id"]).sum())
    near_key = ["age", "education", "city", "signup_date", "email"]
    near = int(dfm.duplicated(subset=near_key).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("إجمالي الصفوف", f"{len(dfm):,}")
    c2.metric("تكرار تام", exact)
    c3.metric("تكرار بـ customer_id", by_id)
    c4.metric("تكرار بمفتاح مركّب", near, help=" + ".join(near_key))
    st.dataframe(
        dfm[dfm.duplicated(keep=False)].sort_values("customer_id").head(12),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "لاحظ الفرق بين الأرقام الثلاثة: كل تعريف للمفتاح يعطي عددًا مختلفًا "
        "من «المكرّرات». **تعريف المفتاح قرار يخصّ المجال، لا الكود.**"
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("كيف يصنع التكرار دقّة وهمية؟", divider="violet")
    st.markdown(
        "تجربة: نأخذ بيانات نظيفة، ننسخ نسبة متزايدة من صفوفها، ثم نقسّم "
        "**عشوائيًا** ونقيس دقّة نموذج 1-NN على الاختبار. "
        "قارن: التقسيم بعد التكرار (الخطأ الشائع) مقابل الحذف قبل التقسيم."
    )

    RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]

    @st.cache_data(ttl=1800, max_entries=8)
    def leak_curve() -> pd.DataFrame:
        base = clean_customers(600)
        X0 = base[["age", "experience_years", "income", "spending", "satisfaction"]]
        y0 = base["churn"]
        rows = []
        for r in RATES:
            rng = np.random.default_rng(int(r * 100) + 1)
            k = int(len(base) * r)
            idx = rng.choice(len(base), k, replace=True) if k else np.array([], dtype=int)
            X = pd.concat([X0, X0.iloc[idx]], ignore_index=True)
            y = pd.concat([y0, y0.iloc[idx]], ignore_index=True)

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
            naive = KNeighborsClassifier(n_neighbors=1).fit(Xtr, ytr).score(Xte, yte)

            dd = pd.concat([X, y.rename("y")], axis=1).drop_duplicates()
            Xd, yd = dd.drop(columns="y"), dd["y"]
            Xtr2, Xte2, ytr2, yte2 = train_test_split(Xd, yd, test_size=0.3, random_state=0, stratify=yd)
            honest = KNeighborsClassifier(n_neighbors=1).fit(Xtr2, ytr2).score(Xte2, yte2)

            rows.append(
                {
                    "dup_rate": round(r * 100),
                    "with duplicates (wrong)": naive,
                    "deduplicated (honest)": honest,
                    "inflation": naive - honest,
                    "n_rows": len(X),
                }
            )
        return pd.DataFrame(rows)

    curve = leak_curve()

    def _render_leak(frame: int) -> None:
        row = curve.iloc[frame]
        k1, k2, k3 = st.columns(3)
        k1.metric("نسبة الصفوف المكرّرة", f"{row['dup_rate']:.0f}%", help=f"{int(row['n_rows'])} صفًا")
        k2.metric("الدقّة مع التكرار", f"{row['with duplicates (wrong)']:.3f}")
        k3.metric(
            "الدقّة بعد حذف التكرار",
            f"{row['deduplicated (honest)']:.3f}",
            delta=f"{-row['inflation']:+.3f}",
            delta_color="inverse",
        )

        sofar = curve.iloc[: frame + 1].melt(
            id_vars="dup_rate",
            value_vars=["with duplicates (wrong)", "deduplicated (honest)"],
            var_name="setup",
            value_name="test_accuracy",
        )
        ch = (
            alt.Chart(sofar)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("dup_rate:Q", title="% duplicated rows", scale=alt.Scale(domain=[0, 72])),
                y=alt.Y("test_accuracy:Q", title="test accuracy (1-NN)", scale=alt.Scale(domain=[0.5, 1.0])),
                color=alt.Color(
                    "setup:N",
                    scale=alt.Scale(
                        domain=["with duplicates (wrong)", "deduplicated (honest)"],
                        range=["#E8590C", "#0CA678"],
                    ),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=["dup_rate", "setup", "test_accuracy"],
            )
            .properties(height=330)
        )
        st.altair_chart(ch, width="stretch")

        if row["inflation"] > 0.05:
            st.markdown(
                f":red-badge[تضخّم {row['inflation']:.1%}] النموذج لم يتعلّم شيئًا "
                "إضافيًا — إنه ببساطة يرى في الاختبار صفوفًا حفظها حرفيًا في "
                "التدريب. هذا هو التسريب بالتكرار."
            )
        else:
            st.markdown(":green-badge[لا تضخّم يُذكر] الفارق ضمن الضجيج العشوائي.")

    play_animation("dup_leak", len(RATES), _render_leak, fps=0.7, label="نسبة التكرار")

    st.caption(
        "اخترنا 1-NN لأنه يُظهر الأثر بأوضح صورة (يحفظ حرفيًا). الأثر نفسه "
        "موجود — بدرجة أقل — في كل النماذج تقريبًا."
    )

# ============================================================== القرار
with t_decide:
    st.subheader("أحذف أم أبقي؟", divider="orange")
    ui.decision_table(
        [
            ("تكرار تام في جدول كيانات (عملاء، منتجات)", ":green-badge[احذف قبل التقسيم]", "خطأ استيراد مؤكّد + خطر تسريب"),
            ("تكرار تام في جدول معاملات", ":orange-badge[تحقّق أولًا]", "قد تكون معاملتان حقيقيتان متطابقتين"),
            ("تكرار بالمفتاح مع قيم مختلفة", ":blue-badge[رتّب ثم أبقِ الأحدث/الأكمل]", "تحديث سجل، لا تكرار عشوائي"),
            ("تكرار غامض (أسماء متقاربة)", ":violet-badge[راجع يدويًا عيّنة أولًا]", "التطابق التقريبي يُنتج إيجابيات كاذبة"),
            ("بيانات بانل / قياسات متكرّرة", ":red-badge[لا تحذف]", "المفتاح مركّب؛ التكرار الظاهري هو البنية نفسها"),
            ("بعد SMOTE / oversampling", ":red-badge[لا تحذف]", "مقصود، لكن يجب أن يبقى داخل طيّات التدريب"),
            ("أعمدة متطابقة", ":green-badge[احذف إحداها]", "تعدّد خطّي تام يكسر الانحدار"),
            ("تكرار بعد JOIN", ":orange-badge[أصلح الـ JOIN نفسه]", "الحذف يخفي خطأ في منطق الدمج، لا يحلّه"),
            ("لا تعرف إن كان مشروعًا", ":blue-badge[أضف عمود `n_duplicates` واحتفظ]", "يحتفظ بالمعلومة ويتيح القياس لاحقًا"),
        ]
    )

    ui.tip(
        "قبل `drop_duplicates()`: اسأل **«ما هو المفتاح الطبيعي لهذا الجدول؟»** "
        "صفّ واحد يمثّل ماذا بالضبط: عميلًا؟ معاملة؟ عميلًا في شهر؟ "
        "الجواب يحدّد `subset` — وبدونه الحذف قرار عشوائي."
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("pandas — DataFrame.duplicated", "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.duplicated.html"),
            ("pandas — DataFrame.drop_duplicates", "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html"),
            ("RapidFuzz — fast fuzzy string matching", "https://rapidfuzz.github.io/RapidFuzz/"),
            ("recordlinkage — Python record linkage toolkit", "https://recordlinkage.readthedocs.io/"),
            ("scikit-learn — Common pitfalls: data leakage", "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage"),
            ("scikit-learn — GroupKFold (منع تسريب المجموعات)", "https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold"),
        ]
    )
    ui.terms(
        [
            ("Exact duplicate", "تكرار تام"),
            ("Near-duplicate / fuzzy", "تكرار تقريبي"),
            ("Business key", "المفتاح الطبيعي للجدول"),
            ("Record linkage", "ربط السجلات"),
            ("Deduplication", "إزالة التكرار"),
            ("GroupKFold", "تقسيم يحافظ على بقاء المجموعة في طيّة واحدة"),
        ]
    )
