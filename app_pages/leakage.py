"""تسريب البيانات واستراتيجيات التقسيم."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

from core import ui
from core.anim import play_animation

ui.page_header(
    "تسريب البيانات والتقسيم (Data leakage)",
    "التسريب هو أن يصل إلى النموذج، أثناء التدريب، **معلومة لن تكون متاحة "
    "وقت التنبّؤ الحقيقي**. نتيجته: أداء مبهر في التقرير، وانهيار في "
    "الإنتاج. وهو العيب المنهجي الأول في الأوراق البحثية التطبيقية.",
    icon=":material/water_damage:",
    badges=[
        ":red-badge[Target leakage]",
        ":orange-badge[Train-test contamination]",
        ":violet-badge[Temporal]",
        ":blue-badge[Group]",
        ":green-badge[Correct splitting]",
    ],
)

t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. ستة أنواع من التسريب", divider="red")
    ui.concept_grid(
        [
            (
                "١. تسريب الهدف (Target leakage)",
                "متغيّر يحمل معلومة عن الهدف لا تتوفّر وقت التنبّؤ: "
                "`تاريخ_إلغاء_الاشتراك` للتنبّؤ بالإلغاء؛ "
                "`جرعة_الدواء` للتنبّؤ بالمرض؛ `تم_الاتصال_بالعميل` بعد "
                "اكتشاف المغادرة.\n\n"
                ":red[العلامة] متغيّر واحد يحتكر كل الأهمية ودقّة قرب 100٪.",
                "red",
            ),
            (
                "٢. تلوّث التدريب/الاختبار",
                "حساب أي إحصاء (متوسط، حدود، فئات، مكوّنات PCA، أهمية "
                "متغيّرات) من **كامل** البيانات قبل التقسيم.\n\n"
                ":orange[الأشيع] وأخطره في اختيار المتغيّرات.",
                "orange",
            ),
            (
                "٣. التسريب الزمني",
                "استعمال بيانات من المستقبل: `bfill`، `rolling` بلا `shift`، "
                "تقسيم عشوائي على سلسلة زمنية، سعر صرف اليوم على معاملة "
                "قديمة.\n\n"
                ":violet[القاعدة] التدريب يسبق الاختبار **زمنيًا** دائمًا.",
                "violet",
            ),
            (
                "٤. تسريب المجموعة",
                "نفس المريض/العميل/الصورة في التدريب والاختبار معًا. النموذج "
                "يتعرّف على **الفرد** لا على الظاهرة.\n\n"
                ":blue[الحل] `GroupKFold` / `GroupShuffleSplit` بمعرّف الفرد.",
                "blue",
            ),
            (
                "٥. التسريب بالتكرار",
                "صفوف مكرّرة موزّعة بين التدريب والاختبار ⇒ حفظ لا تعلّم.\n\n"
                "(راجع درس **القيم المكرّرة**.)",
                "green",
            ),
            (
                "٦. تسريب الضبط (Tuning leakage)",
                "تجريب عشرات النماذج والعتبات والمعاملات على **الاختبار** ثم "
                "الإبلاغ عن الأفضل. الاختبار يتحوّل تدريجيًا إلى مجموعة "
                "تدريب ثانية.\n\n"
                ":red[الحل] تحقّق متقاطع متداخل + اختبار يُفتح مرّة واحدة.",
                "red",
            ),
        ],
        n_cols=3,
    )

    st.subheader("٢. اختبار واحد يكشف معظم الحالات", divider="green")
    st.markdown(
        """
> **لكل متغيّر، اسأل: متى تصبح قيمة هذا العمود معروفة؟**
> وقارنها بـ **لحظة التنبّؤ**.

ارسم خطًّا زمنيًا بسيطًا لكل سجل:

```
   تسجيل العميل ──── لحظة التنبّؤ ──── وقوع الحدث (الهدف) ──── ما بعده
   [متاح ✅]          [الحدّ]           [ممنوع ❌]             [ممنوع ❌]
```

أي متغيّر يقع يمين الخطّ الفاصل — احذفه، مهما كانت قوّته. بل **خصوصًا** إن
كانت قوّته مذهلة.
"""
    )

    st.subheader("٣. استراتيجيات التقسيم الصحيحة", divider="blue")
    st.markdown(
        """
| الحالة | الاستراتيجية | لماذا |
|---|---|---|
| تصنيف عادي، صفوف مستقلّة | `train_test_split(stratify=y)` + `StratifiedKFold` | يحافظ على نسب الفئات في كل طيّة |
| سلسلة زمنية / تنبّؤ مستقبلي | `TimeSeriesSplit` أو قطع بتاريخ ثابت | التدريب يسبق الاختبار دائمًا |
| تكرار للفرد (مرضى، عملاء) | `GroupKFold` / `StratifiedGroupKFold` | الفرد كلّه في جهة واحدة |
| عيّنة صغيرة | `RepeatedStratifiedKFold` أو LOOCV | تقليل تباين التقدير |
| ضبط معاملات + تقييم | **تحقّق متقاطع متداخل** | الحلقة الداخلية تضبط، والخارجية تقيّم |
| انزياح توزيعي متوقّع | تقسيم خارج التوزيع (OOD) عمدًا | يختبر التعميم الحقيقي |

:orange-badge[فجوة زمنية] في السلاسل الزمنية، أضف **فاصلًا (gap)** بين
التدريب والاختبار يساوي أفق التنبّؤ، وإلا تسرّبت معلومة عبر الارتباط
الذاتي. `TimeSeriesSplit(n_splits=5, gap=30)`.
"""
    )

    st.subheader("٤. كيف تكتشف تسريبًا قبل فوات الأوان؟", divider="orange")
    st.markdown(
        """
1. **الأداء أعلى من المعقول.** AUC = 0.99 في مسألة اجتماعية أو طبية؟
   ابحث عن التسريب قبل أن تحتفل.
2. **متغيّر واحد يحتكر الأهمية.** افحص كيف يُحسب بالضبط، ومن أين يأتي.
3. **فجوة صفرية بين التدريب والاختبار.** مريب أحيانًا بقدر الفجوة الكبيرة.
4. **اختبار على فترة زمنية لاحقة.** أقوى اختبار عملي: درّب على 2022 واختبر
   على 2023. إن انهار الأداء فالسبب غالبًا تسريب أو انزياح.
5. **راجع مع صاحب البيانات** كل عمود: *«متى يُملأ هذا الحقل في نظامكم؟»*
   سؤال واحد يكشف ما لا يكشفه أي تحليل.
"""
    )

    ui.beginner_note(
        """
مثال شهير: نموذج يتنبّأ بإصابة المريض بالالتهاب الرئوي بدقّة مذهلة —
ثم اتّضح أن أحد المتغيّرات كان «هل أُعطي مضاد حيوي؟». طبعًا: يُعطى المضاد
**بعد** التشخيص. النموذج لم يتنبّأ بشيء؛ قرأ الإجابة من ورقة أخرى.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. التقسيم الصحيح لكل حالة", divider="blue")
    ui.code(
        """
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GroupKFold, StratifiedGroupKFold,
    TimeSeriesSplit, GroupShuffleSplit, RepeatedStratifiedKFold,
)

# تصنيف عادي
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# صفوف تابعة لمجموعات (مريض/عميل)
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
tr_idx, te_idx = next(gss.split(X, y, groups=df["patient_id"]))

# تحقّق متقاطع يحترم المجموعات والفئات معًا
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X, y, groups=df["patient_id"], cv=cv)

# سلسلة زمنية (مع فاصل بطول أفق التنبّؤ)
tscv = TimeSeriesSplit(n_splits=5, gap=30, test_size=90)

# قطع بتاريخ ثابت — الأوضح للتقارير
train = df[df["date"] < "2024-01-01"]
test  = df[df["date"] >= "2024-01-01"]
"""
    )

    st.subheader("٢. تحقّق متقاطع متداخل (nested CV)", divider="green")
    ui.code(
        """
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

inner = KFold(n_splits=4, shuffle=True, random_state=1)   # لضبط المعاملات
outer = KFold(n_splits=5, shuffle=True, random_state=2)   # للتقييم النزيه

search = GridSearchCV(pipe, param_grid, cv=inner, scoring="roc_auc")
nested_scores = cross_val_score(search, X, y, cv=outer, scoring="roc_auc")

print(f"تقدير نزيه: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")
""",
        caption="الفرق بين النتيجة المتداخلة وغير المتداخلة هو **مقدار "
        "التفاؤل** الناتج عن ضبط المعاملات. أبلغ عن المتداخلة.",
    )

    st.subheader("٣. فحوص آلية للكشف عن التسريب", divider="orange")
    ui.code(
        '''
# (أ) متغيّر مرتبط بالهدف ارتباطًا شبه تام
corr = X.corrwith(y).abs().sort_values(ascending=False)
print("مشبوه:", corr[corr > 0.95].index.tolist())

# (ب) متغيّر واحد يحقّق وحده أداءً مرتفعًا
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
for c in X.columns:
    s = cross_val_score(DecisionTreeClassifier(max_depth=3),
                        X[[c]].fillna(-999), y, cv=3, scoring="roc_auc").mean()
    if s > 0.9:
        print(f"{c}: AUC={s:.3f} بمفرده ⇒ افحصه فورًا")

# (ج) تداخل المجموعات بين التدريب والاختبار
overlap = set(df.loc[tr_idx, "patient_id"]) & set(df.loc[te_idx, "patient_id"])
assert not overlap, f"{len(overlap)} مجموعة في الجهتين!"

# (د) تداخل زمني
assert df.loc[tr_idx, "date"].max() < df.loc[te_idx, "date"].min()

# (هـ) صفوف متطابقة بين الجهتين
dup = pd.merge(X.loc[tr_idx], X.loc[te_idx], how="inner")
print(f"{len(dup)} صفًا مشتركًا بين التدريب والاختبار")
'''
    )

    st.subheader("٤. المتغيّرات الزمنية بلا تسريب", divider="violet")
    ui.code(
        """
df = df.sort_values(["customer_id", "date"])
g = df.groupby("customer_id")["amount"]

# ✅ صحيح: shift أولًا ثم rolling
df["roll_7"] = g.shift(1).rolling(7, min_periods=2).mean()

# ❌ خطأ: يُدخل قيمة اليوم نفسه
df["roll_7_bad"] = g.rolling(7).mean().reset_index(level=0, drop=True)

# ✅ تجميعات تاريخية تراكمية (expanding) بلا الصف الحالي
df["cum_mean"] = g.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
"""
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("استراتيجيات التقسيم أمام عينيك", divider="violet")
    st.markdown(
        "40 صفًا مرتّبة زمنيًا، تنتمي إلى 8 عملاء (كل عميل 5 صفوف). "
        "نعرض كيف توزّع كل استراتيجية الصفوف بين **تدريب** و**اختبار** في "
        "كل طيّة — وأين يحدث التسريب."
    )

    N = 40
    groups = np.repeat(np.arange(8), 5)
    y = np.array([0, 0, 1, 0, 1] * 8)
    X = np.arange(N).reshape(-1, 1)

    STRATS = [
        ("KFold عشوائي", KFold(n_splits=4, shuffle=True, random_state=0), None,
         ":red-badge[خطر] على بيانات زمنية أو مجمّعة: الاختبار يتناثر داخل التدريب، "
         "فيتعلّم النموذج من المستقبل ومن نفس الأفراد."),
        ("StratifiedKFold", StratifiedKFold(n_splits=4, shuffle=True, random_state=0), "y",
         ":green-badge[صحيح] للصفوف المستقلّة: يحافظ على نسبة الفئات في كل طيّة، "
         "وهو المعيار في التصنيف."),
        ("GroupKFold", GroupKFold(n_splits=4), "groups",
         ":blue-badge[ضروري] عند تكرار الفرد: كل عملاء الطيّة في جهة واحدة، "
         "فلا يتعرّف النموذج على الشخص بدل الظاهرة."),
        ("TimeSeriesSplit", TimeSeriesSplit(n_splits=4), None,
         ":violet-badge[إلزامي زمنيًا] التدريب دائمًا **قبل** الاختبار، "
         "وحجم التدريب ينمو تدريجيًا. لاحظ أن الطيّة الأولى تستعمل بيانات قليلة."),
    ]

    @st.cache_data(ttl=1800, max_entries=4)
    def split_frames() -> pd.DataFrame:
        rows = []
        for name, cv, kind, _note in STRATS:
            if kind == "y":
                splits = cv.split(X, y)
            elif kind == "groups":
                splits = cv.split(X, y, groups=groups)
            else:
                splits = cv.split(X)
            for f, (tr, te) in enumerate(splits):
                role = np.array(["unused"] * N, dtype=object)
                role[tr] = "train"
                role[te] = "test"
                rows.append(
                    pd.DataFrame(
                        {
                            "strategy": name,
                            "fold": f + 1,
                            "row": np.arange(N),
                            "group": groups,
                            "role": role,
                        }
                    )
                )
        return pd.concat(rows, ignore_index=True)

    frames = split_frames()

    def _render_split(frame: int) -> None:
        name, _cv, _k, note = STRATS[frame]
        sub = frames[frames["strategy"] == name]
        ch = (
            alt.Chart(sub)
            .mark_rect(stroke="white", strokeWidth=1)
            .encode(
                x=alt.X("row:O", title="rows ordered by time →", axis=None),
                y=alt.Y("fold:O", title="fold"),
                color=alt.Color(
                    "role:N",
                    scale=alt.Scale(
                        domain=["train", "test", "unused"],
                        range=["#4C6EF5", "#E8590C", "#E9ECEF"],
                    ),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=["fold", "row", "group", "role"],
            )
            .properties(height=160)
        )
        grp = (
            alt.Chart(sub[sub["fold"] == 1])
            .mark_rect()
            .encode(
                x=alt.X("row:O", axis=None),
                color=alt.Color("group:N", legend=None),
            )
            .properties(height=22, title="customer id (group)")
        )
        st.altair_chart(ch, width="stretch")
        st.altair_chart(grp, width="stretch")
        st.markdown(f"**:blue-badge[{name}]** {note}")

    play_animation(
        "split_strategies",
        len(STRATS),
        _render_split,
        fps=0.5,
        label="الاستراتيجية",
        frame_labels=[s[0] for s in STRATS],
    )

    st.subheader("شريط التسريب: أين يقع كل خطأ؟", divider="red")
    LEAKS = pd.DataFrame(
        {
            "stage": [
                "جمع البيانات",
                "التنظيف",
                "التقسيم",
                "المعالجة",
                "هندسة المتغيّرات",
                "اختيار المتغيّرات",
                "إعادة المعاينة",
                "ضبط المعاملات",
                "التقييم",
            ],
            "typical_leak": [
                "عمود يُملأ بعد وقوع الهدف",
                "حذف الشواذ من الاختبار أيضًا",
                "تقسيم عشوائي لبيانات زمنية/مجمّعة",
                "`fit_transform` على كل البيانات",
                "تجميعات من كامل الجدول / rolling بلا shift",
                "اختيار أفضل k قبل التقسيم",
                "SMOTE قبل التقسيم",
                "تجريب متكرّر على الاختبار",
                "تقرير أفضل نتيجة من بين عشرات",
            ],
            "severity": [5, 3, 5, 4, 4, 5, 4, 3, 3],
        }
    )
    st.altair_chart(
        alt.Chart(LEAKS)
        .mark_bar()
        .encode(
            y=alt.Y("stage:N", sort=None, title=None),
            x=alt.X("severity:Q", title="خطورة نسبية (1-5)", scale=alt.Scale(domain=[0, 5])),
            color=alt.Color("severity:Q", legend=None),
            tooltip=["stage", "typical_leak", "severity"],
        )
        .properties(height=300),
        width="stretch",
    )
    st.dataframe(LEAKS, width="stretch", hide_index=True)

# ============================================================== القرار
with t_decide:
    st.subheader("شجرة قرار التقسيم", divider="orange")
    ui.decision_table(
        [
            ("صفوف مستقلّة، تصنيف", "`StratifiedKFold` + `train_test_split(stratify=y)`", "يحافظ على نسب الفئات"),
            ("يوجد بُعد زمني وسنتنبّأ بالمستقبل", "`TimeSeriesSplit(gap=أفق التنبّؤ)`", "منع رؤية المستقبل والارتباط الذاتي"),
            ("تكرار الفرد في عدّة صفوف", "`StratifiedGroupKFold`", "الفرد كلّه في جهة واحدة"),
            ("زمني ومجمّع معًا", "قطع بتاريخ + فحص تداخل المجموعات", "لا توجد أداة جاهزة تجمع الاثنين — تحقّق يدويًا"),
            ("عيّنة صغيرة جدًا", "`RepeatedStratifiedKFold`", "تقليل تباين التقدير"),
            ("ستضبط معاملات كثيرة", "تحقّق متقاطع متداخل", "وإلا تسرّب الضبط إلى التقدير"),
            ("أداء مرتفع بشكل مريب", "أوقف كل شيء وافحص المتغيّرات", "الاحتفال المبكر أغلى من ساعة مراجعة"),
            ("قبل التسليم النهائي", "اختبار على فترة زمنية لاحقة", "أقرب محاكاة للإنتاج"),
        ]
    )

    ui.danger(
        "اجعل هذه قاعدة شخصية: **مجموعة الاختبار تُفتح مرّة واحدة فقط، في "
        "نهاية المشروع.** كل قرار (معالجة، متغيّرات، نموذج، معاملات، عتبة) "
        "يُتّخذ بتحقّق متقاطع على التدريب وحده."
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("scikit-learn — Common pitfalls and recommended practices", "https://scikit-learn.org/stable/common_pitfalls.html"),
            ("scikit-learn — Cross-validation: evaluating estimator performance", "https://scikit-learn.org/stable/modules/cross_validation.html"),
            ("scikit-learn — Visualizing cross-validation behavior", "https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html"),
            ("scikit-learn — Nested versus non-nested cross-validation", "https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html"),
            ("Kaufman, Rosset & Perlich (2012) — Leakage in Data Mining", "https://dl.acm.org/doi/10.1145/2020408.2020496"),
            ("Kapoor & Narayanan (2023) — Leakage and the Reproducibility Crisis in ML-based Science", "https://doi.org/10.1016/j.patter.2023.100804"),
            ("Caruana et al. (2015) — Intelligible Models (مثال الالتهاب الرئوي)", "https://doi.org/10.1145/2783258.2788613"),
        ]
    )
    ui.terms(
        [
            ("Data leakage", "تسريب البيانات"),
            ("Target leakage", "تسريب الهدف"),
            ("Train-test contamination", "تلوّث التدريب بالاختبار"),
            ("Temporal leakage", "تسريب زمني"),
            ("Group leakage", "تسريب المجموعة"),
            ("Nested cross-validation", "تحقّق متقاطع متداخل"),
            ("TimeSeriesSplit", "تقسيم زمني متتابع"),
            ("Gap", "فاصل زمني بين التدريب والاختبار"),
        ]
    )
