"""اختيار المتغيّرات (Feature selection)."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core import charts, ui
from core.anim import play_animation

ui.page_header(
    "اختيار المتغيّرات (Feature selection)",
    "ليس كل متغيّر يستحق البقاء. المتغيّرات الزائدة تضيف ضجيجًا وتزيد فرط "
    "الملاءمة وتكلفة الجمع والصيانة. لكن الاختيار نفسه **خطوة مُتعلَّمة**، "
    "وأداؤه خارج `Pipeline` يُنتج أشهر وهم في التعلّم الآلي.",
    icon=":material/filter_alt:",
    badges=[
        ":blue-badge[Filter]",
        ":green-badge[Wrapper]",
        ":orange-badge[Embedded]",
        ":violet-badge[Permutation]",
        ":red-badge[Selection bias]",
    ],
)


# ----------------------------------------------------- بيانات التجربة
@st.cache_data(ttl=1800, max_entries=4)
def noisy_task(n: int = 700, n_informative: int = 5, n_noise: int = 25) -> tuple[pd.DataFrame, pd.Series]:
    """٥ متغيّرات حقيقية + ٢٥ متغيّر ضجيج خالص (نعرف الحقيقة سلفًا)."""
    rng = np.random.default_rng(7)
    info = rng.normal(size=(n, n_informative))
    w = np.array([2.0, -1.6, 1.2, 0.9, -0.7])[:n_informative]
    logit = info @ w + rng.normal(0, 0.8, n)
    y = (logit > np.median(logit)).astype(int)

    noise = rng.normal(size=(n, n_noise))
    cols = [f"signal_{i + 1}" for i in range(n_informative)] + [f"noise_{i + 1}" for i in range(n_noise)]
    X = pd.DataFrame(np.hstack([info, noise]), columns=cols)
    return X, pd.Series(y, name="target")


@st.cache_data(ttl=1800, max_entries=4, show_spinner="جارٍ حساب منحنى الاختيار…")
def rfe_curve() -> pd.DataFrame:
    X, y = noisy_task()
    rows = []
    for k in range(1, X.shape[1] + 1, 2):
        pipe = Pipeline(
            [
                ("sc", StandardScaler()),
                ("sel", RFE(LogisticRegression(max_iter=1000), n_features_to_select=k)),
                ("m", LogisticRegression(max_iter=1000)),
            ]
        )
        s = cross_val_score(pipe, X, y, cv=4, scoring="roc_auc")
        rows.append({"k": k, "roc_auc": float(s.mean()), "std": float(s.std())})
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800, max_entries=4, show_spinner="جارٍ ترتيب المتغيّرات…")
def importance_table() -> pd.DataFrame:
    X, y = noisy_task()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    f_scores = pd.Series(f_classif(Xtr, ytr)[0], index=X.columns)
    mi = pd.Series(mutual_info_classif(Xtr, ytr, random_state=0), index=X.columns)

    lasso = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(penalty="l1", solver="liblinear", C=0.1))]).fit(Xtr, ytr)
    l1 = pd.Series(np.abs(lasso[-1].coef_.ravel()), index=X.columns)

    rf = RandomForestClassifier(n_estimators=250, random_state=0).fit(Xtr, ytr)
    gini = pd.Series(rf.feature_importances_, index=X.columns)
    perm = permutation_importance(rf, Xte, yte, n_repeats=8, random_state=0, scoring="roc_auc")
    pi = pd.Series(perm.importances_mean, index=X.columns)

    out = pd.DataFrame(
        {
            "ANOVA F": f_scores / f_scores.max(),
            "Mutual info": mi / max(mi.max(), 1e-9),
            "L1 (Lasso)": l1 / max(l1.max(), 1e-9),
            "RF impurity (Gini)": gini / gini.max(),
            "Permutation": pi / max(pi.max(), 1e-9),
        }
    )
    out.index.name = "feature"
    out = out.reset_index()
    out["is_signal"] = out["feature"].str.startswith("signal")
    return out


@st.cache_data(ttl=1800, max_entries=4, show_spinner="جارٍ محاكاة تحيّز الاختيار…")
def selection_bias() -> pd.DataFrame:
    """المثال الكلاسيكي: بيانات عشوائية تمامًا + اختيار قبل التقسيم."""
    rng = np.random.default_rng(3)
    n, p = 160, 600
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"v{i}" for i in range(p)])
    y = pd.Series(rng.integers(0, 2, n))  # لا علاقة إطلاقًا

    # (أ) الطريقة الخاطئة: اختيار أفضل 15 متغيّرًا على **كل** البيانات ثم CV
    sel = SelectKBest(f_classif, k=15).fit(X, y)
    X_sel = X.loc[:, sel.get_support()]
    wrong = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y, cv=5, scoring="accuracy").mean()

    # (ب) الطريقة الصحيحة: الاختيار داخل Pipeline ⇒ يُعاد في كل طيّة
    pipe = Pipeline([("sel", SelectKBest(f_classif, k=15)), ("m", LogisticRegression(max_iter=1000))])
    right = cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()

    return pd.DataFrame(
        {
            "approach": ["اختيار قبل التقسيم (خطأ)", "اختيار داخل Pipeline (صواب)", "الحقيقة (تخمين عشوائي)"],
            "accuracy": [float(wrong), float(right), 0.5],
        }
    )


t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. لماذا نختار أصلًا؟", divider="blue")
    st.markdown(
        """
- **لعنة الأبعاد:** كلما زادت الأبعاد، تفرّقت البيانات وصار كل صف «بعيدًا»
  عن الآخر، فتحتاج عيّنة أكبر أسّيًا لنفس الدقّة.
- **فرط الملاءمة:** متغيّر ضجيج قد يبدو مفيدًا بالصدفة. مع 500 متغيّر عشوائي
  ستجد حتمًا عشرات «المعنوية» عند $\\alpha = 0.05$.
- **التكلفة:** كل متغيّر يعني جمعًا وتخزينًا وصيانة ونقطة فشل محتملة في
  الإنتاج.
- **التفسير:** نموذج بـ 8 متغيّرات يمكن شرحه لصاحب القرار؛ نموذج بـ 300 لا.
- **السرعة:** تدريب وتنبّؤ أسرع.

:orange-badge[لكن] الاختيار ليس دائمًا مفيدًا للأداء. مع النماذج المنظّمة
القوية (Ridge/Lasso/GBM) وبيانات كبيرة، قد لا يكسب الاختيار شيئًا —
والمكسب يكون في التفسير والتكلفة لا في الدقّة.
"""
    )

    st.subheader("٢. العائلات الثلاث", divider="green")
    ui.concept_grid(
        [
            (
                "Filter — ترشيح",
                "تقييم كل متغيّر **بمعزل** عن النموذج: التباين، الارتباط، "
                "`ANOVA F`، `chi2`، المعلومات المتبادلة (mutual information).\n\n"
                ":green[مزايا] سريع جدًا، مستقل عن النموذج، مناسب للتصفية "
                "الأولية لآلاف المتغيّرات.\n"
                ":red[عيوب] **يتجاهل التفاعلات**: متغيّران عديما الفائدة "
                "منفردين قد يكونان حاسمين معًا.",
                "blue",
            ),
            (
                "Wrapper — تغليف",
                "جرّب مجموعات فعليًا وقِس أداء النموذج: `RFE`, `RFECV`, "
                "`SequentialFeatureSelector` (أمامي/خلفي).\n\n"
                ":green[مزايا] يراعي التفاعلات والنموذج المستهدف.\n"
                ":red[عيوب] مكلف حسابيًا جدًا، وعرضة لفرط الملاءمة إن لم يكن "
                "داخل تحقّق متقاطع متداخل (nested CV).",
                "green",
            ),
            (
                "Embedded — مدمج",
                "الاختيار جزء من التدريب: `Lasso` (L1) يصفّر معاملات؛ الأشجار "
                "تعطي `feature_importances_`؛ `SelectFromModel`.\n\n"
                ":green[مزايا] توازن ممتاز بين التكلفة والجودة.\n"
                ":red[عيوب] متحيّز لنوع النموذج؛ و L1 يختار عشوائيًا واحدًا من "
                "كل مجموعة متغيّرات مترابطة.",
                "orange",
            ),
            (
                "Permutation importance",
                "اخلط عمودًا واحدًا عشوائيًا وقِس كم تدهور الأداء. "
                "يُحسب على بيانات **لم يرها النموذج**.\n\n"
                ":green[مزايا] يصلح لأي نموذج، ويقيس الأهمية الحقيقية للتنبّؤ.\n"
                ":red[عيوب] بطيء؛ ويُقلّل أهمية المتغيّرات المترابطة (لأن البديل "
                "يعوّض الناقص).",
                "violet",
            ),
        ],
        n_cols=2,
    )

    st.subheader("٣. ثلاثة فخاخ شائعة", divider="red")
    st.markdown(
        """
**أ) تحيّز الاختيار (selection bias):** اختيار المتغيّرات على **كامل**
البيانات ثم التحقّق المتقاطع. النتيجة متفائلة بشكل صارخ — حتى مع بيانات
عشوائية تمامًا. هذا الفخ موثّق في *Elements of Statistical Learning* (§7.10.2)
وتجده في المختبر أسفل الصفحة.

**ب) أهمية الشوائب (impurity importance) متحيّزة:** `feature_importances_`
في الأشجار تميل إلى المتغيّرات **عالية الكاردينالية** والمستمرّة، لأنها تتيح
تقسيمات أكثر. متغيّر عشوائي مستمرّ قد يبدو أهم من متغيّر ثنائي حقيقي.
**البديل:** `permutation_importance` على بيانات التحقّق.

**ج) المتغيّرات المترابطة:** إن كان `A` و `B` مترابطين بـ 0.95، فكل طرق
الأهمية ستقسم الفضل بينهما أو تختار أحدهما عشوائيًا. لا تستنتج أن المتغيّر
غير المختار «عديم الفائدة». عالج الترابط أولًا (VIF، تجميع، PCA).
"""
    )

    st.latex(r"VIF_j = \frac{1}{1 - R_j^2}")
    st.caption(
        "معامل تضخّم التباين: $R_j^2$ من انحدار المتغيّر $j$ على بقيّة "
        "المتغيّرات. قاعدة شائعة: $VIF > 5$ (أو 10) ⇒ تعدّد خطّي يستحق العلاج "
        "— وهو مهم للاستدلال أكثر منه للتنبّؤ."
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. ترشيح سريع (Filter)", divider="blue")
    ui.code(
        """
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, SelectPercentile,
    f_classif, f_regression, mutual_info_classif, chi2,
)

VarianceThreshold(threshold=0.0)                  # احذف الأعمدة الثابتة
SelectKBest(f_classif, k=20)                      # علاقة خطّية مع هدف تصنيفي
SelectKBest(mutual_info_classif, k=20)            # يلتقط العلاقات اللاخطّية أيضًا
SelectPercentile(f_regression, percentile=25)     # هدف مستمر
SelectKBest(chi2, k=20)                           # متغيّرات غير سالبة فقط (تكرارات)

# الترابط بين المتغيّرات نفسها
corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if (upper[c] > 0.95).any()]
"""
    )

    st.subheader("٢. تغليف (Wrapper)", divider="green")
    ui.code(
        """
from sklearn.feature_selection import RFE, RFECV, SequentialFeatureSelector

# حذف تكراري مع عدد محدّد سلفًا
RFE(LogisticRegression(), n_features_to_select=10, step=1)

# يختار العدد الأمثل بنفسه عبر CV
sel = RFECV(LogisticRegression(max_iter=1000), cv=5, scoring="roc_auc", min_features_to_select=3)
sel.fit(X_train, y_train)
print("العدد الأمثل:", sel.n_features_, "|", X.columns[sel.support_].tolist())

# بحث أمامي/خلفي متدرّج
SequentialFeatureSelector(estimator, n_features_to_select="auto", direction="forward", cv=5)
""",
        caption="`RFECV` مكلف لكنه يعطي منحنى الأداء مقابل عدد المتغيّرات — "
        "أفضل من تخمين `k`.",
    )

    st.subheader("٣. مدمج (Embedded) وأهمية التبديل", divider="orange")
    ui.code(
        """
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.inspection import permutation_importance

# L1 يصفّر المعاملات غير المفيدة
SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=0.1))
SelectFromModel(LassoCV(cv=5), threshold="median")

# أهمية الأشجار (حذر: متحيّزة للكاردينالية العالية)
rf.feature_importances_

# أهمية التبديل على بيانات التحقّق (الأدق)
r = permutation_importance(model, X_valid, y_valid, n_repeats=10,
                           random_state=0, scoring="roc_auc")
imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
"""
    )

    st.subheader("٤. المكان الصحيح: داخل Pipeline", divider="red")
    ui.code(
        """
pipe = Pipeline([
    ("pre", preprocessor),
    ("sel", SelectKBest(f_classif, k=20)),   # يُعاد تعلّمه في كل طيّة
    ("clf", LogisticRegression(max_iter=1000)),
])
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")

# ولضبط k نفسه: تحقّق متقاطع متداخل
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, {"sel__k": [5, 10, 20, 40]}, cv=5, scoring="roc_auc")
""",
        caption=":red-badge[القاعدة] إن كان الاختيار يستعمل `y`، فهو **مُتعلَّم**، "
        "ومكانه داخل الـ Pipeline — بلا استثناء.",
    )

    st.subheader("٥. التعدّد الخطّي (VIF)", divider="violet")
    ui.code(
        """
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

Xc = sm.add_constant(X[num_cols].dropna())
vif = pd.Series(
    [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])],
    index=Xc.columns,
).drop("const").sort_values(ascending=False)
print(vif[vif > 5])
"""
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("أ. كم متغيّرًا نحتاج فعلًا؟", divider="violet")
    st.markdown(
        "بيانات فيها **٥ متغيّرات حقيقية** و**٢٥ متغيّر ضجيج خالص**. "
        "نشغّل `RFE` بأعداد متزايدة ونقيس الأداء بتحقّق متقاطع. "
        "راقب أين يتوقّف التحسّن — ثم يبدأ التدهور."
    )

    curve = rfe_curve()
    best_k = int(curve.loc[curve["roc_auc"].idxmax(), "k"])

    def _render_rfe(frame: int) -> None:
        row = curve.iloc[frame]
        k1, k2, k3 = st.columns(3)
        k1.metric("عدد المتغيّرات المختارة", int(row["k"]))
        k2.metric("ROC-AUC", f"{row['roc_auc']:.3f}", delta=f"± {row['std']:.3f}", delta_color="off")
        k3.metric("أفضل k حتى الآن", best_k if frame == len(curve) - 1 else int(curve.iloc[: frame + 1].loc[curve.iloc[: frame + 1]["roc_auc"].idxmax(), "k"]))

        sofar = curve.iloc[: frame + 1]
        ch = (
            alt.Chart(sofar)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("k:Q", title="number of selected features", scale=alt.Scale(domain=[0, 31])),
                y=alt.Y("roc_auc:Q", title="cross-validated ROC-AUC", scale=alt.Scale(domain=[0.6, 1.0])),
                tooltip=["k", "roc_auc"],
            )
            .properties(height=330)
        )
        st.altair_chart(ch + charts.vrule(5, color="#0CA678"), width="stretch")
        st.caption("الخط الأخضر عند k = 5: العدد الحقيقي للمتغيّرات المفيدة.")

    play_animation("rfe_curve", len(curve), _render_rfe, fps=2.0, label="k")

    st.subheader("ب. هل تتفق طرق الأهمية على نفس المتغيّرات؟", divider="orange")
    imp = importance_table()
    methods = ["ANOVA F", "Mutual info", "L1 (Lasso)", "RF impurity (Gini)", "Permutation"]

    def _render_imp(frame: int) -> None:
        m = methods[frame]
        d = imp[["feature", m, "is_signal"]].copy()
        d.columns = ["feature", "importance", "is_signal"]
        d = d.sort_values("importance", ascending=False).head(12)
        top5 = d.head(5)["is_signal"].sum()

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("الطريقة", m)
            st.metric("من أصل أعلى ٥: متغيّرات حقيقية", f"{int(top5)}/5")
        with c2:
            ch = (
                alt.Chart(d)
                .mark_bar()
                .encode(
                    y=alt.Y("feature:N", sort="-x", title=None),
                    x=alt.X("importance:Q", title="normalised importance"),
                    color=alt.Color(
                        "is_signal:N",
                        scale=alt.Scale(domain=[True, False], range=["#0CA678", "#ADB5BD"]),
                        legend=alt.Legend(orient="top", title="متغيّر حقيقي؟"),
                    ),
                    tooltip=["feature", "importance"],
                )
                .properties(height=300)
            )
            st.altair_chart(ch, width="stretch")

        notes = {
            "ANOVA F": "علاقة **خطّية** أحادية مع الهدف. سريع جدًا لكنه يعمى عن أي علاقة غير خطّية أو تفاعلية.",
            "Mutual info": "يلتقط أي ارتباط إحصائي (خطّي أو لا). أبطأ قليلًا وأكثر تذبذبًا في العيّنات الصغيرة.",
            "L1 (Lasso)": "يصفّر المعاملات غير المفيدة مباشرة؛ ممتاز، لكنه يختار واحدًا فقط من كل زوج مترابط.",
            "RF impurity (Gini)": ":orange[حذر] متحيّز نحو المتغيّرات المستمرّة وعالية الكاردينالية — قد ترتفع متغيّرات ضجيج هنا.",
            "Permutation": "يقيس الانخفاض الفعلي في الأداء على بيانات التحقّق ⇒ الأقرب إلى «الأهمية الحقيقية».",
        }
        st.markdown(f"**:violet-badge[{m}]** {notes[m]}")

    play_animation("imp_methods", len(methods), _render_imp, fps=0.5, label="الطريقة", frame_labels=methods)

    st.subheader("ج. مختبر: تحيّز الاختيار على بيانات عشوائية", divider="red")
    st.markdown(
        "بيانات **عشوائية بالكامل**: 160 صفًا و600 متغيّرًا، والهدف قُرعة "
        "نزيهة. الأداء الصحيح الوحيد الممكن هو **0.5**. فلنرَ ماذا يحدث إن "
        "اخترنا أفضل 15 متغيّرًا قبل التقسيم."
    )
    bias = selection_bias()
    st.altair_chart(
        alt.Chart(bias)
        .mark_bar()
        .encode(
            y=alt.Y("approach:N", title=None, sort="-x"),
            x=alt.X("accuracy:Q", title="cross-validated accuracy", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("approach:N", legend=None),
            tooltip=["approach", "accuracy"],
        )
        .properties(height=200),
        width="stretch",
    )
    st.dataframe(bias.assign(accuracy=bias["accuracy"].round(3)), width="stretch", hide_index=True)
    st.markdown(
        """
:red-badge[النتيجة] الاختيار قبل التقسيم يرفع الدقّة فوق 0.5 بوضوح على
بيانات **لا تحتوي أي معلومة**. السبب: المتغيّرات الـ15 اختيرت لأنها الأكثر
ارتباطًا بالهدف **في هذه العيّنة بالذات**، بما في ذلك صفوف التحقّق. الخطوة
الأولى سرّبت الهدف، ثم جاء التحقّق المتقاطع ليبارك النتيجة.

:green-badge[العلاج] ضع `SelectKBest` داخل `Pipeline`: عندها يُعاد الاختيار
داخل كل طيّة من بيانات التدريب وحدها، فتعود النتيجة إلى 0.5 — وهي الحقيقة.
"""
    )

# ============================================================== القرار
with t_decide:
    st.subheader("أي طريقة اختيار؟", divider="orange")
    ui.decision_table(
        [
            ("آلاف المتغيّرات (جينوم، نصوص)", "Filter أولًا ثم Embedded", "التغليف مستحيل حسابيًا بهذا الحجم"),
            ("عشرات المتغيّرات ووقت متاح", "RFECV أو SequentialFeatureSelector", "يراعي التفاعلات ويختار العدد تلقائيًا"),
            ("تريد نموذجًا مفسَّرًا", "L1 (Lasso) أو RFE", "ينتج مجموعة صغيرة صريحة"),
            ("نموذج شجري/GBM", "Permutation importance على التحقّق", "أهمية الشوائب متحيّزة للكاردينالية"),
            ("متغيّرات مترابطة بشدّة", "عالج الترابط أولًا (VIF/تجميع)", "وإلا توزّعت الأهمية عشوائيًا بينها"),
            ("أعمدة ثابتة أو شبه ثابتة", "`VarianceThreshold`", "صفر معلومة"),
            ("بيانات قليلة الصفوف", "قلّل المتغيّرات بقوّة", "نسبة p/n العالية تضمن فرط الملاءمة"),
            ("الهدف هو التنبّؤ فقط، والبيانات كبيرة", "قد لا تحتاج اختيارًا", "التنظيم (regularization) يقوم بالمهمّة"),
            ("نشر في الإنتاج", "فضّل المتغيّرات الرخيصة والمستقرّة", "متغيّر دقيق لكنه غير متاح لحظة التنبّؤ = عديم الفائدة"),
        ],
        headers=("الحالة", "الطريقة", "السبب"),
    )

    ui.danger(
        "ضبط عدد المتغيّرات `k` بنفس التحقّق المتقاطع الذي تُبلغ عنه هو "
        "تسريب أيضًا — وإن كان أخف. الحلّ المنهجي: **تحقّق متقاطع متداخل "
        "(nested CV)**: حلقة داخلية للاختيار والضبط، وخارجية للتقييم."
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("scikit-learn — Feature selection (دليل المستخدم)", "https://scikit-learn.org/stable/modules/feature_selection.html"),
            ("scikit-learn — Permutation feature importance", "https://scikit-learn.org/stable/modules/permutation_importance.html"),
            ("scikit-learn — Permutation Importance vs Random Forest Feature Importance", "https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html"),
            ("scikit-learn — Recursive feature elimination with cross-validation", "https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html"),
            ("scikit-learn — Nested versus non-nested cross-validation", "https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html"),
            ("Hastie, Tibshirani & Friedman — ESL §7.10.2 (الطريقة الخاطئة للتحقّق المتقاطع)", "https://hastie.su.domains/ElemStatLearn/"),
            ("Guyon & Elisseeff (2003) — An Introduction to Variable and Feature Selection", "https://www.jmlr.org/papers/v3/guyon03a.html"),
        ]
    )
    ui.terms(
        [
            ("Filter method", "طريقة ترشيح"),
            ("Wrapper method", "طريقة تغليف"),
            ("Embedded method", "طريقة مدمجة"),
            ("RFE", "الحذف التكراري للمتغيّرات"),
            ("Permutation importance", "أهمية التبديل"),
            ("Multicollinearity / VIF", "التعدّد الخطّي ومعامل تضخّم التباين"),
            ("Nested cross-validation", "تحقّق متقاطع متداخل"),
            ("Selection bias", "تحيّز الاختيار"),
        ]
    )
