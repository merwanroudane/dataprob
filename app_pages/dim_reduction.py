"""تقليل الأبعاد: PCA وما بعده."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core import charts, ui
from core.anim import play_animation
from core.datasets import clean_customers

ui.page_header(
    "تقليل الأبعاد (PCA)",
    "بدل حذف متغيّرات، **ادمجها**: ابحث عن عدد صغير من المحاور الجديدة التي "
    "تحمل معظم المعلومة. أداة قوية — ولها ثمن صريح: **تفقد معنى الأعمدة**.",
    icon=":material/compress:",
    badges=[
        ":blue-badge[PCA]",
        ":green-badge[Explained variance]",
        ":orange-badge[Scaling required]",
        ":violet-badge[t-SNE / UMAP]",
        ":red-badge[Not always better]",
    ],
)


@st.cache_data(ttl=1800, max_entries=4)
def pca_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """بيانات بأعمدة مترابطة بقوّة — الحالة التي يتألّق فيها PCA."""
    base = clean_customers(600)
    rng = np.random.default_rng(5)
    d = pd.DataFrame(
        {
            "income": base["income"],
            "spending": base["spending"],
            "age": base["age"],
            "experience": base["experience_years"],
            "satisfaction": base["satisfaction"],
        }
    )
    # أعمدة مشتقّة مترابطة (كما يحدث كثيرًا في الواقع)
    d["income_eur"] = d["income"] / 145 * (1 + rng.normal(0, 0.01, len(d)))
    d["savings"] = d["income"] - d["spending"] + rng.normal(0, 500, len(d))
    d["seniority"] = d["experience"] * 12 + rng.normal(0, 3, len(d))
    d["spend_ratio"] = d["spending"] / d["income"]
    return d, base["churn"], list(d.columns)


@st.cache_data(ttl=1800, max_entries=4)
def pca_fit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, y, cols = pca_data()
    Xs = StandardScaler().fit_transform(X)
    p = PCA().fit(Xs)
    evr = pd.DataFrame(
        {
            "component": np.arange(1, len(p.explained_variance_ratio_) + 1),
            "explained": p.explained_variance_ratio_,
            "cumulative": np.cumsum(p.explained_variance_ratio_),
            "eigenvalue": p.explained_variance_,
        }
    )
    scores = pd.DataFrame(p.transform(Xs)[:, :2], columns=["PC1", "PC2"])
    scores["churn"] = y.astype(str).to_numpy()
    load = pd.DataFrame(p.components_[:3].T, columns=["PC1", "PC2", "PC3"], index=cols)
    load.index.name = "feature"
    return evr, scores, load.reset_index()


@st.cache_data(ttl=1800, max_entries=4, show_spinner="جارٍ تقييم عدد المكوّنات…")
def pca_performance() -> pd.DataFrame:
    X, y, cols = pca_data()
    rows = []
    for k in range(1, len(cols) + 1):
        pipe = Pipeline(
            [
                ("sc", StandardScaler()),
                ("pca", PCA(n_components=k, random_state=0)),
                ("m", LogisticRegression(max_iter=1000)),
            ]
        )
        s = cross_val_score(pipe, X, y, cv=4, scoring="roc_auc")
        rows.append({"k": k, "roc_auc": float(s.mean())})
    full = cross_val_score(
        Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000))]),
        X, y, cv=4, scoring="roc_auc",
    ).mean()
    out = pd.DataFrame(rows)
    out["all_features"] = float(full)
    return out


t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. ما الذي يفعله PCA بالضبط؟", divider="blue")
    st.markdown(
        """
يبحث عن محاور جديدة (**مكوّنات رئيسية**) بحيث:
1. المحور الأول يلتقط **أكبر تباين** ممكن في البيانات.
2. كل محور تالٍ **متعامد** على ما قبله ويلتقط أكبر تباين متبقٍّ.

رياضيًا: المكوّنات هي **المتّجهات الذاتية** لمصفوفة التغاير، و**القيم
الذاتية** هي مقدار التباين الذي يلتقطه كل مكوّن.
"""
    )
    st.latex(r"\Sigma = \frac{1}{n-1} X^{\top}X, \qquad \Sigma v_j = \lambda_j v_j, \qquad \text{EVR}_j = \frac{\lambda_j}{\sum_k \lambda_k}")

    st.markdown(
        """
المكوّن الأول هو تركيبة خطّية من كل الأعمدة:
$PC_1 = w_1 \\cdot income + w_2 \\cdot age + \\ldots$ — ولهذا **يفقد معناه**:
لم يعد عمودًا تستطيع أن تقول عنه «الدخل»، بل مزيج من كل شيء.
"""
    )

    st.subheader("٢. التقييس ليس اختياريًا هنا", divider="orange")
    st.markdown(
        """
PCA يعظّم **التباين**، والتباين يعتمد على الوحدة. عمود الدخل (تباينه
بالملايين) سيحتكر المكوّن الأول بالكامل لمجرّد أنه مقيس بالدينار.

:red-badge[قاعدة] `StandardScaler` قبل `PCA` دائمًا — إلا إن كانت كل
الأعمدة بنفس الوحدة أصلًا (كشدّة البكسل في الصور).
"""
    )

    st.subheader("٣. كم مكوّنًا نُبقي؟", divider="green")
    st.markdown(
        """
| القاعدة | الوصف | ملاحظة |
|---|---|---|
| **نسبة تباين مستهدفة** | أبقِ ما يفسّر 90٪ أو 95٪ | `PCA(n_components=0.95)` يفعلها مباشرة |
| **الكوع (Scree/elbow)** | ابحث عن الانكسار في منحنى القيم الذاتية | بصري ويحتمل التأويل |
| **معيار Kaiser** | أبقِ المكوّنات بقيمة ذاتية > 1 | على بيانات مُقيَّسة فقط |
| **الأداء بتحقّق متقاطع** | اضبط `n_components` كأي معامل آخر | :green-badge[الأصحّ للتنبّؤ] |

:orange-badge[تنبيه مهم] PCA **غير مُوجَّه (unsupervised)**: لا ينظر إلى
`y` إطلاقًا. المكوّن الأول هو الأكثر تباينًا، وليس بالضرورة الأكثر فائدة
للتنبّؤ. قد تكون المعلومة الحاسمة في المكوّن الخامس — وقد تحذفه.
البدائل المُوجَّهة: **PLS** (`PLSRegression`) و **LDA**.
"""
    )

    st.subheader("٤. ما وراء PCA", divider="violet")
    ui.concept_grid(
        [
            ("KernelPCA", "PCA في فضاء نواة ⇒ يلتقط البنى غير الخطّية. مكلف على البيانات الكبيرة.", "blue"),
            ("TruncatedSVD", "يعمل على المصفوفات المتناثرة دون توسيطها ⇒ الخيار الصحيح لنصوص TF-IDF (يُسمّى LSA).", "green"),
            ("t-SNE / UMAP", ":red[للتصوير فقط]. تحافظ على البنية المحلية لكنها لا توفّر تحويلًا مستقرًّا للبيانات الجديدة؛ استعمالها كمتغيّرات للنموذج خطأ شائع.", "violet"),
            ("Autoencoder", "ضغط غير خطّي متعلَّم عبر شبكة عصبية؛ يحتاج بيانات كبيرة وضبطًا.", "orange"),
            ("NMF", "تفكيك إلى عوامل **غير سالبة** ⇒ قابلية تفسير أفضل (مواضيع النصوص، أطياف).", "red"),
            ("PLS / LDA", "تقليل أبعاد **مُوجَّه بالهدف** ⇒ غالبًا أفضل من PCA حين الغاية التنبّؤ.", "green"),
        ],
        n_cols=3,
    )

    ui.beginner_note(
        """
تشبيه: لديك صور لمجسّم ثلاثي الأبعاد وتريد صورة ثنائية واحدة تحفظ أكبر
قدر من شكله. PCA يختار **زاوية التصوير** التي يبدو فيها المجسّم أكثر
«انتشارًا» (أكبر تباين). تفقد بعدًا، لكنك تحتفظ بأغلب الشكل.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. الاستعمال الصحيح", divider="blue")
    ui.code(
        """
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scale", StandardScaler()),      # إلزامي قبل PCA
    ("pca", PCA(n_components=0.95, random_state=0)),   # 95% من التباين
    ("model", LogisticRegression(max_iter=1000)),
])
pipe.fit(X_train, y_train)

pca = pipe.named_steps["pca"]
print("عدد المكوّنات:", pca.n_components_)
print("التباين المفسَّر:", pca.explained_variance_ratio_.round(3))
print("التراكمي:", pca.explained_variance_ratio_.cumsum().round(3))
""",
        caption=":red-badge[تسريب] `PCA().fit(X)` على كامل البيانات قبل التقسيم "
        "يجعل محاور الاختبار مُشتقّة جزئيًا من الاختبار نفسه.",
    )

    st.subheader("٢. تفسير المكوّنات (Loadings)", divider="green")
    ui.code(
        """
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=X.columns,
)
print(loadings["PC1"].sort_values(key=abs, ascending=False).head(5))
# الأعمدة ذات الوزن الأكبر (موجبًا أو سالبًا) هي التي «تصنع» هذا المحور
"""
    )

    st.subheader("٣. إعادة البناء وقياس الخسارة", divider="orange")
    ui.code(
        """
X_reduced = pca.transform(X_scaled)
X_rebuilt = pca.inverse_transform(X_reduced)

mse = ((X_scaled - X_rebuilt) ** 2).mean()
print(f"خطأ إعادة البناء: {mse:.4f}")

# استعمال طريف: الصفوف ذات خطأ إعادة بناء عالٍ = مرشّحة لتكون شاذّة
recon_err = ((X_scaled - X_rebuilt) ** 2).sum(axis=1)
outlier_candidates = X.index[recon_err > np.percentile(recon_err, 99)]
"""
    )

    st.subheader("٤. بدائل حسب الحالة", divider="violet")
    ui.code(
        """
from sklearn.decomposition import TruncatedSVD, KernelPCA, NMF
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

TruncatedSVD(n_components=100)               # نصوص TF-IDF (مصفوفة متناثرة)
KernelPCA(n_components=10, kernel="rbf")     # بنية غير خطّية
NMF(n_components=10)                          # عوامل غير سالبة قابلة للتفسير
PLSRegression(n_components=5)                 # مُوجَّه بالهدف (انحدار)
LinearDiscriminantAnalysis(n_components=2)    # مُوجَّه بالهدف (تصنيف)
TSNE(n_components=2, perplexity=30)           # تصوير فقط — لا transform للبيانات الجديدة
""",
        caption=":red-badge[انتبه] `TSNE` و `UMAP` لا يملكان `transform` مستقرًّا "
        "لبيانات جديدة ⇒ لا يصلحان كخطوة في Pipeline إنتاجي.",
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("أ. التباين المفسَّر مكوّنًا بعد مكوّن", divider="violet")
    st.markdown(
        "بياناتنا فيها أعمدة مترابطة عمدًا (`income` و `income_eur`، "
        "`experience` و `seniority`) — وهي الحالة التي يتألّق فيها PCA: "
        "تسعة أعمدة تختزل في عدد أقل بكثير."
    )

    evr, scores, load = pca_fit()

    def _render_evr(frame: int) -> None:
        row = evr.iloc[frame]
        k1, k2, k3 = st.columns(3)
        k1.metric("المكوّن", f"PC{int(row['component'])}")
        k2.metric("يفسّر وحده", f"{row['explained']:.1%}")
        k3.metric("التراكمي", f"{row['cumulative']:.1%}")

        sofar = evr.iloc[: frame + 1]
        bars = (
            alt.Chart(sofar)
            .mark_bar(opacity=0.75)
            .encode(
                x=alt.X("component:O", title="principal component"),
                y=alt.Y("explained:Q", title="explained variance ratio", axis=alt.Axis(format="%")),
                tooltip=["component", "explained", "cumulative"],
            )
        )
        line = (
            alt.Chart(sofar)
            .mark_line(point=True, color="#E8590C", strokeWidth=3)
            .encode(x=alt.X("component:O"), y=alt.Y("cumulative:Q", axis=alt.Axis(format="%")))
        )
        st.altair_chart((bars + line).properties(height=320), width="stretch")

        if row["cumulative"] >= 0.95:
            st.markdown(
                f":green-badge[وصلنا 95%] {int(row['component'])} مكوّنات تكفي "
                f"لتمثيل {row['cumulative']:.1%} من تباين {len(evr)} أعمدة."
            )
        else:
            st.markdown(f":blue-badge[ما زلنا عند {row['cumulative']:.1%}] نحتاج مكوّنات إضافية.")

    play_animation("pca_evr", len(evr), _render_evr, fps=1.0, label="المكوّن")

    st.subheader("ب. الإسقاط ثنائي البعد والأوزان", divider="orange")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("###### البيانات على أول مكوّنين")
        st.altair_chart(
            charts.scatter(scores, "PC1", "PC2", color="churn", height=320),
            width="stretch",
        )
        st.caption(
            "لاحظ: PCA لم ينظر إلى `churn` إطلاقًا. إن ظهر فصل بين اللونين "
            "فهو مصادفة سعيدة، لا هدفًا للطريقة."
        )
    with c2:
        st.markdown("###### أوزان الأعمدة في كل مكوّن (loadings)")
        long = load.melt(id_vars="feature", var_name="component", value_name="loading")
        st.altair_chart(
            charts.heat(long, "component", "feature", "loading", height=320),
            width="stretch",
        )
        st.caption(
            "الخلايا الداكنة = وزن كبير. هكذا تفهم «ماذا يمثّل» كل مكوّن — "
            "وهذا أقصى ما يمكن استرجاعه من قابلية التفسير المفقودة."
        )

    st.subheader("ج. وهل PCA يحسّن الأداء دائمًا؟", divider="red")
    perf = pca_performance()
    base_line = float(perf["all_features"].iloc[0])
    ch = (
        alt.Chart(perf)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("k:Q", title="number of principal components"),
            y=alt.Y("roc_auc:Q", title="cross-validated ROC-AUC", scale=alt.Scale(zero=False)),
            tooltip=["k", "roc_auc"],
        )
        .properties(height=300)
    )
    st.altair_chart(ch + charts.hrule(base_line, color="#0CA678"), width="stretch")
    st.caption(
        f"الخط الأخضر = الأداء بكل الأعمدة الأصلية بلا PCA ({base_line:.3f}). "
        "في كثير من الحالات لا يتفوّق PCA على استعمال الأعمدة كما هي — "
        "لأنه غير مُوجَّه بالهدف. استعمله حين تحتاج **ضغطًا** أو **إزالة "
        "ترابط** أو **تصويرًا**، لا لمجرّد «تحسين الدقّة»."
    )

# ============================================================== القرار
with t_decide:
    st.subheader("متى أستعمل تقليل الأبعاد؟", divider="orange")
    ui.decision_table(
        [
            ("أعمدة كثيرة ومترابطة بشدّة", ":green-badge[PCA]", "يزيل الترابط ويضغط المعلومة"),
            ("تعدّد خطّي يكسر الانحدار", ":green-badge[PCA أو حذف يدوي]", "المكوّنات متعامدة بالتعريف"),
            ("تريد تصوير بيانات عالية الأبعاد", ":violet-badge[PCA ثم t-SNE/UMAP]", "للعرض البصري فقط"),
            ("نصوص TF-IDF متناثرة", ":blue-badge[TruncatedSVD]", "لا يتطلّب توسيط المصفوفة"),
            ("الغاية التنبّؤ والبيانات ليست ضخمة", ":orange-badge[جرّب بدون PCA أولًا]", "غالبًا لا يتفوّق، وتخسر التفسير"),
            ("تحتاج شرح النموذج لصاحب قرار", ":red-badge[تجنّب PCA]", "«المكوّن الثاني» لا يعني شيئًا لأحد"),
            ("بنية غير خطّية واضحة", ":blue-badge[KernelPCA أو Autoencoder]", "PCA خطّي بطبيعته"),
            ("تريد تقليلًا مُوجَّهًا بالهدف", ":green-badge[PLS أو LDA]", "يأخذ `y` بالحسبان بخلاف PCA"),
            ("نموذج شجري", ":red-badge[نادرًا ما يفيد]", "PCA يدوّر المحاور فيصعّب التقسيمات المحاذية للمحاور"),
        ]
    )

    ui.danger(
        "`PCA` خطوة مُتعلَّمة بالكامل: المتوسطات والمحاور تُقدَّر من البيانات. "
        "مكانها **داخل `Pipeline` بعد التقسيم** — ومعها `StandardScaler` قبلها."
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("scikit-learn — Decomposing signals in components (PCA)", "https://scikit-learn.org/stable/modules/decomposition.html"),
            ("scikit-learn — PCA", "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html"),
            ("scikit-learn — Importance of Feature Scaling (PCA)", "https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html"),
            ("scikit-learn — Manifold learning (t-SNE)", "https://scikit-learn.org/stable/modules/manifold.html"),
            ("scikit-learn — Cross decomposition (PLS)", "https://scikit-learn.org/stable/modules/cross_decomposition.html"),
            ("Jolliffe & Cadima (2016) — Principal component analysis: a review", "https://doi.org/10.1098/rsta.2015.0202"),
            ("UMAP — documentation", "https://umap-learn.readthedocs.io/"),
        ]
    )
    ui.terms(
        [
            ("Principal Component Analysis", "تحليل المكوّنات الرئيسية"),
            ("Explained variance ratio", "نسبة التباين المفسَّر"),
            ("Eigenvalue / eigenvector", "قيمة ذاتية / متّجه ذاتي"),
            ("Loadings", "أوزان الأعمدة في المكوّن"),
            ("Scree plot", "منحنى القيم الذاتية"),
            ("Reconstruction error", "خطأ إعادة البناء"),
            ("Supervised reduction", "تقليل مُوجَّه بالهدف (PLS/LDA)"),
        ]
    )
