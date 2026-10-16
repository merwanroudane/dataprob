"""الكشف متعدّد المتغيّرات عن القيم الشاذة."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from core import ui
from core.anim import play_animation

ui.page_header(
    "الكشف متعدّد المتغيّرات (Multivariate detection)",
    "هناك صفوف **لا يمكن** لأي فحص عمودي أن يكشفها: كل قيمة فيها طبيعية، "
    "لكن **التركيبة** مستحيلة. هذه الصفحة عن الطرق التي تنظر إلى الأعمدة معًا.",
    icon=":material/blur_on:",
    badges=[
        ":blue-badge[Mahalanobis]",
        ":green-badge[LOF]",
        ":orange-badge[IsolationForest]",
        ":violet-badge[DBSCAN]",
        ":red-badge[One-Class SVM]",
    ],
)


@st.cache_data(ttl=1800, max_entries=4)
def make_2d(n: int = 320) -> pd.DataFrame:
    """سحابة مرتبطة + ثلاثة أنواع من الشواذ المزروعة."""
    rng = np.random.default_rng(42)
    x = rng.normal(50, 9, n)
    y = 1.8 * x + rng.normal(0, 7, n)
    df = pd.DataFrame({"x": x, "y": y, "truth": "normal"})

    global_out = pd.DataFrame({"x": [110, 112, -5], "y": [205, 30, 95], "truth": "outlier"})
    multi_out = pd.DataFrame({"x": [62, 38, 57, 43], "y": [55, 145, 62, 140], "truth": "outlier"})
    cluster_out = pd.DataFrame(
        {"x": rng.normal(78, 1.3, 10), "y": rng.normal(80, 1.3, 10), "truth": "outlier"}
    )
    return pd.concat([df, global_out, multi_out, cluster_out], ignore_index=True)


@st.cache_data(ttl=1800, max_entries=16)
def detect_all(contamination: float = 0.06) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = make_2d()
    X = d[["x", "y"]].to_numpy()
    Xs = StandardScaler().fit_transform(X)
    truth = (d["truth"] == "outlier").to_numpy()

    preds: dict[str, np.ndarray] = {}

    # 1) Mahalanobis الكلاسيكي
    mu, cov = X.mean(axis=0), np.cov(X, rowvar=False)
    inv = np.linalg.pinv(cov)
    diff = X - mu
    md = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))
    cut = np.sqrt(chi2.ppf(0.975, df=2))
    preds["Mahalanobis (classic)"] = md > cut

    # 2) Mahalanobis أمتن (MCD)
    mcd = MinCovDet(random_state=0).fit(X)
    rd = np.sqrt(mcd.mahalanobis(X))
    preds["Mahalanobis (MCD, robust)"] = rd > cut

    # 3) EllipticEnvelope
    preds["EllipticEnvelope"] = EllipticEnvelope(
        contamination=contamination, random_state=0
    ).fit_predict(X) == -1

    # 4) Isolation Forest
    preds["IsolationForest"] = IsolationForest(
        contamination=contamination, random_state=0, n_estimators=200
    ).fit_predict(X) == -1

    # 5) Local Outlier Factor
    preds["LocalOutlierFactor"] = LocalOutlierFactor(
        n_neighbors=20, contamination=contamination
    ).fit_predict(Xs) == -1

    # 6) One-Class SVM
    preds["OneClassSVM (RBF)"] = OneClassSVM(nu=contamination, gamma="scale").fit_predict(Xs) == -1

    # 7) DBSCAN: النقاط التي لا تنتمي لأي عنقود
    preds["DBSCAN (noise)"] = DBSCAN(eps=0.45, min_samples=8).fit_predict(Xs) == -1

    rows = []
    for name, p in preds.items():
        tp = int((p & truth).sum())
        fp = int((p & ~truth).sum())
        fn = int((~p & truth).sum())
        rows.append(
            {
                "method": name,
                "flagged": int(p.sum()),
                "caught": tp,
                "missed": fn,
                "false_alarms": fp,
                "precision": tp / max(p.sum(), 1),
                "recall": tp / max(truth.sum(), 1),
            }
        )
    summary = pd.DataFrame(rows)

    long = pd.concat(
        [
            d.assign(method=name, flagged=np.where(p, "flagged", "kept"))
            for name, p in preds.items()
        ],
        ignore_index=True,
    )
    return summary, long


t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. لماذا لا يكفي الفحص عمودًا عمودًا؟", divider="blue")
    st.markdown(
        """
عمود `age` مداه 18–80، وعمود `experience_years` مداه 0–45. صفّ فيه
`age = 20` و `experience_years = 19` **داخل المدى في كلا العمودين**،
ولن يرفع أي z-score أي علم. لكنه مستحيل: بدأ العمل في سنّ الواحدة.

الطرق متعدّدة المتغيّرات تنظر إلى **بنية الارتباط** بين الأعمدة، فترى أن
هذا الصف خارج «سحابة» البيانات حتى لو كان داخل كل مدى على حدة.
"""
    )

    st.subheader("٢. مسافة ماهالانوبيس", divider="green")
    st.latex(r"D_M(x) = \sqrt{(x - \mu)^{\top}\, \Sigma^{-1} \,(x - \mu)}")
    st.markdown(
        """
- تعميم لـ z-score على عدّة أبعاد: تقيس البعد عن المركز **بعد تعديل**
  التباينات والارتباطات (أي بوحدات الانحراف المعياري في اتجاه السحابة).
- تحت فرض التوزيع الطبيعي متعدّد المتغيّرات: $D_M^2 \\sim \\chi^2_p$،
  فالعتبة الشائعة $\\chi^2_{p,\\,0.975}$.
- **العيب نفسه الذي في z-score:** $\\mu$ و $\\Sigma$ تُحسبان من بيانات ملوّثة
  ⇒ تقنيع. **الحل:** تقدير أمتن بـ **MCD** (Minimum Covariance Determinant)
  عبر `MinCovDet` أو `EllipticEnvelope`.
"""
    )

    st.subheader("٣. الطرق القائمة على التعلّم", divider="orange")
    ui.concept_grid(
        [
            (
                "Isolation Forest",
                "يقسّم الفضاء بتقسيمات عشوائية ويقيس **كم تقسيمًا لزم لعزل "
                "النقطة**. الشاذّة تُعزل بسرعة ⇒ عمق قليل.\n\n"
                ":green[مزايا] سريع جدًا، خطّي في عدد الصفوف، لا يفترض توزيعًا، "
                "يعمل في أبعاد عالية.\n"
                ":red[حذر] يميل إلى الشواذ العامة أكثر من المحلية.",
                "orange",
            ),
            (
                "Local Outlier Factor (LOF)",
                "يقارن **كثافة** النقطة بكثافة جيرانها. النقطة التي كثافتها أقل "
                "بكثير من جيرانها = شاذّة **محلية**.\n\n"
                ":green[مزايا] يكتشف شواذ داخل مناطق كثيفة (ما تفشل فيه الطرق "
                "العامة).\n"
                ":red[حذر] يتطلّب تقييسًا، وحسّاس لـ `n_neighbors`.",
                "green",
            ),
            (
                "One-Class SVM",
                "يرسم حدودًا حول المنطقة «الطبيعية» في فضاء النواة (kernel).\n\n"
                ":green[مزايا] مرن مع أشكال معقّدة؛ مناسب لـ novelty detection.\n"
                ":red[حذر] بطيء ($O(n^2)$–$O(n^3)$)، وحسّاس جدًا لـ `nu` و `gamma`.",
                "red",
            ),
            (
                "DBSCAN",
                "تجميع قائم على الكثافة؛ ما لا ينتمي لأي عنقود يُسمّى **noise** "
                "(التسمية `-1`).\n\n"
                ":green[مزايا] يكتشف الشواذ الجماعية (العناقيد الصغيرة الغريبة).\n"
                ":red[حذر] اختيار `eps` صعب، ويتدهور في الأبعاد العالية.",
                "violet",
            ),
        ],
        n_cols=2,
    )

    st.subheader("٤. ثلاثة قيود يجب أن تعرفها", divider="red")
    st.markdown(
        """
1. **معامل التلوّث (`contamination`)**: معظم المقدّرات تطلب منك أن تخبرها
   **كم نسبة الشواذ المتوقّعة**. إن وضعت 5٪ فستُعلَّم 5٪ تقريبًا — حتى لو
   كانت بياناتك نظيفة تمامًا. هذا الخيار **فرضية منك، لا اكتشاف منها**.
2. **لعنة الأبعاد (curse of dimensionality)**: في الأبعاد العالية تتقارب كل
   المسافات، فيفقد مفهوم «البعيد» معناه. LOF و KNN يتدهوران؛ IsolationForest
   أمتن نسبيًا؛ وقد تحتاج PCA أو Autoencoder أولًا.
3. **التقييس إجباري** لكل ما يعتمد على المسافة (LOF، OCSVM، DBSCAN، KNN).
   عمود بالدينار (آلاف) وآخر نسبة (0–1) ⇒ الأول يبتلع المسافة كلّها.
   :green-badge[استثناء] `IsolationForest` و `Mahalanobis` لا يحتاجان تقييسًا.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. ماهالانوبيس (الكلاسيكي والأمتن)", divider="blue")
    ui.code(
        """
import numpy as np
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

X = df[["age", "experience_years", "income"]].dropna().to_numpy()

# الكلاسيكي
mu, cov = X.mean(0), np.cov(X, rowvar=False)
d2 = np.einsum("ij,jk,ik->i", X - mu, np.linalg.pinv(cov), X - mu)

# الأمتن (مُستحسن): لا يتلوّث بالشواذ نفسها
mcd = MinCovDet(random_state=0).fit(X)
d2_robust = mcd.mahalanobis(X)

cut = chi2.ppf(0.975, df=X.shape[1])
print("classic:", (d2 > cut).sum(), " robust:", (d2_robust > cut).sum())
"""
    )

    st.subheader("٢. مقدّرات scikit-learn", divider="green")
    ui.code(
        """
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(X)      # إجباري للطرق المسافية

iso = IsolationForest(contamination=0.05, n_estimators=300, random_state=0)
pred_iso = iso.fit_predict(X)               # -1 = شاذّة، +1 = طبيعية
score_iso = iso.decision_function(X)        # كلما قلّ كان أشذّ

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
pred_lof = lof.fit_predict(Xs)              # لا يملك predict إلا بـ novelty=True
score_lof = lof.negative_outlier_factor_

ocsvm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale").fit(Xs)
env = EllipticEnvelope(contamination=0.05, random_state=0).fit(X)
""",
        caption="`fit_predict` يرجع `-1` للشاذّة و `+1` للطبيعية. "
        "استعمل `decision_function` لتحصل على **درجة** بدل حكم ثنائي — "
        "فتختار العتبة بنفسك.",
    )

    st.subheader("٣. كشف الجديد في الإنتاج (Novelty detection)", divider="orange")
    ui.code(
        """
# درّب على بيانات نظيفة فقط، ثم راقب الوافد الجديد
lof_nov = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X_clean)
is_new_anomaly = lof_nov.predict(X_new) == -1

ocsvm = OneClassSVM(nu=0.01, gamma="scale").fit(X_clean)
drift_alarm = (ocsvm.predict(X_new) == -1).mean()
print(f"{drift_alarm:.1%} من الوافد الجديد خارج النطاق المعروف")
""",
        caption="هذه هي الطريقة العملية لمراقبة **انزياح البيانات (data drift)** "
        "بعد نشر النموذج.",
    )

    st.subheader("٤. اجمع الأدلّة بدل الاعتماد على طريقة واحدة", divider="violet")
    ui.code(
        """
import pandas as pd

flags = pd.DataFrame({
    "iso": iso.fit_predict(X) == -1,
    "lof": lof.fit_predict(Xs) == -1,
    "mcd": mcd.mahalanobis(X) > cut,
})
votes = flags.sum(axis=1)
suspects = df.loc[votes >= 2]        # اتفقت طريقتان على الأقل
print(len(suspects), "صفًا يستحق فحصًا يدويًا")
"""
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("سبع طرق على نفس البيانات", divider="violet")
    st.markdown(
        "البيانات: سحابة مرتبطة (`y ≈ 1.8x`) + **٣ شواذ عامة** + "
        "**٤ شواذ متعدّدة المتغيّرات** (داخل المدى لكن خارج العلاقة) + "
        "**عنقود شاذ من ١٠ نقاط**. الأحمر = ما علّمته الطريقة."
    )

    cont = st.select_slider(
        "contamination المفترضة",
        options=[0.03, 0.05, 0.06, 0.08, 0.12],
        value=0.06,
        format_func=lambda v: f"{v:.0%}",
        key="multi_cont",
    )
    summary, long = detect_all(cont)
    methods = summary["method"].tolist()

    def _render_method(frame: int) -> None:
        name = methods[frame]
        sub = long[long["method"] == name]
        row = summary.iloc[frame]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("مُعلَّمة", int(row["flagged"]))
        k2.metric("أمسكت صحيحًا", int(row["caught"]), help="من أصل 17 شاذّة مزروعة")
        k3.metric("فاتها", int(row["missed"]), delta_color="inverse")
        k4.metric("إنذارات كاذبة", int(row["false_alarms"]), delta_color="inverse")

        ch = (
            alt.Chart(sub)
            .mark_circle(size=90, opacity=0.8)
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(zero=False), title="x"),
                y=alt.Y("y:Q", scale=alt.Scale(zero=False), title="y"),
                color=alt.Color(
                    "flagged:N",
                    scale=alt.Scale(domain=["kept", "flagged"], range=["#4C6EF5", "#E8590C"]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                shape=alt.Shape(
                    "truth:N",
                    scale=alt.Scale(domain=["normal", "outlier"], range=["circle", "triangle-up"]),
                    legend=alt.Legend(orient="top", title="الحقيقة"),
                ),
                tooltip=["x:Q", "y:Q", "truth:N", "flagged:N"],
            )
            .properties(height=380)
        )
        st.altair_chart(ch, width="stretch")

        notes = {
            "Mahalanobis (classic)": "القطع الناقص المقدَّر تضخّم بسبب الشواذ نفسها ⇒ تقنيع واضح.",
            "Mahalanobis (MCD, robust)": "التقدير الأمتن يتجاهل الملوّثات أثناء التقدير ⇒ كشف أدقّ بكثير.",
            "EllipticEnvelope": "يفترض شكلًا بيضويًا (توزيعًا طبيعيًا متعدّد المتغيّرات) — ممتاز هنا، ويفشل مع أشكال معقّدة.",
            "IsolationForest": "لا يفترض شكلًا؛ قوي مع الشواذ العامة، وأضعف قليلًا مع الشواذ المحلية.",
            "LocalOutlierFactor": "قائم على الكثافة المحلية ⇒ الأفضل غالبًا في التقاط الشواذ داخل السحابة.",
            "OneClassSVM (RBF)": "حدود مرنة، لكنه حسّاس جدًا لـ `nu` و `gamma`؛ يميل إلى الإفراط في التعليم.",
            "DBSCAN (noise)": "يرى العنقود الشاذ كعنقود **شرعي** إن كان كبيرًا بما يكفي ⇒ قد يفوته تمامًا. هذا درس مهم.",
        }
        st.markdown(f"**:violet-badge[{name}]** {notes[name]}")

    play_animation(
        f"multi_methods_{cont}",
        len(methods),
        _render_method,
        fps=0.5,
        label="الطريقة",
        frame_labels=methods,
    )

    st.dataframe(
        summary.assign(
            precision=summary["precision"].map("{:.0%}".format),
            recall=summary["recall"].map("{:.0%}".format),
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "غيّر `contamination` وراقب كيف تتغيّر كل الأرقام: هذه ليست حقيقة "
        "موضوعية تكتشفها الخوارزمية، بل **فرضية تفرضها أنت**."
    )

# ============================================================== القرار
with t_decide:
    st.subheader("أي مقدّر أختار؟", divider="orange")
    ui.decision_table(
        [
            ("أبعاد قليلة، توزيع ≈ طبيعي", "Mahalanobis بـ MCD / EllipticEnvelope", "تفسير إحصائي واضح وعتبة χ² مبرَّرة"),
            ("أبعاد كثيرة، بيانات كبيرة", "IsolationForest", "خطّي في n، لا يفترض توزيعًا، لا يحتاج تقييسًا"),
            ("شواذ محلية داخل مناطق كثيفة", "LocalOutlierFactor", "يقارن الكثافة محليًا لا عالميًا"),
            ("كشف الجديد بعد النشر", "OneClassSVM أو LOF(novelty=True)", "يُدرَّب على «الطبيعي» فقط"),
            ("شواذ جماعية / عناقيد غريبة", "DBSCAN + تحليل العناقيد", "يرى المجموعة لا النقطة"),
            ("بيانات مختلطة (رقمي + فئوي)", "Gower distance أو ترميز ثم IsolationForest", "المسافة الإقليدية بلا معنى على الفئات"),
            ("صور / نصوص / إشارات", "Autoencoder أو خطأ إعادة البناء", "بنية غير جدولية تحتاج تمثيلًا متعلَّمًا"),
            ("لا تعرف نسبة التلوّث", "استعمل `decision_function` ورتّب", "افحص أعلى 1٪ يدويًا بدل فرض عتبة"),
        ],
        headers=("الحالة", "المقدّر", "السبب"),
    )

    ui.danger(
        "كل هذه الطرق تُدرَّب (`fit`) على **التدريب فقط**. تطبيق كشف الشواذ "
        "على كامل البيانات قبل التقسيم = تسريب. وفي الإنتاج، احتفظ بالمقدّر "
        "المُدرَّب لتطبيقه (`predict`) على الوافد الجديد."
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("scikit-learn — Novelty and Outlier Detection (دليل المستخدم)", "https://scikit-learn.org/stable/modules/outlier_detection.html"),
            ("scikit-learn — Comparing anomaly detection algorithms", "https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html"),
            ("scikit-learn — Robust covariance estimation (MinCovDet)", "https://scikit-learn.org/stable/modules/covariance.html#robust-covariance-estimation"),
            ("Liu, Ting & Zhou (2008) — Isolation Forest", "https://doi.org/10.1109/ICDM.2008.17"),
            ("Breunig et al. (2000) — LOF: Identifying Density-Based Local Outliers", "https://doi.org/10.1145/335191.335388"),
            ("Rousseeuw & Driessen (1999) — Fast MCD Estimator", "https://doi.org/10.1080/00401706.1999.10485670"),
        ]
    )
    ui.terms(
        [
            ("Mahalanobis distance", "مسافة ماهالانوبيس"),
            ("MCD", "محدّد التغاير الأدنى (تقدير أمتن)"),
            ("Contamination", "نسبة التلوّث المفترضة"),
            ("Local density", "الكثافة المحلية"),
            ("Curse of dimensionality", "لعنة الأبعاد"),
            ("decision_function", "درجة الشذوذ المستمرّة"),
            ("Novelty detection", "كشف الأنماط الجديدة"),
        ]
    )
