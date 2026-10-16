"""التقييس Scaling: الطرق وأثرها على الخوارزميات."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC

from core import charts, ui
from core.anim import play_animation
from core.datasets import clean_customers

ui.page_header(
    "التقييس Scaling",
    "عمود الدخل بعشرات الآلاف وعمود الرضا من 1 إلى 10. أي خوارزمية تحسب "
    "**مسافة** ستظنّ أن الدخل هو كل شيء. التقييس يضع الأعمدة على مقياس "
    "واحد — وهو شرط لا غنى عنه لعائلة كاملة من النماذج.",
    icon=":material/straighten:",
    badges=[
        ":blue-badge[Standard]",
        ":green-badge[MinMax]",
        ":orange-badge[Robust]",
        ":violet-badge[Quantile]",
        ":red-badge[Trees don't care]",
    ],
)

SCALERS = [
    ("لا تقييس (الأصل)", None),
    ("StandardScaler", StandardScaler()),
    ("MinMaxScaler", MinMaxScaler()),
    ("RobustScaler", RobustScaler()),
    ("MaxAbsScaler", MaxAbsScaler()),
    ("PowerTransformer (Yeo-Johnson)", PowerTransformer(method="yeo-johnson")),
    ("QuantileTransformer (normal)", QuantileTransformer(output_distribution="normal", n_quantiles=300, random_state=0)),
]


@st.cache_data(ttl=1800, max_entries=8)
def scaled_frames(with_outliers: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = clean_customers(400).copy()
    v = df[["income", "satisfaction"]].to_numpy(dtype=float)
    if with_outliers:
        v = np.vstack([v, np.array([[v[:, 0].max() * 9, 9.5], [v[:, 0].max() * 7, 2.0]])])

    frames, stats = [], []
    for name, sc in SCALERS:
        out = v if sc is None else sc.fit_transform(v)
        frames.append(
            pd.DataFrame({"income": out[:, 0], "satisfaction": out[:, 1], "scaler": name})
        )
        stats.append(
            {
                "scaler": name,
                "income_min": out[:, 0].min(),
                "income_max": out[:, 0].max(),
                "income_mean": out[:, 0].mean(),
                "income_std": out[:, 0].std(),
                "range_ratio": (out[:, 0].max() - out[:, 0].min())
                / max(out[:, 1].max() - out[:, 1].min(), 1e-9),
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(stats)


@st.cache_data(ttl=1800, max_entries=4, show_spinner="جارٍ تدريب النماذج…")
def scaling_effect() -> pd.DataFrame:
    df = clean_customers(600)
    X = df[["age", "experience_years", "income", "spending", "satisfaction"]]
    y = df["churn"]
    models = {
        "KNN (k=15)": KNeighborsClassifier(n_neighbors=15),
        "SVM (RBF)": SVC(),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=0),
    }
    rows = []
    for mname, m in models.items():
        for sname, sc in (("بدون تقييس", None), ("StandardScaler", StandardScaler())):
            pipe = m if sc is None else Pipeline([("sc", sc), ("m", m)])
            s = cross_val_score(pipe, X, y, cv=4, scoring="roc_auc")
            rows.append({"model": mname, "setup": sname, "roc_auc": float(s.mean())})
    out = pd.DataFrame(rows)
    piv = out.pivot(index="model", columns="setup", values="roc_auc").reset_index()
    piv["gain"] = piv["StandardScaler"] - piv["بدون تقييس"]
    return piv


t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. لماذا نقيّس أصلًا؟", divider="blue")
    st.markdown(
        """
المسافة الإقليدية بين صفّين:
"""
    )
    st.latex(r"d(a,b) = \sqrt{(income_a - income_b)^2 + (satisfaction_a - satisfaction_b)^2}")
    st.markdown(
        """
فرق 10,000 دينار في الدخل يعطي $10^8$ داخل الجذر، بينما فرق 9 نقاط في الرضا
يعطي 81. أي أن **الرضا غير موجود عمليًا** في الحساب. هذا هو السبب الأول.

الأسباب الأخرى:
- **تقارب النزول التدرّجي (gradient descent):** مقاييس متباينة تجعل سطح
  الخسارة بيضويًا ممدودًا، فيتذبذب التحسين ويبطؤ كثيرًا.
- **عدالة التنظيم (regularization):** عقوبة Ridge/Lasso تعاقب المعاملات
  الكبيرة؛ ومعامل العمود ذي الوحدات الصغيرة يكون كبيرًا رقميًا، فيُعاقَب
  ظلمًا. **التقييس قبل التنظيم إلزامي.**
- **PCA:** يبحث عن اتجاه أكبر تباين. بلا تقييس، العمود ذو الوحدة الأكبر
  يحتكر المكوّن الأول لمجرّد وحدته.
- **الشبكات العصبية:** دوال التفعيل تتشبّع عند المدخلات الكبيرة.
"""
    )

    st.subheader("٢. الطرق السبع", divider="green")
    st.markdown(
        """
| المحوّل | الصيغة | المدى الناتج | مع الشواذ |
|---|---|---|---|
| `StandardScaler` | $(x-\\mu)/\\sigma$ | غير محدود، متوسط 0 وتباين 1 | :red[حسّاس جدًا] |
| `MinMaxScaler` | $(x-\\min)/(\\max-\\min)$ | $[0,1]$ | :red[كارثي: شاذّة واحدة تضغط الباقي] |
| `RobustScaler` | $(x-\\tilde{x})/IQR$ | غير محدود | :green[أمتن — يستعمل الوسيط و IQR] |
| `MaxAbsScaler` | $x/\\max|x|$ | $[-1,1]$ | :orange[حسّاس، لكنه يحافظ على التناثر (sparsity)] |
| `Normalizer` | $x/\\|x\\|$ **لكل صف** | طول الصف = 1 | :orange[ليس تقييس أعمدة أصلًا] |
| `QuantileTransformer` | رتبة ← توزيع هدف | $[0,1]$ أو طبيعي | :green[يبتلع الشواذ تمامًا — غير خطّي] |
| `PowerTransformer` | Box-Cox / Yeo-Johnson | متوسط 0 وتباين 1 | :green[يقلّل الانحراف والشواذ معًا] |

:red-badge[تنبيه شائع] `Normalizer` يعمل على **الصفوف** لا الأعمدة. هو لتوحيد
أطوال المتجهات (نصوص، TF-IDF)، وليس بديلًا عن `StandardScaler`.
"""
    )

    st.subheader("٣. من يحتاج التقييس ومن لا يحتاجه؟", divider="orange")
    ui.concept_grid(
        [
            (
                ":red-badge[إلزامي]",
                "- KNN, K-Means, DBSCAN, SVM (كل ما يحسب مسافة)\n"
                "- Ridge, Lasso, ElasticNet (التنظيم)\n"
                "- الشبكات العصبية\n"
                "- PCA, LDA, تحليل العوامل\n"
                "- أي نزول تدرّجي (SGD)",
                "red",
            ),
            (
                ":green-badge[غير ضروري]",
                "- DecisionTree, RandomForest, ExtraTrees\n"
                "- GradientBoosting, XGBoost, LightGBM, CatBoost\n"
                "- Naive Bayes (الغاوسي يقدّر لكل عمود على حدة)\n\n"
                "السبب: التقسيم يعتمد على **ترتيب** القيم، والترتيب لا يتغيّر "
                "بأي تحويل رتيب (monotonic).",
                "green",
            ),
            (
                ":orange-badge[يعتمد]",
                "- `LogisticRegression` بدون تنظيم: اختياري، لكنه يسرّع التقارب "
                "ويجعل المعاملات قابلة للمقارنة.\n"
                "- الانحدار الخطّي البسيط: لا يغيّر التنبّؤات، لكنه يغيّر تفسير "
                "المعاملات.",
                "orange",
            ),
        ],
        n_cols=3,
    )

    ui.beginner_note(
        """
التقييس **لا يغيّر شكل** التوزيع ولا ترتيب القيم ولا يحذف شيئًا: إنه مجرّد
تغيير وحدة قياس، مثل التحويل من سنتيمتر إلى متر. أمّا `PowerTransformer` و
`QuantileTransformer` فهما **يغيّران الشكل** فعلًا — وهذا موضوع الدرس التالي.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. الاستعمال الصحيح — داخل Pipeline", divider="blue")
    ui.code(
        """
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression()),
])
pipe.fit(X_train, y_train)        # المتوسط والانحراف من التدريب فقط
pipe.score(X_test, y_test)        # تُطبَّق نفس المعاملات على الاختبار

# يدويًا (إن اضطررت)
sc = StandardScaler().fit(X_train)
X_train_s, X_test_s = sc.transform(X_train), sc.transform(X_test)
print(sc.mean_, sc.scale_)        # المعاملات المُتعلَّمة
""",
        caption=":red-badge[خطأ قاتل] `StandardScaler().fit_transform(X_test)` — "
        "يعيد تعلّم معاملات من الاختبار فتختلف عن التدريب، وتُفسَد التنبّؤات.",
    )

    st.subheader("٢. تقييس مختلف لكل مجموعة أعمدة", divider="green")
    ui.code(
        """
from sklearn.compose import ColumnTransformer

pre = ColumnTransformer([
    ("skewed",  PowerTransformer(method="yeo-johnson"), ["income", "spending"]),
    ("normal",  StandardScaler(),                        ["age", "satisfaction"]),
    ("bounded", MinMaxScaler(),                          ["ratio"]),
    ("binary",  "passthrough",                           ["is_subscribed"]),
], remainder="drop")
""",
        caption="لا يوجد مقيّاس واحد يناسب كل الأعمدة — اختر حسب شكل كل عمود.",
    )

    st.subheader("٣. مخرجات pandas بدل مصفوفات numpy", divider="orange")
    ui.code(
        """
# يحافظ على أسماء الأعمدة — يسهّل التصحيح والتفسير كثيرًا
pre.set_output(transform="pandas")
X_t = pre.fit_transform(X_train)
print(X_t.head())

# أو عالميًا
from sklearn import set_config
set_config(transform_output="pandas")
"""
    )

    st.subheader("٤. التقييس العكسي", divider="violet")
    ui.code(
        """
# بعد التنبّؤ بقيمة مُقيَّسة، أعدها إلى وحدتها الأصلية
y_pred_original = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

# في الانحدار: قيّس `y` فقط إن احتجت، واحفظ المقيّاس
from sklearn.compose import TransformedTargetRegressor
model = TransformedTargetRegressor(
    regressor=Ridge(), transformer=StandardScaler()
)   # يتكفّل بالعكس آليًا
"""
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("نفس البيانات تحت سبعة مقاييس", divider="violet")
    with_out = st.toggle(
        "أضف قيمتين شاذّتين في الدخل",
        value=True,
        help="فعّله ثم شغّل الأنيميشن: سترى أي المقاييس ينهار وأيها يصمد",
        key="sc_out",
    )
    frames, stats = scaled_frames(with_out)

    def _render_scaler(frame: int) -> None:
        name = SCALERS[frame][0]
        sub = frames[frames["scaler"] == name]
        row = stats.iloc[frame]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("أدنى قيمة (income)", f"{row['income_min']:,.2f}")
        k2.metric("أعلى قيمة (income)", f"{row['income_max']:,.2f}")
        k3.metric("المتوسط / الانحراف", f"{row['income_mean']:.2f} / {row['income_std']:.2f}")
        k4.metric(
            "نسبة المدى income:satisfaction",
            f"{row['range_ratio']:,.1f}×",
            help="كلما اقتربت من 1 كان العمودان متكافئين في حساب المسافة",
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("###### توزيع `income` بعد التحويل")
            st.altair_chart(charts.hist(sub, "income", bins=45, height=280), width="stretch")
        with c2:
            st.markdown("###### الفضاء ثنائي البعد (كما تراه الخوارزمية)")
            st.altair_chart(
                charts.scatter(sub, "income", "satisfaction", height=280, tooltip=["income", "satisfaction"]),
                width="stretch",
            )

        notes = {
            "لا تقييس (الأصل)": "الدخل يبتلع المسافة تمامًا؛ الرضا كأنه غير موجود.",
            "StandardScaler": "متوسط 0 وتباين 1. :red[لكن] الشاذّة تضخّم الانحراف فتُضغط بقيّة القيم في نطاق ضيّق.",
            "MinMaxScaler": ":red[الأسوأ مع الشواذ]: القيمة القصوى تصبح 1، وكل البيانات الحقيقية تتكدّس قرب الصفر.",
            "RobustScaler": "يستعمل الوسيط و IQR ⇒ الشواذ تخرج عن المدى لكنها **لا تشوّه** بقيّة القيم. :green[الخيار الافتراضي عند وجود شواذ].",
            "MaxAbsScaler": "يقسم على أكبر قيمة مطلقة؛ يحافظ على الأصفار (مفيد للمصفوفات المتناثرة من النصوص).",
            "PowerTransformer (Yeo-Johnson)": "يغيّر **شكل** التوزيع نحو الطبيعي ويقلّل الانحراف — ليس تقييسًا خطّيًا.",
            "QuantileTransformer (normal)": "يحوّل الرتب إلى توزيع طبيعي: يبتلع الشواذ كليًا. :orange[حذر] غير خطّي ويشوّه المسافات الأصلية؛ ويحتاج عيّنة كبيرة.",
        }
        st.markdown(f"**:violet-badge[{name}]** {notes[name]}")

    play_animation(
        f"scaler_tour_{with_out}",
        len(SCALERS),
        _render_scaler,
        fps=0.5,
        label="المقيّاس",
        frame_labels=[s[0] for s in SCALERS],
    )

    st.subheader("هل يغيّر التقييس الأداء فعلًا؟", divider="orange")
    eff = scaling_effect()
    st.dataframe(eff.round(3), width="stretch", hide_index=True)
    melted = eff.melt(
        id_vars="model", value_vars=["بدون تقييس", "StandardScaler"], var_name="setup", value_name="roc_auc"
    )
    st.altair_chart(
        alt.Chart(melted)
        .mark_bar()
        .encode(
            y=alt.Y("model:N", title=None),
            x=alt.X("roc_auc:Q", title="ROC-AUC", scale=alt.Scale(domain=[0.4, 0.85])),
            color=alt.Color("setup:N", legend=alt.Legend(orient="top", title=None)),
            yOffset="setup:N",
            tooltip=["model", "setup", "roc_auc"],
        )
        .properties(height=260),
        width="stretch",
    )
    st.caption(
        "النتيجة المتوقّعة: **KNN و SVM** يقفزان بعد التقييس؛ "
        "**RandomForest** لا يتغيّر تقريبًا (الفارق ضجيج عشوائي)؛ "
        "**LogisticRegression** يستفيد قليلًا وبشكل أساسي في سرعة التقارب."
    )

# ============================================================== القرار
with t_decide:
    st.subheader("أي مقيّاس أختار؟", divider="orange")
    ui.decision_table(
        [
            ("توزيع ≈ طبيعي، بلا شواذ", "StandardScaler", "الخيار الافتراضي المعتاد"),
            ("توجد شواذ حقيقية تريد الاحتفاظ بها", "RobustScaler", "الوسيط و IQR لا يتأثّران بها"),
            ("تحتاج مدى محدودًا [0,1]", "MinMaxScaler", "مطلوب في بعض الشبكات ومعالجة الصور — **بعد** معالجة الشواذ"),
            ("بيانات متناثرة (TF-IDF)", "MaxAbsScaler", "لا يكسر التناثر ولا يحوّل الأصفار"),
            ("توزيع منحرف بشدّة", "PowerTransformer (Yeo-Johnson)", "يصحّح الانحراف ويقيّس في خطوة واحدة"),
            ("توزيع غريب جدًا / متعدّد القمم", "QuantileTransformer", "يفرض شكلًا هدفًا بغضّ النظر عن الأصل"),
            ("نموذج شجري", "لا تقيّس", "لا فائدة، وتخسر قابلية قراءة القيم"),
            ("تريد مقارنة أهمية المعاملات", "StandardScaler", "المعاملات تصبح على وحدة مشتركة"),
            ("توحيد أطوال متّجهات الصفوف", "Normalizer", "عملية على الصفوف لا الأعمدة (نصوص، تشابه جيبي)"),
        ],
        headers=("الحالة", "المقيّاس", "السبب"),
    )

    ui.danger(
        "رتّب الخطوات هكذا: **(١) عالج الفقد ← (٢) قرّر بشأن الشواذ ← "
        "(٣) صحّح الانحراف ← (٤) قيّس.** التقييس قبل معالجة الشواذ يعني أن "
        "الشواذ ستحدّد معاملات التقييس نفسها."
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("scikit-learn — Preprocessing data", "https://scikit-learn.org/stable/modules/preprocessing.html"),
            ("scikit-learn — Compare the effect of different scalers on data with outliers", "https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html"),
            ("scikit-learn — Importance of Feature Scaling", "https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html"),
            ("scikit-learn — StandardScaler", "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html"),
            ("scikit-learn — RobustScaler", "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html"),
            ("scikit-learn — set_output API", "https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html"),
        ]
    )
    ui.terms(
        [
            ("Scaling", "التقييس"),
            ("Standardization", "التوحيد المعياري (z-score)"),
            ("Normalization", "التطبيع (غالبًا MinMax، أو توحيد طول الصف)"),
            ("Sparsity", "التناثر (كثرة الأصفار)"),
            ("Monotonic transform", "تحويل رتيب يحفظ الترتيب"),
            ("inverse_transform", "التحويل العكسي"),
        ]
    )
