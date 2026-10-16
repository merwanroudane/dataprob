"""عدم توازن الفئات (Class imbalance)."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core import charts, ui
from core.anim import play_animation
from core.datasets import imbalanced_fraud

ui.page_header(
    "عدم توازن الفئات (Imbalanced data)",
    "فئة الاحتيال 2٪ من البيانات. نموذج يقول «لا احتيال» دائمًا يحقّق دقّة "
    "**98٪** — وهو عديم الفائدة تمامًا. هذا الدرس عن المقاييس الصحيحة أولًا، "
    "ثم عن العلاج.",
    icon=":material/balance:",
    badges=[
        ":blue-badge[class_weight]",
        ":green-badge[Threshold tuning]",
        ":orange-badge[SMOTE]",
        ":violet-badge[PR-AUC]",
        ":red-badge[Accuracy paradox]",
    ],
)

FEATS = ["amount", "night_ratio", "n_devices"]


# ------------------------------------------------------- SMOTE مبسّط
def simple_smote(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """تنفيذ تعليمي لـ SMOTE: توليد نقاط على القطع الواصلة بين الجيران.

    في الإنتاج استعمل `imblearn.over_sampling.SMOTE` — هذه النسخة للتوضيح
    فقط (بلا معالجة للفئات المتعدّدة ولا للمتغيّرات الفئوية).
    """
    rng = np.random.default_rng(seed)
    minority = X[y == 1]
    n_needed = int((y == 0).sum() - (y == 1).sum())
    if len(minority) < 2 or n_needed <= 0:
        return X, y

    d = ((minority[:, None, :] - minority[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    neigh = np.argsort(d, axis=1)[:, : min(k, len(minority) - 1)]

    base_idx = rng.integers(0, len(minority), n_needed)
    new = []
    for i in base_idx:
        j = neigh[i][rng.integers(0, neigh.shape[1])]
        gap = rng.random()
        new.append(minority[i] + gap * (minority[j] - minority[i]))
    new = np.array(new)
    return np.vstack([X, new]), np.concatenate([y, np.ones(len(new), dtype=int)])


@st.cache_data(ttl=1800, max_entries=8, show_spinner="جارٍ تدريب الاستراتيجيات…")
def compare_strategies(pos_rate: float = 0.02) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = imbalanced_fraud(3000, pos_rate)
    X, y = df[FEATS].to_numpy(float), df["fraud"].to_numpy(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    def evaluate(name, proba, thr=0.5):
        pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
        return {
            "strategy": name,
            "ROC-AUC": roc_auc_score(yte, proba),
            "PR-AUC": average_precision_score(yte, proba),
            "F1": f1_score(yte, pred, zero_division=0),
            "balanced_acc": balanced_accuracy_score(yte, pred),
            "MCC": matthews_corrcoef(yte, pred) if pred.sum() else 0.0,
            "recall": tp / max(tp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "accuracy": (tp + tn) / len(yte),
            "threshold": thr,
        }

    rows = []

    # 0) النموذج الكسول
    rows.append(evaluate("التنبّؤ بالفئة الأغلب دائمًا", np.zeros(len(yte))))

    # 1) أساس بلا معالجة
    base = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]).fit(Xtr, ytr)
    p_base = base.predict_proba(Xte)[:, 1]
    rows.append(evaluate("Logistic بلا معالجة", p_base))

    # 2) class_weight
    cw = Pipeline(
        [("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]
    ).fit(Xtr, ytr)
    rows.append(evaluate("class_weight='balanced'", cw.predict_proba(Xte)[:, 1]))

    # 3) ضبط العتبة على التدريب (بالتحقّق المتقاطع)
    oof = np.zeros(len(ytr))
    for tr_i, va_i in StratifiedKFold(4, shuffle=True, random_state=0).split(Xtr, ytr):
        mdl = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]).fit(Xtr[tr_i], ytr[tr_i])
        oof[va_i] = mdl.predict_proba(Xtr[va_i])[:, 1]
    prec, rec, thrs = precision_recall_curve(ytr, oof)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    best_thr = float(thrs[int(np.nanargmax(f1s[:-1]))])
    rows.append(evaluate(f"ضبط العتبة (thr={best_thr:.3f})", p_base, best_thr))

    # 4) Random undersampling
    rng = np.random.default_rng(0)
    neg_idx = np.where(ytr == 0)[0]
    pos_idx = np.where(ytr == 1)[0]
    keep = rng.choice(neg_idx, size=len(pos_idx) * 3, replace=False)
    sel = np.concatenate([keep, pos_idx])
    us = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]).fit(Xtr[sel], ytr[sel])
    rows.append(evaluate("Random undersampling (3:1)", us.predict_proba(Xte)[:, 1]))

    # 5) SMOTE (على التدريب فقط)
    Xs, ys = simple_smote(Xtr, ytr)
    sm = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]).fit(Xs, ys)
    rows.append(evaluate("SMOTE (تدريب فقط)", sm.predict_proba(Xte)[:, 1]))

    # 6) RandomForest موزون
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced_subsample", random_state=0
    ).fit(Xtr, ytr)
    rows.append(evaluate("RandomForest موزون", rf.predict_proba(Xte)[:, 1]))

    res = pd.DataFrame(rows)

    pr = precision_recall_curve(yte, p_base)
    curve = pd.DataFrame({"precision": pr[0][:-1], "recall": pr[1][:-1], "threshold": pr[2]})
    return res, curve


t_theory, t_code, t_anim, t_decide, t_ref = ui.lesson_tabs()

# ============================================================== النظرية
with t_theory:
    st.subheader("١. مفارقة الدقّة (Accuracy paradox)", divider="red")
    st.markdown(
        """
بفئة موجبة نسبتها 1٪، النموذج الذي يقول «سالب» دائمًا يحقّق **دقّة 99٪**
و **صفر** قيمة. ولهذا:

:red-badge[لا تستعمل `accuracy` مع البيانات غير المتوازنة أبدًا.]

| المقياس | متى يصلح | ملاحظة |
|---|---|---|
| `accuracy` | التوازن فقط | مُضلّل تمامًا هنا |
| `ROC-AUC` | مقارنة عامة | **متفائل** مع الاختلال الشديد لأن المقام يضمّ كل السوالب |
| `PR-AUC` (Average Precision) | :green-badge[الأفضل للفئة النادرة] | يركّز على الفئة الموجبة حصرًا |
| `recall` (الحساسية) | حين يكلّف الفوت غاليًا (مرض، احتيال) | يُقرأ دائمًا مع `precision` |
| `precision` | حين يكلّف الإنذار الكاذب غاليًا | |
| `F1` | توازن بين الاثنين | يتجاهل السوالب الصحيحة |
| `balanced_accuracy` | متوسط الحساسية والنوعية | قراءة متوازنة بسيطة |
| `MCC` | :green-badge[الأمتن إجمالًا] | يستعمل خلايا مصفوفة الالتباس الأربع |
| تكلفة مخصّصة | حين تعرف التكاليف الحقيقية | الأصدق عمليًا: اضرب الأخطاء في ثمنها |
"""
    )

    st.subheader("٢. أرخص علاج وأكثره إهمالًا: العتبة", divider="green")
    st.markdown(
        """
معظم المصنّفات تُخرج **احتمالًا**، ثم تقارنه بعتبة `0.5` لإنتاج القرار.
والعتبة `0.5` ليست مقدّسة إطلاقًا — إنها مجرّد قيمة افتراضية.

خفض العتبة إلى `0.1` يزيد الحساسية (recall) ويخفض الدقّة الموجبة
(precision). هذا **تحريك على منحنى موجود**، لا تغيير للنموذج:

:green-badge[مجاني، لا يشوّه البيانات، ولا يحتاج إعادة تدريب.]

:red-badge[شرط] تُختار العتبة على بيانات **تحقّق** (أو بتحقّق متقاطع على
التدريب)، لا على الاختبار — وإلا فهي شكل آخر من التسريب.
"""
    )

    st.subheader("٣. ترسانة المعالجة", divider="blue")
    ui.concept_grid(
        [
            (
                "class_weight / sample_weight",
                "ضاعف وزن خطأ الفئة النادرة في دالة الخسارة. "
                "`class_weight='balanced'` يضبط الأوزان عكسيًا مع التكرار.\n\n"
                ":green[الأفضل للبدء]: لا يغيّر البيانات ولا يضيف وقت تدريب.",
                "blue",
            ),
            (
                "Random oversampling",
                "كرّر صفوف الفئة النادرة.\n\n"
                ":red[خطر] فرط ملاءمة: النموذج يرى نفس الصف عشرات المرّات "
                "فيحفظه.",
                "orange",
            ),
            (
                "Random undersampling",
                "احذف من الفئة الأغلب.\n\n"
                ":green[مفيد] مع البيانات الضخمة (يسرّع التدريب كثيرًا).\n"
                ":red[خطر] تهدر معلومة حقيقية.",
                "violet",
            ),
            (
                "SMOTE وعائلته",
                "ولّد نقاطًا **اصطناعية** على القطع الواصلة بين جيران الفئة "
                "النادرة. المتغيّرات: Borderline-SMOTE، ADASYN، SMOTE-NC "
                "(للفئوي)، SMOTE-Tomek و SMOTE-ENN (توليد + تنظيف).\n\n"
                ":red[قيود] يحتاج تقييسًا، ويضعف في الأبعاد العالية، ولا "
                "يناسب الفئوي مباشرة.",
                "green",
            ),
            (
                "تجميعات متوازنة",
                "`BalancedRandomForestClassifier`، `EasyEnsemble`، "
                "`RUSBoost` (من `imbalanced-learn`): كل مقدّر يُدرَّب على عيّنة "
                "متوازنة مختلفة.\n\n"
                ":green[غالبًا الأقوى] على الجداول الحقيقية.",
                "red",
            ),
            (
                "إعادة صياغة المشكلة",
                "إن كانت النسبة أقل من 0.1٪، فقد لا تكون مسألة تصنيف أصلًا "
                "بل **كشف شذوذ** (`IsolationForest`, `OneClassSVM`) — "
                "تُدرَّب على الفئة الطبيعية وحدها.",
                "blue",
            ),
        ],
        n_cols=3,
    )

    ui.danger(
        "**إعادة المعاينة تُطبَّق على بيانات التدريب فقط، وداخل كل طيّة من "
        "التحقّق المتقاطع.** تطبيق SMOTE قبل التقسيم يضع نقاطًا اصطناعية "
        "مولّدة من صفوف التدريب داخل مجموعة الاختبار ⇒ نتائج مبهرة وزائفة. "
        "ولهذا توجد `imblearn.pipeline.Pipeline` المنفصلة: فهي الوحيدة التي "
        "تطبّق إعادة المعاينة في `fit` دون `transform`."
    )

    st.subheader("٤. الأثر الجانبي الذي يُنسى: المعايرة", divider="orange")
    st.markdown(
        """
إعادة المعاينة تغيّر التوزيع المسبق للفئات ⇒ **الاحتمالات الناتجة لم تعد
معايرة**: النموذج المُدرَّب على بيانات متوازنة اصطناعيًا سيقول «احتمال
الاحتيال 40٪» بينما الواقع 2٪.

إن كنت تحتاج احتمالات حقيقية (لحساب تكلفة متوقّعة، أو لترتيب حالات على
محلّل بشري) فإمّا:
- استعمل `class_weight` بدل إعادة المعاينة، أو
- أعد المعايرة: `CalibratedClassifierCV(method="isotonic" | "sigmoid")`.
"""
    )

# ============================================================== الكود
with t_code:
    st.subheader("١. ابدأ من هنا دائمًا: الأوزان", divider="blue")
    ui.code(
        """
LogisticRegression(class_weight="balanced", max_iter=2000)
RandomForestClassifier(class_weight="balanced_subsample")
SVC(class_weight="balanced")

# أو أوزان مخصّصة حسب التكلفة الحقيقية
LogisticRegression(class_weight={0: 1, 1: 25})

# في XGBoost / LightGBM
XGBClassifier(scale_pos_weight=(y == 0).sum() / (y == 1).sum())
"""
    )

    st.subheader("٢. ضبط العتبة بشكل سليم", divider="green")
    ui.code(
        """
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict

# احتمالات خارج الطيّة (out-of-fold) على التدريب — بلا تسريب
proba_oof = cross_val_predict(model, X_train, y_train, cv=5, method="predict_proba")[:, 1]

prec, rec, thr = precision_recall_curve(y_train, proba_oof)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
best_thr = thr[f1[:-1].argmax()]

# أو اختر العتبة التي تحقّق حساسية مطلوبة
target_recall = 0.80
best_thr = thr[(rec[:-1] >= target_recall).nonzero()[0][-1]]

y_pred = (model.predict_proba(X_test)[:, 1] >= best_thr).astype(int)
""",
        caption="scikit-learn يوفّر أيضًا `TunedThresholdClassifierCV` "
        "لضبط العتبة آليًا داخل التحقّق المتقاطع.",
    )

    st.subheader("٣. إعادة المعاينة بشكل سليم (imbalanced-learn)", divider="orange")
    ui.code(
        """
# pip install imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline      # ليست pipeline العادية!
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier

pipe = ImbPipeline([
    ("scale", StandardScaler()),
    ("smote", SMOTE(k_neighbors=5, random_state=0)),   # يُطبَّق في fit فقط
    ("model", LogisticRegression(max_iter=2000)),
])

scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="average_precision")

# متغيّرات فئوية؟ SMOTE العادي لا يصلح
SMOTENC(categorical_features=[3, 7], random_state=0)

# تجميعة متوازنة جاهزة
BalancedRandomForestClassifier(n_estimators=300, random_state=0)
""",
        caption=":red-badge[حرج] استعمال `sklearn.pipeline.Pipeline` مع SMOTE "
        "يطبّق التوليد على بيانات التحقّق أيضًا. `imblearn.pipeline.Pipeline` "
        "هي الوحيدة التي تعرف أن إعادة المعاينة تخصّ `fit` وحده.",
    )

    st.subheader("٤. التقييم والمعايرة", divider="violet")
    ui.code(
        """
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             classification_report, confusion_matrix, matthews_corrcoef)

print("PR-AUC :", average_precision_score(y_test, proba))     # الأهم هنا
print("ROC-AUC:", roc_auc_score(y_test, proba))
print("MCC    :", matthews_corrcoef(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))
print(confusion_matrix(y_test, y_pred))

# أعد المعايرة إن احتجت احتمالات حقيقية
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
cal = CalibratedClassifierCV(model, method="isotonic", cv=5).fit(X_train, y_train)
"""
    )

# ============================================================== الأنيميشن
with t_anim:
    st.subheader("أ. سبع استراتيجيات على نفس البيانات", divider="violet")

    pos_rate = st.select_slider(
        "نسبة الفئة النادرة",
        options=[0.01, 0.02, 0.05, 0.1],
        value=0.02,
        format_func=lambda v: f"{v:.0%}",
        key="imb_rate",
    )
    res, pr_curve = compare_strategies(pos_rate)

    def _render_strategy(frame: int) -> None:
        row = res.iloc[frame]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("الدقّة (accuracy)", f"{row['accuracy']:.3f}", help="لاحظ كم هي مرتفعة دائمًا — ولهذا لا تصلح")
        k2.metric("PR-AUC", f"{row['PR-AUC']:.3f}")
        k3.metric("الحساسية (recall)", f"{row['recall']:.3f}")
        k4.metric("الدقّة الموجبة (precision)", f"{row['precision']:.3f}")

        sofar = res.iloc[: frame + 1][["strategy", "PR-AUC", "recall", "accuracy"]].melt(
            id_vars="strategy", var_name="metric", value_name="value"
        )
        ch = (
            alt.Chart(sofar)
            .mark_bar()
            .encode(
                y=alt.Y("strategy:N", title=None, sort=None),
                x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                color=alt.Color("metric:N", legend=alt.Legend(orient="top", title=None)),
                yOffset="metric:N",
                tooltip=["strategy", "metric", "value"],
            )
            .properties(height=max(200, 48 * (frame + 1)))
        )
        st.altair_chart(ch, width="stretch")
        st.markdown(f"**:violet-badge[{row['strategy']}]**")

    play_animation(
        f"imb_strategies_{pos_rate}",
        len(res),
        _render_strategy,
        fps=0.55,
        label="الاستراتيجية",
        frame_labels=res["strategy"].tolist(),
    )

    st.dataframe(
        res[["strategy", "accuracy", "ROC-AUC", "PR-AUC", "F1", "balanced_acc", "MCC", "recall", "precision"]].round(3),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "ركّز على الصف الأول: النموذج الكسول يحقّق أعلى **دقّة** وأسوأ كل "
        "شيء آخر. هذه مفارقة الدقّة في رقم واحد."
    )

    st.subheader("ب. منحنى الدقّة-الحساسية وأثر العتبة", divider="orange")
    st.markdown(
        "كل نقطة على المنحنى = عتبة مختلفة لنفس النموذج. "
        "اختيار العتبة **قرار عملي** يعتمد على تكلفة الخطأ، لا على رقم سحري."
    )
    thr_pick = st.slider("العتبة", 0.01, 0.95, 0.5, 0.01, key="imb_thr")
    near = pr_curve.iloc[(pr_curve["threshold"] - thr_pick).abs().argsort()[:1]]
    line = (
        alt.Chart(pr_curve)
        .mark_line(strokeWidth=3, color="#4C6EF5")
        .encode(
            x=alt.X("recall:Q", title="recall", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("precision:Q", title="precision", scale=alt.Scale(domain=[0, 1])),
            tooltip=["threshold", "precision", "recall"],
        )
    )
    point = alt.Chart(near).mark_point(size=250, color="#E8590C", filled=True).encode(x="recall:Q", y="precision:Q")
    st.altair_chart((line + point).properties(height=320), width="stretch")
    if len(near):
        n0 = near.iloc[0]
        c1, c2 = st.columns(2)
        c1.metric("الحساسية عند هذه العتبة", f"{n0['recall']:.3f}")
        c2.metric("الدقّة الموجبة عند هذه العتبة", f"{n0['precision']:.3f}")
    st.caption(
        "في كشف الاحتيال أو تشخيص مرض خطير: ارفع الحساسية حتى لو انخفضت "
        "الدقّة الموجبة (الفوت أغلى من الإنذار الكاذب). في تصفية البريد "
        "المزعج: العكس تمامًا."
    )

    st.subheader("ج. ماذا يفعل SMOTE بالبيانات؟", divider="green")
    df_v = imbalanced_fraud(1200, pos_rate)
    Xv, yv = df_v[["amount", "night_ratio"]].to_numpy(float), df_v["fraud"].to_numpy(int)
    Xa, ya = simple_smote(Xv, yv)
    orig = pd.DataFrame(Xv, columns=["amount", "night_ratio"]).assign(
        kind=np.where(yv == 1, "fraud (real)", "normal")
    )
    synth = pd.DataFrame(Xa[len(Xv):], columns=["amount", "night_ratio"]).assign(kind="fraud (synthetic)")
    both = pd.concat([orig, synth], ignore_index=True)
    st.altair_chart(
        charts.scatter(both.sample(min(len(both), 1500), random_state=0), "amount", "night_ratio", color="kind", height=340),
        width="stretch",
    )
    st.caption(
        "النقاط الاصطناعية تقع **على القطع الواصلة** بين نقاط الاحتيال "
        "الحقيقية. وهنا يظهر القيد: إن كانت الفئة النادرة متفرّقة أو "
        "متداخلة مع الأغلبية، فإن SMOTE يولّد نقاطًا داخل منطقة الأغلبية "
        "ويزيد الالتباس بدل أن يقلّله."
    )

# ============================================================== القرار
with t_decide:
    st.subheader("ما الذي أجرّبه، وبأي ترتيب؟", divider="orange")
    ui.decision_table(
        [
            ("دائمًا، الخطوة ١", "غيّر المقياس إلى PR-AUC / MCC / تكلفة", "الدقّة تخفي الفشل تمامًا"),
            ("دائمًا، الخطوة ٢", "`class_weight='balanced'`", "مجاني، بلا تشويه للبيانات"),
            ("دائمًا، الخطوة ٣", "اضبط العتبة على تحقّق", "أكبر مكسب بأقل تكلفة"),
            ("اختلال معتدل (10–30٪)", "غالبًا لا تحتاج أكثر من ذلك", "النماذج تتعامل معه جيدًا"),
            ("اختلال شديد (1–5٪)", "جرّب SMOTE/undersampling + تجميعات متوازنة", "وقارن بـ PR-AUC داخل CV"),
            ("اختلال متطرّف (< 0.1٪)", "أعد الصياغة ككشف شذوذ", "لا توجد أمثلة كافية للتعلّم المُوجَّه"),
            ("بيانات ضخمة والفئة الأغلب هائلة", "Undersampling", "يسرّع التدريب بلا خسارة تُذكر"),
            ("متغيّرات فئوية كثيرة", "`SMOTENC` أو أوزان", "SMOTE العادي يولّد فئات غير موجودة"),
            ("تحتاج احتمالات حقيقية", "أوزان + `CalibratedClassifierCV`", "إعادة المعاينة تكسر المعايرة"),
            ("تعرف تكلفة الخطأين", "تعلّم حسّاس للتكلفة + عتبة مثلى", "الأصدق تجاريًا من أي مقياس عام"),
        ],
        headers=("الحالة", "الإجراء", "السبب"),
    )

    ui.tip(
        "قبل أي معالجة، اسأل: **هل المشكلة في البيانات أم في المقياس؟** "
        "كثير من «مشاكل عدم التوازن» تختفي بمجرّد استبدال `accuracy` بـ "
        "`average_precision` وضبط العتبة."
    )

# ============================================================== المصادر
with t_ref:
    ui.refs(
        [
            ("imbalanced-learn — التوثيق الرسمي", "https://imbalanced-learn.org/stable/"),
            ("imbalanced-learn — Common pitfalls and recommended practices", "https://imbalanced-learn.org/stable/common_pitfalls.html"),
            ("scikit-learn — Metrics for imbalanced classification", "https://scikit-learn.org/stable/modules/model_evaluation.html"),
            ("scikit-learn — Precision-Recall", "https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html"),
            ("scikit-learn — Probability calibration", "https://scikit-learn.org/stable/modules/calibration.html"),
            ("scikit-learn — TunedThresholdClassifierCV", "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html"),
            ("Chawla et al. (2002) — SMOTE", "https://doi.org/10.1613/jair.953"),
            ("Saito & Rehmsmeier (2015) — The PR plot is more informative than ROC", "https://doi.org/10.1371/journal.pone.0118432"),
        ]
    )
    ui.terms(
        [
            ("Class imbalance", "عدم توازن الفئات"),
            ("Accuracy paradox", "مفارقة الدقّة"),
            ("PR-AUC / Average precision", "المساحة تحت منحنى الدقّة-الحساسية"),
            ("MCC", "معامل ارتباط ماثيوز"),
            ("Oversampling / Undersampling", "زيادة/تقليل المعاينة"),
            ("SMOTE", "توليد عيّنات اصطناعية للفئة النادرة"),
            ("Decision threshold", "عتبة القرار"),
            ("Calibration", "معايرة الاحتمالات"),
        ]
    )
