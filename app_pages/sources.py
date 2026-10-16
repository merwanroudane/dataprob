"""المصادر الرسمية وقائمة المصطلحات."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core import ui

ui.page_header(
    "المصادر والمصطلحات",
    "كل ما في هذه المنصّة مبني على التوثيق الرسمي للمكتبات والمراجع "
    "الأكاديمية المذكورة أدناه. اجعل التوثيق الرسمي مرجعك الأول دائمًا: "
    "فهو الوحيد الذي يتحدّث عن إصدارك أنت.",
    icon=":material/library_books:",
    badges=[":blue-badge[scikit-learn]", ":green-badge[pandas]", ":orange-badge[مراجع أكاديمية]"],
)

t_docs, t_books, t_glossary, t_env = st.tabs(
    [
        ":material/link: التوثيق الرسمي",
        ":material/menu_book: كتب وأوراق",
        ":material/translate: قائمة المصطلحات",
        ":material/terminal: البيئة والتشغيل",
    ]
)

DOCS: dict[str, list[tuple[str, str]]] = {
    "scikit-learn — الأساس": [
        ("User Guide — الصفحة الرئيسية", "https://scikit-learn.org/stable/user_guide.html"),
        ("Common pitfalls and recommended practices", "https://scikit-learn.org/stable/common_pitfalls.html"),
        ("Dataset transformations", "https://scikit-learn.org/stable/data_transforms.html"),
        ("Pipelines and composite estimators", "https://scikit-learn.org/stable/modules/compose.html"),
        ("Cross-validation", "https://scikit-learn.org/stable/modules/cross_validation.html"),
        ("Metrics and scoring", "https://scikit-learn.org/stable/modules/model_evaluation.html"),
        ("API Reference", "https://scikit-learn.org/stable/api/index.html"),
    ],
    "scikit-learn — التحضير": [
        ("Preprocessing data (التقييس والترميز والتحويل)", "https://scikit-learn.org/stable/modules/preprocessing.html"),
        ("Imputation of missing values", "https://scikit-learn.org/stable/modules/impute.html"),
        ("Feature selection", "https://scikit-learn.org/stable/modules/feature_selection.html"),
        ("Permutation feature importance", "https://scikit-learn.org/stable/modules/permutation_importance.html"),
        ("Novelty and outlier detection", "https://scikit-learn.org/stable/modules/outlier_detection.html"),
        ("Decomposing signals in components (PCA)", "https://scikit-learn.org/stable/modules/decomposition.html"),
        ("Probability calibration", "https://scikit-learn.org/stable/modules/calibration.html"),
        ("Model persistence", "https://scikit-learn.org/stable/model_persistence.html"),
    ],
    "scikit-learn — أمثلة مفتاحية": [
        ("Compare the effect of different scalers on data with outliers", "https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html"),
        ("Imputing missing values before building an estimator", "https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html"),
        ("Comparing anomaly detection algorithms", "https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html"),
        ("Column Transformer with Mixed Types", "https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html"),
        ("Target Encoder's Internal Cross fitting", "https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html"),
        ("Visualizing cross-validation behavior", "https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html"),
        ("Nested versus non-nested cross-validation", "https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html"),
        ("Time-related feature engineering", "https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html"),
    ],
    "pandas": [
        ("User Guide", "https://pandas.pydata.org/docs/user_guide/index.html"),
        ("Working with missing data", "https://pandas.pydata.org/docs/user_guide/missing_data.html"),
        ("Categorical data", "https://pandas.pydata.org/docs/user_guide/categorical.html"),
        ("Working with text data", "https://pandas.pydata.org/docs/user_guide/text.html"),
        ("Time series / date functionality", "https://pandas.pydata.org/docs/user_guide/timeseries.html"),
        ("Group by: split-apply-combine", "https://pandas.pydata.org/docs/user_guide/groupby.html"),
        ("Windowing operations", "https://pandas.pydata.org/docs/user_guide/window.html"),
        ("Nullable integer data type", "https://pandas.pydata.org/docs/user_guide/integer_na.html"),
    ],
    "مكتبات متخصّصة": [
        ("imbalanced-learn — عدم توازن الفئات", "https://imbalanced-learn.org/stable/"),
        ("imbalanced-learn — Common pitfalls", "https://imbalanced-learn.org/stable/common_pitfalls.html"),
        ("feature-engine — هندسة المتغيّرات", "https://feature-engine.trainindata.com/"),
        ("category_encoders — ترميز فئوي شامل", "https://contrib.scikit-learn.org/category_encoders/"),
        ("missingno — تصوير أنماط الفقد", "https://github.com/ResidentMario/missingno"),
        ("ydata-profiling — تقرير استكشافي آلي", "https://docs.profiling.ydata.ai/"),
        ("pandera — تحقّق من صحّة الجداول", "https://pandera.readthedocs.io/"),
        ("Great Expectations — اختبارات جودة البيانات", "https://docs.greatexpectations.io/"),
        ("statsmodels — MICE والاستدلال", "https://www.statsmodels.org/stable/imputation.html"),
        ("SciPy — الاختبارات الإحصائية", "https://docs.scipy.org/doc/scipy/reference/stats.html"),
        ("RapidFuzz — مطابقة نصّية تقريبية", "https://rapidfuzz.github.io/RapidFuzz/"),
        ("UMAP — تقليل أبعاد غير خطّي", "https://umap-learn.readthedocs.io/"),
    ],
}

BOOKS: list[tuple[str, str, str]] = [
    ("Little & Rubin — Statistical Analysis with Missing Data (3rd ed., 2019)", "المرجع الأمّ للقيم المفقودة", "https://doi.org/10.1002/9781119482260"),
    ("van Buuren — Flexible Imputation of Missing Data (2nd ed.)", "كتاب مجاني بالكامل على الإنترنت، عملي وممتاز", "https://stefvanbuuren.name/fimd/"),
    ("Rubin (1976) — Inference and Missing Data", "الورقة التي أسّست تصنيف MCAR/MAR/MNAR", "https://doi.org/10.1093/biomet/63.3.581"),
    ("Hawkins (1980) — Identification of Outliers", "التعريف الكلاسيكي للقيمة الشاذة", "https://doi.org/10.1007/978-94-015-3994-4"),
    ("Aggarwal — Outlier Analysis (2nd ed., 2017)", "مرجع شامل في كشف الشذوذ", "https://doi.org/10.1007/978-3-319-47578-3"),
    ("Rousseeuw & Leroy — Robust Regression and Outlier Detection", "الأساس النظري للطرق الأمتن", "https://doi.org/10.1002/0471725382"),
    ("Hastie, Tibshirani & Friedman — The Elements of Statistical Learning", "متاح مجانًا؛ §7.10.2 عن تحيّز الاختيار", "https://hastie.su.domains/ElemStatLearn/"),
    ("Kuhn & Johnson — Feature Engineering and Selection", "كتاب مجاني مخصّص لهندسة واختيار المتغيّرات", "http://www.feat.engineering/"),
    ("Kaufman, Rosset & Perlich (2012) — Leakage in Data Mining", "التصنيف المرجعي لأنواع التسريب", "https://dl.acm.org/doi/10.1145/2020408.2020496"),
    ("Kapoor & Narayanan (2023) — Leakage and the Reproducibility Crisis", "مسح لمئات الأوراق المتأثّرة بالتسريب", "https://doi.org/10.1016/j.patter.2023.100804"),
    ("Chawla et al. (2002) — SMOTE", "الورقة الأصلية", "https://doi.org/10.1613/jair.953"),
    ("Guyon & Elisseeff (2003) — Variable and Feature Selection", "مقدّمة مرجعية", "https://www.jmlr.org/papers/v3/guyon03a.html"),
    ("Saito & Rehmsmeier (2015) — PR plot vs ROC", "لماذا PR-AUC مع البيانات غير المتوازنة", "https://doi.org/10.1371/journal.pone.0118432"),
    ("Jolliffe & Cadima (2016) — PCA: a review", "مراجعة حديثة لتحليل المكوّنات الرئيسية", "https://doi.org/10.1098/rsta.2015.0202"),
    ("Leys et al. (2013) — Do not use SD around the mean", "لماذا MAD أفضل من z-score", "https://doi.org/10.1016/j.jesp.2013.03.013"),
]

GLOSSARY: list[tuple[str, str, str]] = [
    ("Accuracy paradox", "مفارقة الدقّة", "عدم التوازن"),
    ("ADASYN", "توليد تكيّفي للعيّنات الاصطناعية", "عدم التوازن"),
    ("Attenuation bias", "تخفيف المعامل نحو الصفر", "الفقد"),
    ("Binning / Discretization", "التجزيء إلى شرائح", "التحويلات"),
    ("Box-Cox", "تحويل بوكس-كوكس (قيم موجبة)", "التحويلات"),
    ("Breakdown point", "نقطة الانهيار (نسبة التلوّث المحتملة)", "الشواذ"),
    ("Calibration", "معايرة الاحتمالات", "عدم التوازن"),
    ("Cardinality", "عدد الفئات المختلفة", "البيانات الفئوية"),
    ("Class imbalance", "عدم توازن الفئات", "عدم التوازن"),
    ("ColumnTransformer", "معالجات مختلفة لمجموعات أعمدة مختلفة", "Pipelines"),
    ("Complete-case analysis", "التحليل على الصفوف المكتملة فقط", "الفقد"),
    ("Contamination", "نسبة التلوّث المفترضة", "الشواذ"),
    ("Contextual outlier", "شاذ سياقي", "الشواذ"),
    ("Cook's distance", "مسافة كوك لقياس التأثير", "الشواذ"),
    ("Curse of dimensionality", "لعنة الأبعاد", "اختيار المتغيّرات"),
    ("Data drift", "انزياح توزيع البيانات مع الزمن", "التشخيص"),
    ("Data leakage", "تسريب البيانات", "التسريب"),
    ("Deduplication", "إزالة التكرار", "التكرار"),
    ("Dummy variable trap", "فخّ المتغيّرات الصورية", "الترميز"),
    ("Embedded method", "طريقة اختيار مدمجة في التدريب", "اختيار المتغيّرات"),
    ("Explained variance ratio", "نسبة التباين المفسَّر", "PCA"),
    ("Feature engineering", "هندسة المتغيّرات", "المتغيّرات"),
    ("Filter method", "طريقة ترشيح مستقلّة عن النموذج", "اختيار المتغيّرات"),
    ("fit / transform", "تعلّم المعاملات / تطبيقها", "Pipelines"),
    ("GroupKFold", "تقسيم يُبقي المجموعة في جهة واحدة", "التقسيم"),
    ("Heteroscedasticity", "عدم تجانس التباين", "التحويلات"),
    ("High cardinality", "كاردينالية عالية", "الترميز"),
    ("Huber loss", "خسارة هوبر (تربيعية ثم خطّية)", "الشواذ"),
    ("Imputation", "التعويض عن القيم المفقودة", "الفقد"),
    ("Influential point", "نقطة مؤثّرة على نتائج النموذج", "الشواذ"),
    ("Isolation Forest", "غابة العزل لكشف الشذوذ", "الشواذ"),
    ("Kurtosis", "التفلطح (ثقل الذيول)", "التحويلات"),
    ("Leverage", "رافعة المشاهدة في الانحدار", "الشواذ"),
    ("Listwise deletion", "حذف الصف كاملًا عند أي فقد", "الفقد"),
    ("LOF", "عامل الشذوذ المحلي", "الشواذ"),
    ("MAD", "متوسط الانحراف المطلق عن الوسيط", "الشواذ"),
    ("Mahalanobis distance", "مسافة ماهالانوبيس", "الشواذ"),
    ("MAR", "مفقود عشوائيًا مشروطًا بمتغيّرات مرصودة", "الفقد"),
    ("MCAR", "مفقود عشوائيًا تمامًا", "الفقد"),
    ("MCD", "محدّد التغاير الأدنى (تقدير أمتن)", "الشواذ"),
    ("MCC", "معامل ارتباط ماثيوز", "التقييم"),
    ("MICE", "تعويض متعدّد بالمعادلات المتسلسلة", "الفقد"),
    ("Missing indicator", "عمود ثنائي يسجّل واقعة الفقد", "الفقد"),
    ("MNAR", "مفقود غير عشوائي (يعتمد على القيمة نفسها)", "الفقد"),
    ("Monotonic transform", "تحويل رتيب يحفظ الترتيب", "التحويلات"),
    ("Multicollinearity", "التعدّد الخطّي بين المتغيّرات", "اختيار المتغيّرات"),
    ("Multiple Imputation", "التعويض المتعدّد", "الفقد"),
    ("Nested cross-validation", "تحقّق متقاطع متداخل", "التقسيم"),
    ("Normalization", "التطبيع (MinMax أو توحيد طول الصف)", "التقييس"),
    ("Novelty detection", "كشف الأنماط الجديدة", "الشواذ"),
    ("Nullity correlation", "ارتباط بين مؤشّرات الفقد", "الفقد"),
    ("One-Hot encoding", "ترميز ثنائي لكل فئة", "الترميز"),
    ("Ordinal encoding", "ترميز ترتيبي", "الترميز"),
    ("Outlier", "قيمة شاذة", "الشواذ"),
    ("Oversampling", "زيادة معاينة الفئة النادرة", "عدم التوازن"),
    ("PCA", "تحليل المكوّنات الرئيسية", "PCA"),
    ("Permutation importance", "أهمية التبديل", "اختيار المتغيّرات"),
    ("Pipeline", "سلسلة خطوات تُعامَل كنموذج واحد", "Pipelines"),
    ("PR-AUC", "المساحة تحت منحنى الدقّة-الحساسية", "التقييم"),
    ("Q-Q plot", "رسم المئينات المقارن", "التحويلات"),
    ("Record linkage", "ربط السجلات المتطابقة", "التكرار"),
    ("RFE", "الحذف التكراري للمتغيّرات", "اختيار المتغيّرات"),
    ("Robust statistic", "إحصاءة أمتن لا تتأثّر بالشواذ", "الشواذ"),
    ("ROC-AUC", "المساحة تحت منحنى ROC", "التقييم"),
    ("Rubin's rules", "قواعد دمج نتائج التعويض المتعدّد", "الفقد"),
    ("Scaling", "التقييس", "التقييس"),
    ("Selection bias", "تحيّز الاختيار", "التسريب"),
    ("Sensitivity analysis", "تحليل الحساسية", "المنهجية"),
    ("Sentinel value", "قيمة تُستعمل كرمز للفقد مثل -99", "التشخيص"),
    ("Skewness", "الالتواء", "التحويلات"),
    ("SMOTE", "توليد عيّنات اصطناعية للفئة النادرة", "عدم التوازن"),
    ("Sparsity", "التناثر (كثرة الأصفار)", "التقييس"),
    ("Standardization", "التوحيد المعياري (z-score)", "التقييس"),
    ("Stratified split", "تقسيم يحافظ على نسب الفئات", "التقسيم"),
    ("Target encoding", "ترميز بمتوسط الهدف", "الترميز"),
    ("Target leakage", "تسريب الهدف", "التسريب"),
    ("Temporal leakage", "تسريب زمني من المستقبل", "التسريب"),
    ("Threshold tuning", "ضبط عتبة القرار", "عدم التوازن"),
    ("TimeSeriesSplit", "تقسيم زمني متتابع", "التقسيم"),
    ("Undersampling", "تقليل معاينة الفئة الأغلب", "عدم التوازن"),
    ("Unseen category", "فئة لم تظهر في التدريب", "الترميز"),
    ("VIF", "معامل تضخّم التباين", "اختيار المتغيّرات"),
    ("Winsorizing", "قصّ القيم عند مئينات محدّدة", "الشواذ"),
    ("Wrapper method", "طريقة اختيار بالتغليف", "اختيار المتغيّرات"),
    ("Yeo-Johnson", "تحويل يقبل القيم السالبة والصفر", "التحويلات"),
    ("z-score", "الدرجة المعيارية", "الشواذ"),
]

with t_docs:
    st.caption(
        "كل الروابط إلى التوثيق الرسمي المستقرّ (stable). تحقّق دائمًا من "
        "أن صفحة التوثيق تطابق إصدار المكتبة لديك."
    )
    for section, items in DOCS.items():
        with st.container(border=True):
            st.markdown(f"##### :blue-badge[{section}]")
            st.markdown("\n".join(f"- [{t}]({u})" for t, u in items))

with t_books:
    st.markdown("##### :material/school: مراجع أكاديمية وكتب")
    for title, note, url in BOOKS:
        with st.container(border=True):
            st.markdown(f"**[{title}]({url})**")
            st.caption(note)

with t_glossary:
    st.markdown(
        "**قائمة المصطلحات (Glossary)** — ابحث بالإنجليزية أو بالعربية. "
        f"العدد الكلّي: {len(GLOSSARY)} مصطلحًا."
    )
    q = st.text_input("بحث", placeholder="مثال: imputation أو تعويض", key="gloss_q")
    g = pd.DataFrame(GLOSSARY, columns=["Term (English)", "المعنى بالعربية", "الموضوع"])
    topics = sorted(g["الموضوع"].unique())
    picked = st.multiselect("تصفية حسب الموضوع", topics, key="gloss_topic")

    view = g
    if q:
        mask = view.apply(lambda r: q.strip().lower() in " ".join(r.astype(str)).lower(), axis=1)
        view = view[mask]
    if picked:
        view = view[view["الموضوع"].isin(picked)]

    st.dataframe(view, width="stretch", hide_index=True, height=520)
    st.download_button(
        "نزّل قائمة المصطلحات (CSV)",
        data=g.to_csv(index=False).encode("utf-8-sig"),
        file_name="ml_data_prep_glossary.csv",
        mime="text/csv",
        icon=":material/download:",
    )

with t_env:
    st.markdown("##### :material/package_2: تشغيل هذه المنصّة محليًا")
    ui.code(
        """
pip install streamlit pandas numpy scikit-learn scipy altair statsmodels
streamlit run streamlit_app.py
""",
        lang="bash",
    )
    st.markdown("##### :material/extension: حزم اختيارية تُذكر في الدروس")
    ui.code(
        """
pip install imbalanced-learn   # SMOTE والتجميعات المتوازنة
pip install missingno          # تصوير أنماط الفقد
pip install category_encoders  # ترميز فئوي إضافي
pip install feature-engine     # هندسة متغيّرات جاهزة
pip install rapidfuzz          # مطابقة نصّية تقريبية
pip install ydata-profiling    # تقرير استكشافي آلي
""",
        lang="bash",
    )

    st.markdown("##### :material/info: ملاحظة منهجية")
    st.markdown(
        """
كل الأمثلة في هذه المنصّة تستعمل **بيانات مُصنّعة** عمدًا: فقط حين نعرف
«الحقيقة» سلفًا نستطيع قياس خطأ كل معالجة بدقّة. على بياناتك الحقيقية لن
تملك هذه الرفاهية — ولهذا تحديدًا تصبح **المنهجية** (التقسيم المبكر،
التحقّق المتقاطع، تحليل الحساسية، التوثيق) بديلك الوحيد عن معرفة الحقيقة.
"""
    )
