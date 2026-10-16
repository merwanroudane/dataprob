"""منصّة تعليمية شاملة: تحضير البيانات وتنظيفها للتعلّم الآلي.

نقطة الدخول: تبني التنقّل (شريط جانبي على اليمين) وتشغّل الصفحة المختارة.
"""

from __future__ import annotations

import streamlit as st

from core.rtl import inject_rtl

st.set_page_config(
    page_title="تحضير البيانات للتعلّم الآلي",
    page_icon=":material/mop:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_rtl()

P = "app_pages"

pages = {
    "ابدأ من هنا": [
        st.Page(f"{P}/home.py", title="الصفحة الرئيسية", icon=":material/home:", default=True),
        st.Page(f"{P}/roadmap.py", title="خريطة الطريق الكاملة", icon=":material/map:"),
        st.Page(f"{P}/data_types.py", title="أنواع البيانات ومستويات القياس", icon=":material/category:"),
        st.Page(f"{P}/audit.py", title="فحص جودة البيانات", icon=":material/fact_check:"),
    ],
    "القيم المفقودة Missing data": [
        st.Page(f"{P}/missing_types.py", title="أنواع الفقد: MCAR / MAR / MNAR", icon=":material/help_center:"),
        st.Page(f"{P}/missing_diagnose.py", title="تشخيص الفقد واختباراته", icon=":material/stethoscope:"),
        st.Page(f"{P}/missing_delete.py", title="الحذف: متى ولماذا", icon=":material/delete_sweep:"),
        st.Page(f"{P}/missing_impute.py", title="التعويض Imputation", icon=":material/healing:"),
        st.Page(f"{P}/missing_impact.py", title="أثر الفقد على الخوارزميات", icon=":material/psychology:"),
    ],
    "القيم الشاذة Outliers": [
        st.Page(f"{P}/outliers_types.py", title="أنواع القيم الشاذة", icon=":material/scatter_plot:"),
        st.Page(f"{P}/outliers_uni.py", title="الكشف أحادي المتغيّر", icon=":material/show_chart:"),
        st.Page(f"{P}/outliers_multi.py", title="الكشف متعدّد المتغيّرات", icon=":material/blur_on:"),
        st.Page(f"{P}/outliers_treat.py", title="المعالجة: نحتفظ أم نحذف؟", icon=":material/rule:"),
    ],
    "التكرار والتناسق": [
        st.Page(f"{P}/duplicates.py", title="القيم المكرّرة", icon=":material/content_copy:"),
        st.Page(f"{P}/consistency.py", title="توحيد الصيغ والأنواع", icon=":material/format_align_right:"),
    ],
    "التحويلات Transformations": [
        st.Page(f"{P}/scaling.py", title="التقييس Scaling", icon=":material/straighten:"),
        st.Page(f"{P}/distributions.py", title="تحويل التوزيعات والتجزيء", icon=":material/area_chart:"),
        st.Page(f"{P}/encoding.py", title="ترميز المتغيّرات الفئوية", icon=":material/translate:"),
    ],
    "هندسة المتغيّرات Features": [
        st.Page(f"{P}/feature_engineering.py", title="هندسة المتغيّرات", icon=":material/construction:"),
        st.Page(f"{P}/feature_selection.py", title="اختيار المتغيّرات", icon=":material/filter_alt:"),
        st.Page(f"{P}/dim_reduction.py", title="تقليل الأبعاد PCA", icon=":material/compress:"),
    ],
    "ما قبل النموذج": [
        st.Page(f"{P}/imbalance.py", title="عدم توازن الفئات", icon=":material/balance:"),
        st.Page(f"{P}/leakage.py", title="تسريب البيانات والتقسيم", icon=":material/water_damage:"),
        st.Page(f"{P}/pipelines.py", title="بناء Pipeline كامل", icon=":material/account_tree:"),
    ],
    "مرجع سريع": [
        st.Page(f"{P}/checklist.py", title="قائمة التحقق النهائية", icon=":material/checklist:"),
        st.Page(f"{P}/sources.py", title="المصادر والمصطلحات", icon=":material/library_books:"),
    ],
}

page = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.space("small")
    st.caption(
        "منصّة تعليمية عربية · المصطلحات بالإنجليزية · "
        "الأكواد بـ scikit-learn / pandas"
    )

page.run()
