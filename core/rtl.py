"""دعم الكتابة من اليمين إلى اليسار (RTL) ووضع الشريط الجانبي على اليمين.

Streamlit لا يوفّر خيارًا أصليًا في `config.toml` لقلب اتجاه الواجهة، لذلك
هذا هو الموضع الوحيد في المنصة الذي نستخدم فيه CSS مخصّصًا — وبطلب صريح من
المستخدم (الشريط الجانبي من جهة اليمين). كل ما تبقّى من الألوان والخطوط
والحدود يُضبط عبر `.streamlit/config.toml`.
"""

from __future__ import annotations

import streamlit as st

_RTL_CSS = """
<style>
/* ===== 1) اتجاه التطبيق ككل: من اليمين إلى اليسار ===== */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="stBottomBlockContainer"] {
    direction: rtl;
}

/* اتجاه rtl على حاوية الـ flex يكفي لنقل الشريط الجانبي إلى جهة اليمين */

/* ===== 2) محاذاة النصوص ===== */
[data-testid="stMarkdownContainer"],
[data-testid="stHeadingWithActionElements"],
[data-testid="stCaptionContainer"],
[data-testid="stText"],
[data-testid="stMetricLabel"],
[data-testid="stWidgetLabel"],
[data-testid="stExpander"] summary,
[data-testid="stAlertContentInfo"],
[data-testid="stAlertContentSuccess"],
[data-testid="stAlertContentWarning"],
[data-testid="stAlertContentError"] {
    text-align: right;
}

/* القوائم النقطية والمرقّمة */
[data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] ol {
    padding-right: 1.4rem;
    padding-left: 0;
}

/* الاقتباسات: الشريط الملوّن ينتقل إلى اليمين */
[data-testid="stMarkdownContainer"] blockquote {
    border-right: 4px solid var(--border-color, #DCE6FA);
    border-left: none;
    padding-right: 1rem;
    padding-left: 0;
}

/* ===== 3) استثناءات: محتوى يجب أن يبقى LTR ===== */
/* الأكواد، المخرجات، الجداول، الرسوم والمعادلات لغتها الأصلية إنجليزية */
[data-testid="stCode"],
[data-testid="stCodeBlock"],
[data-testid="stCode"] *,
pre, code, kbd, samp,
[data-testid="stJson"],
[data-testid="stDataFrame"],
[data-testid="stDataFrameResizable"],
[data-testid="stTable"],
[data-testid="stVegaLiteChart"],
[data-testid="stPlotlyChart"],
[data-testid="stPyplotChart"],
[data-testid="stMermaidChart"] {
    direction: ltr;
    text-align: left;
}

/* جداول Markdown: نص عربي لكن ترتيب الأعمدة من اليمين */
[data-testid="stMarkdownContainer"] table {
    direction: rtl;
    text-align: right;
}
[data-testid="stMarkdownContainer"] th,
[data-testid="stMarkdownContainer"] td {
    text-align: right;
}

/* معادلات KaTeX تبقى LTR */
.katex, .katex-display {
    direction: ltr;
}

/* ===== 4) لمسات صغيرة على عناصر الإدخال ===== */
[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"] {
    direction: ltr;
}
</style>
"""


def inject_rtl() -> None:
    """يُحقن مرة واحدة في بداية كل صفحة (عبر `streamlit_app.py`)."""
    st.html(_RTL_CSS)
