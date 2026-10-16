"""عناصر واجهة مشتركة بين جميع صفحات المنصة.

الهدف: شكل موحّد لكل درس — عنوان، مقدّمة، بطاقات مفاهيم، جداول قرار
(«متى نحذف / متى نعوّض؟»)، مصطلحات، ومراجع رسمية.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import streamlit as st

# ترتيب موحّد لتبويبات كل درس
TAB_THEORY = "النظرية"
TAB_CODE = "التطبيق بالبايثون"
TAB_ANIM = "أنيميشن تفاعلي"
TAB_DECIDE = "متى؟ جدول القرار"
TAB_REF = "المصادر"


def page_header(title: str, lead: str, *, icon: str = ":material/school:", badges: Sequence[str] = ()) -> None:
    """ترويسة الدرس: عنوان + جملة تأطير + شارات للمفاتيح الأساسية."""
    st.header(title, icon=icon)
    st.markdown(lead)
    if badges:
        st.markdown(" ".join(badges))


def lesson_tabs(*, anim: bool = True, decide: bool = True, refs: bool = True):
    """تبويبات الدرس بالترتيب المعتاد."""
    labels = [f":material/menu_book: {TAB_THEORY}", f":material/code: {TAB_CODE}"]
    if anim:
        labels.append(f":material/animation: {TAB_ANIM}")
    if decide:
        labels.append(f":material/rule: {TAB_DECIDE}")
    if refs:
        labels.append(f":material/link: {TAB_REF}")
    return st.tabs(labels)


def concept(title: str, body: str, *, color: str = "blue", icon: str | None = None) -> None:
    """بطاقة مفهوم: عنوان ملوّن + شرح."""
    with st.container(border=True):
        head = f":{color}-badge[{title}]"
        if icon:
            head = f"{icon} {head}"
        st.markdown(f"##### {head}")
        st.markdown(body)


def concept_grid(items: Sequence[tuple[str, str, str]], *, n_cols: int = 2) -> None:
    """شبكة بطاقات: (العنوان، الشرح، اللون)."""
    cols = st.columns(min(n_cols, max(len(items), 1)))
    for i, (title, body, color) in enumerate(items):
        with cols[i % len(cols)]:
            concept(title, body, color=color)


def decision_table(rows: Sequence[tuple[str, str, str]], *, headers: tuple[str, str, str] = ("الحالة", "القرار", "السبب / الخطر")) -> None:
    """جدول قرار من ثلاثة أعمدة بصيغة Markdown."""
    md = [f"| {headers[0]} | {headers[1]} | {headers[2]} |", "|---|---|---|"]
    md += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    st.markdown("\n".join(md))


def terms(pairs: Sequence[tuple[str, str]]) -> None:
    """جدول مصطلحات: المصطلح بالإنجليزية ↔ المعنى بالعربية."""
    md = ["| Term (English) | المعنى بالعربية |", "|---|---|"]
    md += [f"| `{en}` | {ar} |" for en, ar in pairs]
    st.markdown("\n".join(md))


def code(snippet: str, *, caption: str | None = None, lang: str = "python") -> None:
    """مقطع كود + تعليق سفلي اختياري."""
    st.code(snippet.strip("\n"), language=lang)
    if caption:
        st.caption(caption)


def refs(items: Iterable[tuple[str, str]]) -> None:
    """قائمة مراجع رسمية: (العنوان، الرابط)."""
    st.markdown("##### :material/verified: مصادر رسمية ومراجع")
    st.markdown("\n".join(f"- [{t}]({u})" for t, u in items))


def takeaways(points: Sequence[str]) -> None:
    """خلاصة الدرس في نقاط."""
    with st.container(border=True):
        st.markdown("##### :material/lightbulb: :green-badge[خلاصة الدرس]")
        st.markdown("\n".join(f"- {p}" for p in points))


def danger(text: str) -> None:
    st.warning(text, icon=":material/report:")


def tip(text: str) -> None:
    st.info(text, icon=":material/tips_and_updates:")


def formula(latex: str, explain: str = "") -> None:
    st.latex(latex)
    if explain:
        st.caption(explain)


def beginner_note(text: str) -> None:
    """صندوق «للمبتدئ» — تبسيط إضافي بلغة يومية."""
    with st.container(border=True):
        st.markdown(f"##### :material/child_care: :violet-badge[بلغة مبسّطة]")
        st.markdown(text)
