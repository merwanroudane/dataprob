"""دوال رسم مختصرة مبنية على Altair / Vega-Lite.

ملاحظة عن اللغة: عناوين المحاور تُكتب بالإنجليزية لأن محرّك الرسم
(Vega-Lite) يرسم النص داخل SVG ولا يطبّق تشكيل الحروف العربية بشكل موثوق.
الشرح العربي يوضع دائمًا في النص المحيط بالرسم.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

HEIGHT = 320


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    size: str | None = None,
    tooltip: list[str] | None = None,
    height: int = HEIGHT,
    opacity: float = 0.75,
    domain_x: tuple[float, float] | None = None,
    domain_y: tuple[float, float] | None = None,
) -> alt.Chart:
    enc: dict = {
        "x": alt.X(f"{x}:Q", title=x, scale=alt.Scale(domain=list(domain_x), nice=False) if domain_x else alt.Scale(zero=False)),
        "y": alt.Y(f"{y}:Q", title=y, scale=alt.Scale(domain=list(domain_y), nice=False) if domain_y else alt.Scale(zero=False)),
    }
    if color:
        enc["color"] = alt.Color(f"{color}:N", title=color, legend=alt.Legend(orient="top"))
    if size:
        enc["size"] = alt.Size(f"{size}:Q", title=size)
    enc["tooltip"] = tooltip or [c for c in df.columns[:6]]
    return (
        alt.Chart(df)
        .mark_circle(opacity=opacity, size=70)
        .encode(**enc)
        .properties(height=height)
    )


def hist(
    df: pd.DataFrame,
    col: str,
    *,
    bins: int = 40,
    color: str | None = None,
    height: int = HEIGHT,
    title: str = "",
) -> alt.Chart:
    enc: dict = {
        "x": alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins), title=col),
        "y": alt.Y("count()", title="count"),
    }
    if color:
        enc["color"] = alt.Color(f"{color}:N", title=color, legend=alt.Legend(orient="top"))
    return alt.Chart(df).mark_bar(opacity=0.85).encode(**enc).properties(height=height, title=title)


def bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    height: int = HEIGHT,
    sort_desc: bool = True,
    title: str = "",
    horizontal: bool = False,
) -> alt.Chart:
    sort = "-y" if sort_desc else None
    if horizontal:
        enc = {
            "y": alt.Y(f"{x}:N", title=x, sort="-x" if sort_desc else None),
            "x": alt.X(f"{y}:Q", title=y),
        }
    else:
        enc = {
            "x": alt.X(f"{x}:N", title=x, sort=sort),
            "y": alt.Y(f"{y}:Q", title=y),
        }
    if color:
        enc["color"] = alt.Color(f"{color}:N", title=color, legend=alt.Legend(orient="top"))
    enc["tooltip"] = list(df.columns[:5])
    return alt.Chart(df).mark_bar().encode(**enc).properties(height=height, title=title)


def line(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    height: int = HEIGHT,
    points: bool = True,
    title: str = "",
) -> alt.Chart:
    enc = {
        "x": alt.X(f"{x}:Q", title=x, scale=alt.Scale(zero=False)),
        "y": alt.Y(f"{y}:Q", title=y, scale=alt.Scale(zero=False)),
    }
    if color:
        enc["color"] = alt.Color(f"{color}:N", title=color, legend=alt.Legend(orient="top"))
    return (
        alt.Chart(df)
        .mark_line(point=points, strokeWidth=3)
        .encode(**enc)
        .properties(height=height, title=title)
    )


def box(df: pd.DataFrame, cat: str, val: str, *, height: int = HEIGHT, title: str = "") -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_boxplot(extent="min-max", size=40)
        .encode(
            x=alt.X(f"{cat}:N", title=cat),
            y=alt.Y(f"{val}:Q", title=val, scale=alt.Scale(zero=False)),
            color=alt.Color(f"{cat}:N", legend=None),
        )
        .properties(height=height, title=title)
    )


def vrule(value: float, *, color: str = "#E8590C", dash: list[int] | None = None) -> alt.Chart:
    """خط عمودي مرجعي (مثلاً حدّ القطع threshold)."""
    return (
        alt.Chart(pd.DataFrame({"v": [value]}))
        .mark_rule(color=color, strokeWidth=2, strokeDash=dash or [6, 4])
        .encode(x="v:Q")
    )


def hrule(value: float, *, color: str = "#E8590C", dash: list[int] | None = None) -> alt.Chart:
    return (
        alt.Chart(pd.DataFrame({"v": [value]}))
        .mark_rule(color=color, strokeWidth=2, strokeDash=dash or [6, 4])
        .encode(y="v:Q")
    )


def reg_line(df: pd.DataFrame, x: str, y: str, *, color: str = "#4C6EF5") -> alt.Chart:
    """خط انحدار بسيط (OLS) لإظهار أثر الشواذ على الميل."""
    d = df[[x, y]].dropna()
    if len(d) < 2:
        return alt.Chart(pd.DataFrame({x: [], y: []})).mark_line()
    b, a = np.polyfit(d[x], d[y], 1)
    xs = np.linspace(d[x].min(), d[x].max(), 50)
    fit = pd.DataFrame({x: xs, y: a + b * xs})
    return alt.Chart(fit).mark_line(color=color, strokeWidth=3).encode(x=f"{x}:Q", y=f"{y}:Q")


def heat(df: pd.DataFrame, x: str, y: str, val: str, *, height: int = 320, title: str = "") -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(f"{x}:N", title=x),
            y=alt.Y(f"{y}:N", title=y),
            color=alt.Color(f"{val}:Q", title=val),
            tooltip=[x, y, val],
        )
        .properties(height=height, title=title)
    )
