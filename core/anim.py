"""محرّك الأنيميشن المشترك في المنصة.

الفكرة: كل أنيميشن = دالة `render(frame)` ترسم إطارًا واحدًا (رسم بياني أو
جدول أو نص). المحرّك يضيف حولها شريط تحكّم (تشغيل / إيقاف / السابق / التالي /
تصفير) ومؤشّر إطار، ويعتمد على `st.fragment(run_every=...)` حتى يتحرّك الرسم
وحده دون إعادة تشغيل الصفحة كاملة.
"""

from __future__ import annotations

from typing import Callable, Sequence

import streamlit as st

RenderFn = Callable[[int], None]


def play_animation(
    key: str,
    n_frames: int,
    render: RenderFn,
    *,
    fps: float = 1.6,
    label: str = "الإطار",
    frame_labels: Sequence[str] | None = None,
    help: str | None = None,
    show_slider: bool = True,
) -> None:
    """يعرض أنيميشن بزر تشغيل.

    Parameters
    ----------
    key : str
        مُعرّف فريد للأنيميشن داخل الصفحة (يُستخدم في مفاتيح session state).
    n_frames : int
        عدد الإطارات.
    render : Callable[[int], None]
        دالة ترسم الإطار رقم `frame` (من 0 إلى n_frames-1).
    fps : float
        عدد الإطارات في الثانية أثناء التشغيل التلقائي.
    frame_labels : قائمة نصوص
        شرح قصير يظهر أسفل شريط التحكّم لكل إطار.
    """
    if n_frames < 1:
        return

    frame_key = f"anim_frame_{key}"
    play_key = f"anim_play_{key}"
    st.session_state.setdefault(frame_key, 0)
    st.session_state.setdefault(play_key, False)

    # نحتفظ بالحالة الحالية لأن `run_every` يُحسم لحظة إنشاء الـ fragment
    playing = bool(st.session_state[play_key])
    interval = 1.0 / max(fps, 0.1)

    def _toggle() -> None:
        st.session_state[play_key] = not st.session_state[play_key]

    def _step(delta: int) -> None:
        st.session_state[play_key] = False
        st.session_state[frame_key] = (st.session_state[frame_key] + delta) % n_frames

    def _reset() -> None:
        st.session_state[play_key] = False
        st.session_state[frame_key] = 0

    @st.fragment(run_every=interval if playing else None)
    def _animation_block() -> None:
        # تغيّرت حالة التشغيل ⇒ نحتاج إعادة تشغيل كاملة ليتغيّر `run_every`
        if bool(st.session_state[play_key]) != playing:
            st.rerun()

        # تقدّم تلقائي (قبل إنشاء المنزلق حتى يُسمح بالكتابة في مفتاحه)
        if st.session_state[play_key]:
            st.session_state[frame_key] = (st.session_state[frame_key] + 1) % n_frames

        is_playing = bool(st.session_state[play_key])
        with st.container(horizontal=True, vertical_alignment="center", wrap=False):
            st.button(
                "إيقاف" if is_playing else "تشغيل",
                icon=":material/pause:" if is_playing else ":material/play_arrow:",
                key=f"anim_toggle_{key}",
                on_click=_toggle,
                type="primary",
                help=help,
            )
            st.button(
                "السابق",
                icon=":material/skip_previous:",
                key=f"anim_prev_{key}",
                on_click=_step,
                args=(-1,),
            )
            st.button(
                "التالي",
                icon=":material/skip_next:",
                key=f"anim_next_{key}",
                on_click=_step,
                args=(1,),
            )
            st.button(
                "تصفير",
                icon=":material/restart_alt:",
                key=f"anim_reset_{key}",
                on_click=_reset,
            )

        if show_slider and n_frames > 1:
            frame = st.slider(
                label,
                min_value=0,
                max_value=n_frames - 1,
                key=frame_key,
                label_visibility="collapsed",
            )
        else:
            frame = int(st.session_state[frame_key])

        if frame_labels is not None and 0 <= frame < len(frame_labels):
            st.markdown(
                f"**:blue-badge[إطار {frame + 1}/{n_frames}]** {frame_labels[frame]}"
            )

        render(frame)

    _animation_block()


def story_animation(
    key: str,
    steps: Sequence[tuple[str, str]],
    render: RenderFn,
    *,
    fps: float = 0.8,
) -> None:
    """أنيميشن «قصة»: كل إطار له عنوان وشرح، مع رسم مرافق.

    `steps` قائمة من (عنوان الخطوة، شرح الخطوة).
    """

    def _render(frame: int) -> None:
        title, body = steps[frame]
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.markdown(body)
            render(frame)

    play_animation(
        key,
        len(steps),
        _render,
        fps=fps,
        label="الخطوة",
        help="شغّل السرد خطوة بخطوة",
    )
