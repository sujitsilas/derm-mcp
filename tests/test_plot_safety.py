"""Figure building under concurrency.

The MCP SDK runs every sync tool in a worker thread, while matplotlib's rcParams
and figure registry are process-global. Two plotting tools at once used to
interleave their rc_context enter/exit, so a figure was drawn with whatever
profile the other thread had just installed — silently, with both tools
reporting success.
"""

from __future__ import annotations

import threading
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from skinmcp.style.rcparams import style  # noqa: E402


def test_concurrent_styles_do_not_bleed():
    """Each thread must see its own rcParams for the whole of its block.

    The sleep stands in for drawing: without it the check runs in the same GIL
    slice as the update and the race is invisible, which is exactly why this
    survived so long.
    """
    seen_wrong = []

    def worker(size: float) -> None:
        for _ in range(40):
            with style(**{"font.size": size}):
                time.sleep(0.0005)
                if plt.rcParams["font.size"] != size:
                    seen_wrong.append(size)

    threads = [threading.Thread(target=worker, args=(s,)) for s in (7.0, 13.0, 21.0)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert seen_wrong == [], f"{len(seen_wrong)} figures drawn with another thread's style"


def test_failed_plot_does_not_leak_a_figure():
    """A tool raising between subplots() and close() must not leak."""
    plt.close("all")
    try:
        with style():
            plt.subplots()
            raise RuntimeError("boom mid-plot")
    except RuntimeError:
        pass
    assert plt.get_fignums() == []


def test_nested_style_leaves_the_outer_figure_alone():
    """panels.* applies its own profile inside a caller that already holds one."""
    plt.close("all")
    with style():
        outer, _ = plt.subplots()
        with style("standard"):
            inner, _ = plt.subplots()
            plt.close(inner)
        assert plt.fignum_exists(outer.number), "nested style() closed the caller's figure"
        plt.close(outer)
