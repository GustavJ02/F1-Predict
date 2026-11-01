"""A text chart of how the predicted Q1 / Q2 cutoff moved during the part, for the predictions panel of the qualifying dashboard.

Braille dots (2 x 4 per text cell, like live.practice.trackmap): a bright line for the projected cutoff, a lightly stippled band between the
model's lowest and highest cutoff, and a red dashed line for the cutoff as it stands now (the slowest time inside the advancing
places). The time axis runs from the first projection of the part to the latest, so the picture compresses as the part goes on and
always shows all of it. Rows beyond the room the layout has are not drawn: no room, no chart.
"""
import numpy as np
import pandas as pd
from rich.text import Text

BRAILLE = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))      # [row][col] dot bits of U+2800
PRED, NOW, BAND = "bold cyan", "red", "blue"
GUTTER = 9                                                               # width of the lap time labels ("1:18.521")
MIN_ROWS = 4
MAX_ROWS = 10
MIN_WIDTH = 24


def lap_label(seconds, _pos=None):
    """78.493 -> '1:18.493'."""
    m, s = divmod(float(seconds), 60)
    return f"{int(m)}:{s:06.3f}"


def part_rows(preds):
    """The projections of the part that is running (the part of the latest row), time-ordered: (part, DataFrame) or (None, None).
    Q3 has pole probabilities instead of a cutoff, so there is no chart then, not even of Q2: a Q3 row has no projection, and the
    running part is read from the latest row before rows without a projection are dropped."""
    if preds is None or preds.empty or not {"part", "pred", "lo", "hi", "ts"} <= set(preds.columns):
        return None, None
    parts = preds["part"].dropna()
    if parts.empty or int(parts.iloc[-1]) not in (1, 2):
        return None, None
    part = int(parts.iloc[-1])
    d = preds[(preds["part"] == part) & preds["pred"].notna()]
    return (part, d.sort_values("ts")) if len(d) else (None, None)


class CutoffChart:
    """Renderable of `rows` text rows of chart plus an axis row and a legend row (rows + 2 lines)."""
    def __init__(self, d, rows):
        self.d, self.rows = d, rows

    def _series(self, cols, t0, span):
        """Per dot column: the latest projection at or before that moment (a step), as arrays of pred / lo / hi / current."""
        x = ((self.d["ts"] - t0).dt.total_seconds().to_numpy() / span * (cols - 1))
        idx = np.searchsorted(x, np.arange(cols), side="right") - 1
        ok = idx >= 0
        idx = np.maximum(idx, 0)
        pick = lambda name: np.where(ok, self.d[name].to_numpy(float)[idx], np.nan)
        return pick("pred"), pick("lo"), pick("hi"), (pick("current") if "current" in self.d else np.full(cols, np.nan))

    def __rich_console__(self, console, options):
        W = max(options.max_width - GUTTER - 1, 8)                       # chart cells; GUTTER + a separator column to the left
        H = self.rows
        cols, dots = W * 2, H * 4
        d = self.d
        t0, t1 = d["ts"].iloc[0], d["ts"].iloc[-1]
        span = max((t1 - t0).total_seconds(), 1.0)
        pred, lo, hi, cur = self._series(cols, t0, span)
        low, high = np.nanmin(lo), np.nanmax(hi)
        pad = 0.06 * max(high - low, 0.05)
        low, high = low - pad, high + pad
        row_of = lambda v: np.clip(np.rint((high - v) / (high - low) * (dots - 1)), 0, dots - 1).astype(int)
        bits = np.zeros((H, W), int)
        kinds = np.zeros((H, W), int)                                   # 0 empty, 1 band, 2 cutoff now, 3 projection (wins)

        def put(c, r, kind):
            bits[r // 4, c // 2] |= BRAILLE[r % 4][c % 2]
            kinds[r // 4, c // 2] = max(kinds[r // 4, c // 2], kind)

        for c in range(cols):
            if np.isnan(pred[c]):
                continue
            top, bottom = row_of(hi[c]), row_of(lo[c])
            for r in range(top, bottom + 1):                            # stippled band, solid edges
                if (r % 2 == 0 and c % 2 == 0) or r in (top, bottom):
                    put(c, r, 1)
            if not np.isnan(cur[c]) and low <= cur[c] <= high and c % 2 == 0:
                put(c, row_of(cur[c]), 2)
        prev = None
        for c in range(cols):                                           # the projection as a connected line
            if np.isnan(pred[c]):
                prev = None
                continue
            r = row_of(pred[c])
            for rr in range(min(r, prev), max(r, prev) + 1) if prev is not None else [r]:
                put(c, rr, 3)
            prev = r
        labels = {0: lap_label(high), H // 2: lap_label((low + high) / 2), H - 1: lap_label(low)}
        for y in range(H):
            line = Text(no_wrap=True)
            line.append(f"{labels.get(y, ''):>{GUTTER}}", style="dim")
            line.append("│", style="dim")
            for x in range(W):
                k = kinds[y, x]
                line.append(chr(0x2800 + bits[y, x]) if bits[y, x] else " ", style={1: BAND, 2: NOW, 3: PRED}.get(k))
            yield line
        axis = Text(no_wrap=True)
        left, right = f"{t0:%H:%M:%S}", f"now {t1:%H:%M:%S}"
        axis.append(" " * GUTTER + "└", style="dim")
        axis.append(left + " " * max(W - len(left) - len(right), 1) + right, style="dim")
        yield axis
        legend = Text(no_wrap=True)
        legend.append(" " * (GUTTER + 1))
        legend.append("⠶ projected cutoff", style=PRED)
        legend.append("   ⠒ 90% band", style=BAND)
        legend.append("   ⠤ cutoff now", style=NOW)
        yield legend


def cutoff_chart(preds, rows, width=None):
    """The chart for the running part's projections with `rows` chart rows, or None: no projections, Q3, too few rows or too narrow."""
    part, d = part_rows(preds)
    if d is None or rows < MIN_ROWS or (width is not None and width < MIN_WIDTH + GUTTER + 1):
        return None
    return CutoffChart(d, rows)
