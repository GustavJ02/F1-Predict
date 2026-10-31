"""A circuit map for the terminal: the reference lap drawn in braille dots (coloured by sector) with a dot and a code per car.

Each text cell holds 2 x 4 braille dots, so the track keeps its shape at any size; the map is scaled to fit the cell box it is
given (a cell is about twice as tall as wide, which the 2 x 4 grid cancels out).
"""
import numpy as np
from rich.text import Text

from live.race.watch import SECTOR_COLOUR

BRAILLE = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))      # [row][col] dot bits of U+2800
LABEL_SPOTS = ((1, 0), (-4, 0), (0, -1), (0, 1), (1, -1), (1, 1), (-4, -1), (-4, 1), (2, 0), (-5, 0), (0, -2), (0, 2))


def fit(ref, width, height, margin=1):
    """Scale and offset taking the reference points into a box of width x height braille dots: -> function (x, y) -> (col, row) dots."""
    ref = np.asarray(ref, float)
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    scale = min((width - 2 * margin - 1) / span[0], (height - 2 * margin - 1) / span[1])
    off = np.array([margin, margin]) + ((width - 2 * margin - 1) - span[0] * scale) / 2, \
        np.array([0, margin + ((height - 2 * margin - 1) - span[1] * scale) / 2])

    def to_dots(x, y):
        col = off[0][0] + (np.asarray(x, float) - lo[0]) * scale
        row = off[1][1] + (hi[1] - np.asarray(y, float)) * scale           # y up on the map
        return col, row
    return to_dots


class TrackMap:
    """cars: [(code, progress, style, mode)], mode in PUSH / OUT / SLOW / RUN / STOPPED / PIT ('PIT' cars are not drawn).
    sectors: [end of S1, end of S2] lap fractions or None; corners: [(label, fraction)]."""
    def __init__(self, ref, cars, sectors=None, corners=(), flash=False):
        self.ref, self.cars, self.sectors, self.corners, self.flash = np.asarray(ref, float), cars, sectors, corners, flash

    def _sector_style(self, frac):
        if not self.sectors:
            return "dim"
        return SECTOR_COLOUR[sum(frac >= b for b in self.sectors)]

    def __rich_console__(self, console, options):
        W, H = max(options.max_width, 20), max(options.height or 20, 6)
        to_dots = fit(self.ref, W * 2, H * 4)
        n = len(self.ref)
        bits = np.zeros((H, W), int)
        style = [[""] * W for _ in range(H)]
        text = [[" "] * W for _ in range(H)]
        closed = np.vstack([self.ref, self.ref[:1]])
        for i in range(n):                                            # draw each segment densely enough that there are no gaps
            (c0, r0), (c1, r1) = to_dots(*closed[i]), to_dots(*closed[i + 1])
            for t in np.linspace(0, 1, max(int(max(abs(c1 - c0), abs(r1 - r0))) * 2, 2)):
                col, row = int(round(c0 + (c1 - c0) * t)), int(round(r0 + (r1 - r0) * t))
                if 0 <= col < W * 2 and 0 <= row < H * 4:
                    bits[row // 4, col // 2] |= BRAILLE[row % 4][col % 2]
                    style[row // 4][col // 2] = self._sector_style(i / n)
        for y in range(H):
            for x in range(W):
                if bits[y, x]:
                    text[y][x] = chr(0x2800 + bits[y, x])
        taken = set()                                                 # cells holding a label or dot: braille under a label is fine

        def spot(frac):
            col, row = to_dots(*self.ref[int(frac * n) % n])
            return int(col // 2), int(row // 4)
        sx, sy = spot(0)
        if 0 <= sx < W and 0 <= sy < H:
            text[sy][sx], style[sy][sx] = "▌", "bold white"
            taken.add((sx, sy))
        for label, frac in self.corners:                                  # dim corner numbers beside the line, where free
            x, y = spot(frac)
            for dx, dy in ((2, 0), (-len(label) - 1, 0), (0, -1), (0, 1)):
                cells = [(x + dx + k, y + dy) for k in range(len(label))]
                if all(0 <= cx < W and 0 <= cy < H and (cx, cy) not in taken and text[cy][cx] == " " for cx, cy in cells):
                    for (cx, cy), ch in zip(cells, label):
                        text[cy][cx], style[cy][cx] = ch, "dim"
                        taken.add((cx, cy))
                    break
        order = {"STOPPED": 0, "PUSH": 1, "SLOW": 2, "RUN": 3, "OUT": 4}
        for code, frac, car_style, mode in sorted(self.cars, key=lambda c: order.get(c[3], 5)):
            if mode == "PIT":
                continue
            x, y = spot(frac)
            if not (0 <= x < W and 0 <= y < H):
                continue
            dot, label = "●", f"bold {car_style}"
            if mode == "PUSH":
                label = f"bold black on {car_style}"
            elif mode in ("OUT", "SLOW"):
                dot, label = ("▼" if mode == "SLOW" else "○"), "dim"
            elif mode == "STOPPED":
                dot, label = "✖", "bold white on red" if self.flash else "bold red"
            text[y][x], style[y][x] = dot, label if mode != "RUN" else f"bold {car_style}"
            taken.add((x, y))
            for dx, dy in LABEL_SPOTS:
                cells = [(x + dx + k, y + dy) for k in range(len(code))]
                if all(0 <= cx < W and 0 <= cy < H and (cx, cy) not in taken for cx, cy in cells):
                    for (cx, cy), ch in zip(cells, code):
                        text[cy][cx], style[cy][cx] = ch, label
                        taken.add((cx, cy))
                    break
        for y in range(H):
            t = Text(no_wrap=True)
            for x in range(W):
                t.append(text[y][x], style=style[y][x] or None)
            yield t
