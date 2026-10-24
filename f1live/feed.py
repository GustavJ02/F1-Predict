"""Shared plumbing for F1 live-timing recordings: decoding, delta merging, file tailing, value parsing."""
import ast, base64, json, re, time, zlib
from datetime import datetime


def decode(d):
    """Payloads come as dicts (deltas), JSON text (keyframes) or base64+deflate (.z topics)."""
    if isinstance(d, dict):
        return d
    try:
        d = json.loads(d)
    except ValueError:
        pass
    if isinstance(d, str):
        d = json.loads(zlib.decompress(base64.b64decode(d), -zlib.MAX_WBITS))
    return d


def merge(dst, src):
    """Apply a partial update. Deltas address list items by string index ({"Sectors": {"1": {...}}})."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), (dict, list)):
            if isinstance(dst[k], list):
                for i, sub in v.items():
                    i = int(i)
                    while len(dst[k]) <= i:
                        dst[k].append({})
                    if isinstance(sub, dict) and isinstance(dst[k][i], dict):
                        merge(dst[k][i], sub)
                    else:
                        dst[k][i] = sub
            else:
                merge(dst[k], v)
        else:
            dst[k] = v


def parse_line(line):
    """One line of a FastF1 recording -> (topic, data, ts) or None. ts == '' for initial keyframes."""
    try:
        topic, data, ts = ast.literal_eval(line.strip())
    except Exception:
        return None
    return topic.removesuffix(".z"), data, ts


def follow(path, live=True, poll=0.2):
    """Yield complete lines from a file; with live=True keep waiting for new ones (like tail -f)."""
    while True:
        try:
            f = open(path)
            break
        except FileNotFoundError:
            if not live:
                raise
            time.sleep(poll)
    with f:
        buf = ""
        while True:
            chunk = f.readline()
            if chunk:
                buf += chunk
                if buf.endswith("\n"):
                    yield buf
                    buf = ""
            elif live:
                time.sleep(poll)
            else:
                if buf:
                    yield buf
                return


def val(x, *path):
    """Safe nested get that works on lists, dicts and dicts-with-string-indices. '' -> None."""
    for p in path:
        if isinstance(x, list):
            x = x[p] if isinstance(p, int) and len(x) > p else None
        elif isinstance(x, dict):
            x = x.get(p, x.get(str(p)) if isinstance(p, int) else None)
        else:
            return None
        if x is None:
            return None
    return None if x in ({}, "", []) else x


def to_sec(t):
    """'1:34.567' -> 94.567, '25.648' -> 25.648."""
    if not t:
        return None
    try:
        m, _, s = str(t).rpartition(":")
        return round((int(m) * 60 if m else 0) + float(s), 3)
    except ValueError:
        return None


_LAPS = re.compile(r"(\d+)\s*L")


def parse_gap(g):
    """Race gaps: '+1.234' -> (1.234, 0), '1L' / '+2 LAPS' -> (None, 2), 'LAP 23' (leader) -> (0.0, 0)."""
    if g is None or g == "":
        return None, None
    g = str(g).strip()
    if g.upper().startswith("LAP"):
        return 0.0, 0
    m = _LAPS.search(g.upper())
    if m:
        return None, int(m.group(1))
    try:
        return float(g.lstrip("+")), 0
    except ValueError:
        return None, None


def iso(t):
    return datetime.fromisoformat(t.rstrip("Z")[:26])
