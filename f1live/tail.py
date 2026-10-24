"""Incrementally read a CSV that another process appends to."""
import io, os
import pandas as pd


class CsvTail:
    """read() -> DataFrame of new complete rows, None if nothing new, 'reset' if the file was rewritten.

    `prepare` post-processes each batch of new rows (dtype conversion etc.)."""
    def __init__(self, path, prepare=lambda df: df):
        self.path, self.prepare = path, prepare
        self.offset, self.buf, self.header = 0, "", None

    def read(self):
        try:
            size = os.path.getsize(self.path)
        except FileNotFoundError:
            return None
        if size < self.offset:                      # rewritten from the start (parser restarted)
            self.offset, self.buf, self.header = 0, "", None
            return "reset"
        if size == self.offset:
            return None
        with open(self.path) as f:
            f.seek(self.offset)
            self.buf += f.read()
            self.offset = f.tell()
        complete, _, self.buf = self.buf.rpartition("\n")   # keep a half-written line for next time
        lines = [l for l in complete.splitlines() if l.strip()]
        if self.header is None and lines:
            self.header, lines = lines[0], lines[1:]
        if not lines:
            return None
        return self.prepare(pd.read_csv(io.StringIO("\n".join([self.header] + lines))))
