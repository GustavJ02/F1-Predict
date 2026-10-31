"""Delay for the live viewers: a TV broadcast usually runs behind the live timing, so the viewers can show the session as it was
N seconds ago. The keys ] [ change the delay by 1 s, } { by 10 s, 0 goes back to live; the last delay is remembered per data folder
(shared by all viewers: it is the same broadcast)."""
import os, sys, time
from pathlib import Path

MAX_DELAY = 300         # s: the longest delay a viewer can show
STEPS = {"]": 1.0, "[": -1.0, "}": 10.0, "{": -10.0}


def adjust_delay(delay, key, maximum=MAX_DELAY):
    """New delay in seconds after a key press: ] [ change it by 1 s, } { by 10 s, 0 goes back to live. Between 0 and `maximum`."""
    if key == "0":
        return 0.0
    return min(max(delay + STEPS.get(key, 0.0), 0.0), float(maximum))


def delay_file(data_dir):
    return Path(data_dir) / ".watch_delay"


def load_delay(data_dir, maximum=MAX_DELAY):
    """The delay used last time (broadcast delay rarely changes), 0 if none was saved."""
    try:
        return min(max(float(delay_file(data_dir).read_text().strip()), 0.0), float(maximum))
    except (OSError, ValueError):
        return 0.0


def save_delay(data_dir, delay):
    try:
        delay_file(data_dir).write_text(f"{delay:g}\n")
    except OSError:
        pass


class Keys:
    """Read single key presses without waiting for Enter (a no-op unless stdin is a terminal)."""
    def __init__(self):
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None

    def __enter__(self):
        if self.fd is not None:
            import termios, tty
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)                 # Ctrl-C still works
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read(self, timeout):
        """The keys pressed within `timeout` seconds (waits that long if none), as a string."""
        if self.fd is None:
            time.sleep(timeout)
            return ""
        import select
        if select.select([sys.stdin], [], [], timeout)[0]:
            return os.read(self.fd, 32).decode(errors="ignore")
        return ""
