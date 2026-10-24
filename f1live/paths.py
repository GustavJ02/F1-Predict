"""Where a session's files live: data/<year>/<race>/<session>/{raw.txt,events.csv,predictions.csv}."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@dataclass(frozen=True)
class Session:
    year: str
    race: str
    name: str                      # quali, race, sprint, replay, ...
    data_dir: Path = DATA

    @classmethod
    def parse(cls, spec, data_dir=DATA):
        """'2026/Austin/quali' -> Session"""
        parts = spec.strip("/").split("/")
        if len(parts) != 3:
            raise ValueError(f"expected year/race/session, got '{spec}'")
        return cls(*parts, data_dir=Path(data_dir))

    @property
    def dir(self):
        return self.data_dir / self.year / self.race / self.name

    @property
    def raw(self):
        return self.dir / "raw.txt"

    @property
    def events(self):
        return self.dir / "events.csv"

    @property
    def predictions(self):
        return self.dir / "predictions.csv"

    def __str__(self):
        return f"{self.year}/{self.race}/{self.name}"


def add_session_args(ap):
    ap.add_argument("year")
    ap.add_argument("race", help="folder name, e.g. Austin")
    ap.add_argument("session", help="folder name, e.g. quali, race, sprint")
    ap.add_argument("--data-dir", default=DATA, type=Path)


def session_from_args(a):
    return Session(a.year, a.race, a.session, a.data_dir)
