"""Where a session's files live: data/<year>/<race>/<session>/{raw.txt,events.csv,predictions.csv}."""
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def folder_name(name):
    """'United States' -> 'United_States', 'São Paulo' -> 'Sao_Paulo': no spaces or accents in folder names."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return "_".join(ascii_name.split())


@dataclass(frozen=True)
class Session:
    year: str
    race: str
    name: str                      # quali, race, sprint, replay, ...
    data_dir: Path = DATA

    def __post_init__(self):                     # accept 'United States' as well as 'United_States'
        object.__setattr__(self, "race", folder_name(self.race))
        object.__setattr__(self, "name", folder_name(self.name))

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
