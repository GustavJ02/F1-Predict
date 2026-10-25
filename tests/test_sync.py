"""f1live.sync against a plain directory standing in for the central store."""
import pytest

from f1live import sync


def put(root, rel, text):
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    return f


@pytest.fixture
def dirs(tmp_path):
    return tmp_path / "store", tmp_path / "local"


def run(cmd, dirs, *args):
    store, local = dirs
    return sync.main([cmd, *args, "--remote", str(store), "--data-dir", str(local), "--quiet"])


def test_pull_fetches_only_the_wanted_kind(dirs):
    store, local = dirs
    put(store, "2025/Monaco/race/events.csv", "e")
    put(store, "2025/Monaco/race/raw.txt", "r")
    put(store, "2025/Monaco/race/telemetry.csv", "t")
    assert run("pull", dirs) == 0
    assert (local / "2025/Monaco/race/events.csv").exists()
    assert not (local / "2025/Monaco/race/raw.txt").exists() and not (local / "2025/Monaco/race/telemetry.csv").exists()
    assert run("pull", dirs, "--what", "raw", "telemetry") == 0
    assert (local / "2025/Monaco/race/raw.txt").exists() and (local / "2025/Monaco/race/telemetry.csv").exists()


def test_pull_narrows_by_year_race_session_and_accepts_spaces(dirs):
    store, local = dirs
    for rel in ("2025/United_States/quali", "2025/United_States/race", "2025/Monaco/quali", "2026/United_States/quali"):
        put(store, rel + "/events.csv", "x")
    assert run("pull", dirs, "2025", "United States", "quali") == 0
    assert sorted(p.relative_to(local).as_posix() for p in local.rglob("*.csv")) == ["2025/United_States/quali/events.csv"]


def test_pull_never_overwrites_a_local_file(dirs):
    store, local = dirs
    put(store, "2026/Austin/quali/raw.txt", "old short")
    mine = put(local, "2026/Austin/quali/raw.txt", "newer, longer local recording")
    run("pull", dirs, "--what", "raw")
    assert mine.read_text() == "newer, longer local recording"


def test_push_appends_and_never_shortens_the_store(dirs):
    store, local = dirs
    remote_raw = put(store, "2026/Austin/quali/raw.txt", "line1\n")
    put(local, "2026/Austin/quali/raw.txt", "line1\nline2\n")
    assert run("push", dirs, "2026", "Austin", "quali") == 0
    assert remote_raw.read_text() == "line1\nline2\n"
    put(local, "2026/Austin/quali/raw.txt", "line1\n")                  # a stale, shorter local copy
    run("push", dirs)
    assert remote_raw.read_text() == "line1\nline2\n"


def test_push_skips_tmp_files_and_deletes_nothing(dirs):
    store, local = dirs
    put(local, "2026/Austin/quali/raw.txt", "a")
    put(local, "2026/Austin/quali/events.tmp", "partial")
    keep = put(store, "2025/Old/race/raw.txt", "only on the store")
    run("push", dirs)
    assert (store / "2026/Austin/quali/raw.txt").exists() and not (store / "2026/Austin/quali/events.tmp").exists()
    assert keep.exists()


def test_dry_run_transfers_nothing(dirs):
    store, local = dirs
    put(store, "2025/Monaco/race/events.csv", "e")
    assert run("pull", dirs, "--dry-run") == 0
    assert not list(local.rglob("events.csv"))


def test_unreachable_store_is_an_error_not_a_crash(dirs, capsys):
    store, local = dirs
    assert sync.main(["push", "--remote", "nohost.invalid:/srv/x", "--data-dir", str(local), "--quiet"]) != 0
    assert "[sync] push failed" in capsys.readouterr().err
