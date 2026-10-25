"""Sync the local data directory with the central store (default rpi:/srv/F1-data) using rsync over ssh.

The central store is the master copy; data/ on this machine is a working copy. Usage (from the repo root):

    python -m f1live.sync status                          # what differs, by file type (transfers nothing)
    python -m f1live.sync pull                            # fetch missing events.csv files (the small parsed files)
    python -m f1live.sync pull 2026 Austin quali --what raw telemetry     # more files of one session
    python -m f1live.sync pull --what all 2025            # everything of one year
    python -m f1live.sync push                            # send everything new or grown to the store
    python -m f1live.sync push 2026 Austin quali          # one session (what run_session.sh does while recording)

Optional positional year / race / session narrow it down (omitted = all). --what picks file types:
events, telemetry, raw, raw_telemetry, predictions, track, all (default: events for pull, all for push/status).
Add --dry-run to see what would be transferred.

Safety:
  pull only fetches files that don't exist locally. It never overwrites, so a recording that hasn't been pushed
  yet can't be replaced by an older copy. (To refresh a local file, delete it and pull again.)
  push appends to files that grew (rsync --append-verify, so a live raw.txt only sends its new tail) and skips files
  that are not longer than the copy on the store, so a short or partial local file never replaces a complete one.
  Nothing is ever deleted on either side.

F1_REMOTE overrides the store (host:path or a plain directory), F1_DATA the local directory.
"""
import argparse, subprocess, sys
from pathlib import Path
from f1live.paths import DATA, REMOTE, folder_name

KINDS = {"events": ["events.csv"], "telemetry": ["telemetry.csv"], "raw": ["raw.txt"],
         "raw_telemetry": ["raw_telemetry.txt"], "predictions": ["predictions.csv"]}
RACE_LEVEL = {"track": "track.csv"}          # lives in <year>/<race>/, not in a session folder (see live.race.track)
SSH = "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=3"
VANISHED = 24      # rsync's "some source files vanished", normal while a file is being rewritten


def filters(year, race, session, kinds):
    """rsync filter rules: only <year>/<race>/<session>/<file> for the wanted file types ('*' = any)."""
    y, r, s = year or "*", folder_name(race) if race else "*", folder_name(session) if session else "*"
    names = [n for k in kinds if k in KINDS for n in KINDS[k]]
    rules = ["--exclude=*.tmp", f"--include=/{y}/", f"--include=/{y}/{r}/", f"--include=/{y}/{r}/{s}/"]
    rules += [f"--include=/{y}/{r}/{s}/{n}" for n in names]
    rules += [f"--include=/{y}/{r}/{RACE_LEVEL[k]}" for k in kinds if k in RACE_LEVEL]
    return rules + ["--exclude=*"]


def command(direction, remote, data, rules, dry_run=False, extra=()):
    remote = remote.rstrip("/") + "/"
    src, dst, mode = (remote, str(data) + "/", "--ignore-existing") if direction == "pull" else \
                     (str(data) + "/", remote, "--append-verify")
    cmd = ["rsync", "-rlt", "--partial", "--prune-empty-dirs", mode, *extra]
    if ":" in remote:
        cmd += ["-e", SSH]
    if dry_run:
        cmd.append("-n")
    return cmd + rules + [src, dst]


def run(direction, remote, data, year, race, session, kinds, dry_run=False, quiet=False):
    """-> (exit code, files transferred, bytes). Output is the rsync stats unless quiet."""
    Path(data).mkdir(parents=True, exist_ok=True)
    cmd = command(direction, remote, data, filters(year, race, session, kinds), dry_run, ["--stats"])
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode not in (0, VANISHED):
        print(f"[sync] {direction} failed ({res.returncode}): {res.stderr.strip() or res.stdout.strip()}",
              file=sys.stderr)
        return res.returncode, 0, 0
    files = size = 0
    for line in res.stdout.splitlines():
        if line.startswith("Number of regular files transferred"):
            files = int(line.split(":")[1].replace(",", ""))
        elif line.startswith("Total transferred file size"):
            size = int(line.split(":")[1].split()[0].replace(",", ""))
    if not quiet:
        verb = {"pull": "pulled", "push": "pushed"}[direction]
        print(f"[sync] {'would have ' if dry_run else ''}{verb} {files} files, {size / 1e6:.1f} MB "
              f"({remote} {'->' if direction == 'pull' else '<-'} {data})")
    return 0, files, size


def status(remote, data, year, race, session, kinds):
    """Dry-run both directions and list, per file type, what only the store has and what is ahead locally."""
    rows = {}
    for direction, label in (("pull", "only on store"), ("push", "ahead locally")):
        cmd = command(direction, remote, data, filters(year, race, session, kinds), True,
                      ["--itemize-changes", "--out-format=%i %n %l"])
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode not in (0, VANISHED):
            print(f"[sync] status failed ({res.returncode}): {res.stderr.strip()}", file=sys.stderr)
            return res.returncode
        for line in res.stdout.splitlines():
            flags, _, rest = line.partition(" ")
            if flags[:2] not in ("<f", ">f"):
                continue
            name, _, size = rest.rpartition(" ")
            kind = name.rsplit("/", 1)[-1]
            n, b = rows.setdefault(kind, {}).get(label, (0, 0))
            rows[kind][label] = (n + 1, b + int(size))
    if not rows:
        print("[sync] local and store are identical (for the selected files)")
    for kind, d in sorted(rows.items()):
        print(f"{kind:<18}" + "  ".join(f"{label}: {d[label][0]} files, {d[label][1] / 1e6:.1f} MB"
                                       for label in ("only on store", "ahead locally") if label in d))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pull", "push", "status"):
        p = sub.add_parser(name)
        p.add_argument("year", nargs="?")
        p.add_argument("race", nargs="?", help="folder name, e.g. Austin or United_States")
        p.add_argument("session", nargs="?", help="folder name, e.g. quali, race, fp1")
        p.add_argument("--what", nargs="+", choices=[*KINDS, *RACE_LEVEL, "all"], default=["events"] if name == "pull" else ["all"])
        p.add_argument("--remote", default=REMOTE, help=f"default {REMOTE}")
        p.add_argument("--data-dir", default=DATA, type=Path)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--quiet", action="store_true", help="print errors only")
    a = ap.parse_args(argv)
    kinds = [*KINDS, *RACE_LEVEL] if "all" in a.what else a.what
    if a.cmd == "status":
        return status(a.remote, a.data_dir, a.year, a.race, a.session, kinds)
    return run(a.cmd, a.remote, a.data_dir, a.year, a.race, a.session, kinds, a.dry_run, a.quiet)[0]


if __name__ == "__main__":
    sys.exit(main())
