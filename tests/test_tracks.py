"""Track layouts and timing sectors for a whole calendar."""
import numpy as np
import pandas as pd
import pytest

from f1live.paths import Session
from live.race import tracks as K
from live.race.track import resample
from test_battles import race
from test_track import LAP_S, RADIUS, ring_laps, ring_telemetry


def sector_laps(shares=(1 / 3, 1 / 3, 1 / 3), laps=8):
    h = ring_laps() if laps == 5 else race({"AAA": LAP_S}, {"AAA": 0}, laps=laps)
    h["lap_time_s"] = LAP_S
    for k, share in enumerate(shares):
        h[f"s{k + 1}_s"] = share * LAP_S
    h["driver"] = "10"
    return h


def test_sector_boundaries_are_where_the_car_was_when_it_finished_each_sector():
    tel = ring_telemetry(driver="10", laps=8)
    ref = resample(tel[["x", "y"]].to_numpy(float)[: int(LAP_S / 0.27)])
    found = K.sector_boundaries(ref, tel.assign(driver="10"), sector_laps((0.3, 0.45, 0.25)))
    assert found is not None
    bounds, per_lap = found
    assert bounds == pytest.approx([0.3, 0.75], abs=0.01)          # the car moves at constant angular speed
    assert per_lap.shape[1] == 2 and len(per_lap) >= K.MIN_LAPS


def test_sectors_need_enough_clean_laps_with_matching_sector_times():
    tel = ring_telemetry(driver="10", laps=8)
    ref = resample(tel[["x", "y"]].to_numpy(float)[: int(LAP_S / 0.27)])
    bad = sector_laps()
    bad["s3_s"] += 1.0                                            # sectors no longer add up to the lap time
    assert K.clean_sector_laps(bad).empty and K.sector_boundaries(ref, tel, bad) is None
    wet = sector_laps()
    wet["track_status"] = "VSCDeployed"
    assert K.sector_boundaries(ref, tel, wet) is None


def test_the_sectors_table_has_three_sectors_that_cover_the_lap():
    tel = ring_telemetry(driver="10", laps=8)
    ref = resample(tel[["x", "y"]].to_numpy(float)[: int(LAP_S / 0.27)])
    bounds, per_lap = K.sector_boundaries(ref, tel, sector_laps())
    t = K.sectors_table(ref, bounds, per_lap, "2026/Test/quali")
    assert list(t["sector"]) == [1, 2, 3] and list(t.columns) == K.COLUMNS
    assert t["start_frac"].iloc[0] == 0 and t["end_frac"].iloc[2] == 1
    assert t["length_m"].sum() == pytest.approx(2 * np.pi * RADIUS / 10, rel=0.01)
    assert (t["start_m"].iloc[1:].to_numpy() == t["end_m"].iloc[:2].to_numpy()).all()
    assert t["length_m"].iloc[0] == pytest.approx(t["length_m"].sum() / 3, rel=0.05)


def test_event_folder_prefers_an_existing_folder():
    existing = {"Austin", "Australian"}
    assert K.event_folder("United States Grand Prix", "Austin", "United States", existing) == "Austin"
    assert K.event_folder("Australian Grand Prix", "Melbourne", "Australia", existing) == "Australian"
    assert K.event_folder("São Paulo Grand Prix", "São Paulo", "Brazil", existing) == "Sao_Paulo"
    assert K.event_folder("Mexico City Grand Prix", "Mexico City", "Mexico", existing) == "Mexico_City"


def write_session(data_dir, year, race_, name):
    sess = Session(str(year), race_, name, data_dir)
    sess.dir.mkdir(parents=True)
    sess.events.write_text("x\n")
    sess.telemetry.write_text("x\n")


def test_a_circuit_without_a_race_this_year_is_built_from_last_year(tmp_path):
    write_session(tmp_path, 2025, "Mexico_City", "fp1")
    write_session(tmp_path, 2025, "Mexico_City", "quali")
    write_session(tmp_path, 2026, "Austin", "race")
    write_session(tmp_path, 2025, "United_States", "quali")
    got = K.source_session(2026, "Mexico_City", tmp_path)
    assert (got.year, got.race, got.name) == ("2025", "Mexico_City", "quali")        # quali is preferred
    got = K.source_session(2026, "Austin", tmp_path)
    assert (got.year, got.race, got.name) == ("2026", "Austin", "race")             # this year's wins
    assert K.source_session(2026, "Monaco", tmp_path) is None


def test_a_circuit_keeps_working_across_renamed_folders(tmp_path):
    write_session(tmp_path, 2025, "United_States", "quali")
    got = K.source_session(2026, "Austin", tmp_path)                                  # Austin == United_States
    assert got is not None and got.race == "United_States"


def test_empty_telemetry_is_not_a_source(tmp_path):
    sess = Session("2026", "Austin", "quali", tmp_path)
    sess.dir.mkdir(parents=True)
    sess.events.write_text("x\n")
    sess.telemetry.write_text("")
    assert not K.usable(sess)
