import json

from src.api.live_store import LiveSessionStore
from src.config import USEFUL_PIDS


def test_rows_csv_has_header_and_dedupes_by_elapsed(tmp_path):
    store = LiveSessionStore(tmp_path / "s1")
    row = {pid: 1.0 for pid in USEFUL_PIDS}
    store.append_row(elapsed_s=1, row=row)
    store.append_row(elapsed_s=1, row=row)  # same second -> ignored
    store.append_row(elapsed_s=2, row=row)
    store.close()

    lines = (tmp_path / "s1" / "rows.csv").read_text().strip().splitlines()
    assert lines[0].split(",")[0] == "elapsed_s"
    assert len(lines) == 3  # header + 2 unique seconds


def test_nan_pid_serialised_as_empty_cell(tmp_path):
    store = LiveSessionStore(tmp_path / "s2")
    row = {pid: 1.0 for pid in USEFUL_PIDS}
    row["TIMING_ADVANCE"] = float("nan")
    store.append_row(elapsed_s=1, row=row)
    store.close()

    header, data = (tmp_path / "s2" / "rows.csv").read_text().strip().splitlines()
    idx = header.split(",").index("TIMING_ADVANCE")
    assert data.split(",")[idx] == ""


def test_marks_written_immediately_with_elapsed(tmp_path):
    store = LiveSessionStore(tmp_path / "s3")
    store.record_mark(state="start", elapsed_s=42)
    store.record_mark(state="stop", elapsed_s=99)

    marks = json.loads((tmp_path / "s3" / "marks.json").read_text())
    assert [m["state"] for m in marks] == ["start", "stop"]
    assert [m["elapsed_s"] for m in marks] == [42, 99]
    store.close()


def test_alerts_written_immediately(tmp_path):
    store = LiveSessionStore(tmp_path / "s1")
    store.record_alert({"kind": "stable", "fault_type": "fuel_system",
                        "confidence": 0.91, "elapsed_s": 130})
    store.record_alert({"kind": "rule", "rule": "ect_sensor_frozen", "elapsed_s": 95})
    data = json.loads((tmp_path / "s1" / "alerts.json").read_text())
    assert len(data) == 2
    assert data[0]["fault_type"] == "fuel_system"
    store.close()
