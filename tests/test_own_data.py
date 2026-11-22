"""Checks for the "Use your own data" adapter (scripts/own_cases.py). No models, no workers."""

import csv
import os
import shutil
import subprocess
import sys

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from atlas_pipeline import config  # noqa: E402
from own_cases import check, lab_vector  # noqa: E402

XRAY = os.path.join(config.XRAY_DIR, "test", "NORMAL", sorted(os.listdir(os.path.join(config.XRAY_DIR, "test", "NORMAL")))[0])
AGE = "Patient age quantile"


def make_folder(path, rows, columns=None, xrays=("a.jpeg",)):
    os.makedirs(path / "xrays")
    for name in xrays:
        shutil.copy(XRAY, path / "xrays" / name)
    columns = columns or sorted({k for r in rows for k in r}, key=lambda c: (c != "case_id", c))
    with open(path / "cases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    return path


def labs(**overrides):
    values = {c: "0.1" for c in config.LAB_COLUMNS}
    values[AGE] = "7"
    values.update(overrides)
    return values


def test_good_folder(tmp_path):
    folder = make_folder(tmp_path, [
        {"case_id": "A", "xray": "a.jpeg", "note": "Cough and fever.", **labs(), "xray_label": "0"},
        {"case_id": "B", "xray": "", "note": "Chest pain on exertion.", **labs(Platelets="")},
    ])
    cases, problems = check(str(folder))
    assert problems == []
    assert [c["case_id"] for c in cases] == ["A", "B"]
    assert cases[0]["labels"] == {"radiology": 0}
    vec = lab_vector(cases[1]["labs_values"])   # same layout as data.lab_features: 14 + 14 flags + age
    assert len(vec) == 29 and vec[-1] == pytest.approx(0.35)
    assert vec[14 + config.LAB_COLUMNS.index("Platelets")] == 1.0


def test_column_names_without_the_datasets_odd_spaces(tmp_path):
    clean = {" ".join(c.replace("\xa0", " ").split()): v for c, v in labs().items()}
    cases, problems = check(str(make_folder(tmp_path, [{"case_id": "A", **clean}])))
    assert problems == [] and "labs_values" in cases[0]


@pytest.mark.parametrize("row,expected", [
    ({"case_id": "A", "xray": "missing.jpeg"}, "xrays/missing.jpeg: named on cases.csv line 2 but not found"),
    ({"case_id": "A", "xray": "scan.dcm"}, "not an image type the encoder reads"),
    ({"case_id": "", "note": "x"}, "empty case_id"),
    ({"case_id": "A", "note": ""}, "case A has no X-ray, note or lab values"),
    ({"case_id": "A", **labs(Platelets="250")}, "Platelets 250 doesn't look standardized"),
    ({"case_id": "A", **labs(Hemoglobin="low")}, "Hemoglobin 'low' is not a number"),
    ({"case_id": "A", **labs(**{AGE: "35"})}, "is not a whole number from 0 to 19"),
    ({"case_id": "A", "note": "x", "note_label": "yes"}, "note_label 'yes' is not 0 or 1"),
    ({"case_id": "A", "note": "x", "heart_rate": "80"}, "unknown column 'heart_rate'"),
])
def test_problems_are_reported(tmp_path, row, expected):
    _, problems = check(str(make_folder(tmp_path, [row])))
    assert any(expected in p for p in problems), problems


def test_duplicate_ids(tmp_path):
    _, problems = check(str(make_folder(tmp_path, [{"case_id": "A", "note": "x"}, {"case_id": "A", "note": "y"}])))
    assert problems == ["cases.csv line 3: duplicate case_id A"]


def test_labs_without_age(tmp_path):
    row = {"case_id": "A", **{c: "0.1" for c in config.LAB_COLUMNS}}
    _, problems = check(str(make_folder(tmp_path, [row])))
    assert any('without "Patient age quantile"' in p for p in problems)


@pytest.mark.parametrize("content,expected", [
    (b"version https://git-lfs.github.com/spec/v1\n", "Git LFS pointer"),
    (b"not an image", "can't open as an image"),
    (b"", "empty file"),
])
def test_bad_xray_files(tmp_path, content, expected):
    folder = make_folder(tmp_path, [{"case_id": "A", "xray": "a.jpeg"}])
    (folder / "xrays" / "a.jpeg").write_bytes(content)
    _, problems = check(str(folder))
    assert len(problems) == 1 and expected in problems[0], problems


def test_tiny_xray(tmp_path):
    folder = make_folder(tmp_path, [{"case_id": "A", "xray": "a.jpeg"}])
    Image.new("L", (40, 40)).save(folder / "xrays" / "a.jpeg")
    assert check(str(folder))[1] == ["xrays/a.jpeg: too small (40x40), needs at least 64x64"]


def test_missing_or_latin1_csv(tmp_path):
    assert "cases.csv: missing" in check(str(tmp_path))[1][0]
    (tmp_path / "cases.csv").write_bytes("case_id,note\nA,Fi\xe8vre\n".encode("latin-1"))
    assert check(str(tmp_path))[1] == ["cases.csv: not UTF-8 text (byte 17)"]


def test_check_only_runs_nothing_on_problems(tmp_path):
    folder = make_folder(tmp_path / "in", [{"case_id": "A", "xray": "missing.jpeg"}])
    run = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "own_cases.py"), "--dir", str(folder)],
                         capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 1
    assert "nothing was run" in run.stdout and "Scoring" not in run.stdout
