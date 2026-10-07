"""Shared fixtures. Every test runs in a throwaway project root with the test lock closed."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ecgaug import paths
from ecgaug.data.fake import write_fake_mitdb
from ecgaug.data.records import DS1, DS2

REPO = Path(__file__).resolve().parents[1]
FAKE_DS1 = (101, 106, 108, 114, 118, 201)
FAKE_DS2 = (100, 213)


@pytest.fixture(autouse=True)
def tmp_root(tmp_path, monkeypatch):
    """Point the package at an empty temporary root and clear every test-access flag."""
    monkeypatch.setenv("ECGAUG_ROOT", str(tmp_path))
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    monkeypatch.delenv("ALLOW_TEST_BUILD", raising=False)
    return tmp_path


@pytest.fixture(scope="session")
def _fake_raw_small(tmp_path_factory):
    raw = tmp_path_factory.mktemp("fake_small")
    return raw, write_fake_mitdb(raw, FAKE_DS1 + FAKE_DS2, seconds=90.0)


@pytest.fixture(scope="session")
def _fake_raw_full(tmp_path_factory):
    raw = tmp_path_factory.mktemp("fake_full")
    return raw, write_fake_mitdb(raw, DS1 + DS2, seconds=40.0)


@pytest.fixture
def fake_root(tmp_root, _fake_raw_small):
    """Temporary root holding fake raw records for a few DS1 and DS2 record numbers."""
    raw, written = _fake_raw_small
    shutil.copytree(raw, paths.raw_dir())
    return tmp_root, written


@pytest.fixture
def full_fake_root(tmp_root, _fake_raw_full):
    """Temporary root holding fake raw records for all 44 non-paced record numbers."""
    raw, written = _fake_raw_full
    shutil.copytree(raw, paths.raw_dir())
    return tmp_root, written
