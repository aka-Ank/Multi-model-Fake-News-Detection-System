
import pytest

from fnews.data.external_storage import external_dataset_path
from fnews.data.visualnews_acquisition import ORIGIN_BYTES, plan


def test_external_dataset_path_uses_configured_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FNEWS_DATA_ROOT", str(tmp_path))
    assert external_dataset_path("downloads", "visualnews") == tmp_path / "downloads" / "visualnews"


def test_missing_external_root_is_rejected(monkeypatch) -> None:
    monkeypatch.delenv("FNEWS_DATA_ROOT", raising=False)
    with pytest.raises(RuntimeError):
        external_dataset_path("downloads")


def test_visualnews_plan_tracks_external_destination(tmp_path) -> None:
    result = plan(tmp_path)
    assert result["expected_bytes"] == ORIGIN_BYTES
    assert result["destination"].endswith("downloads/visualnews/origin.tar")
