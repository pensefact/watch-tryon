from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from src.watch_pipeline.reconstruction import reconstruct_watch, get_cached_mesh


def test_get_cached_mesh_returns_none_when_missing(tmp_path):
    result = get_cached_mesh("nonexistent-ref", cache_dir=tmp_path)
    assert result is None


def test_get_cached_mesh_returns_path_when_exists(tmp_path):
    mesh_file = tmp_path / "test-ref.glb"
    mesh_file.write_bytes(b"fake mesh data")
    result = get_cached_mesh("test-ref", cache_dir=tmp_path)
    assert result == mesh_file


@pytest.mark.asyncio
async def test_reconstruct_watch_calls_api(tmp_path):
    fake_glb = b"\x67\x6c\x54\x46"  # glTF magic bytes

    mock_response_create = MagicMock()
    mock_response_create.status_code = 200
    mock_response_create.json.return_value = {"task_id": "task-123"}

    mock_response_poll = MagicMock()
    mock_response_poll.status_code = 200
    mock_response_poll.json.return_value = {
        "status": "completed",
        "output": {"model_url": "https://example.com/mesh.glb"},
    }

    mock_response_download = MagicMock()
    mock_response_download.status_code = 200
    mock_response_download.content = fake_glb

    with patch("src.watch_pipeline.reconstruction.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response_create)
        mock_client.get = AsyncMock(side_effect=[mock_response_poll, mock_response_download])
        mock_client_cls.return_value = mock_client

        result = await reconstruct_watch(
            photo_path=Path("tests/fixtures/watch_front.png"),
            api_key="test-key",
            cache_dir=tmp_path,
            reference="test-watch",
        )

    assert result.exists()
    assert result.suffix == ".glb"
    assert result.read_bytes() == fake_glb


def test_reconstruct_watch_uses_cache(tmp_path):
    mesh_file = tmp_path / "cached-ref.glb"
    mesh_file.write_bytes(b"cached mesh")
    result = get_cached_mesh("cached-ref", cache_dir=tmp_path)
    assert result == mesh_file
