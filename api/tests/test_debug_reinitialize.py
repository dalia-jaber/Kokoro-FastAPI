from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from api.src.main import app

client = TestClient(app)


def test_reinitialize_endpoint():
    model_manager = AsyncMock()
    voice_manager = AsyncMock()
    model_manager.initialize_with_warmup.return_value = ("cpu", "kokoro_v1", 1)

    with (
        patch(
            "api.src.inference.model_manager.get_manager",
            AsyncMock(return_value=model_manager),
        ),
        patch(
            "api.src.inference.voice_manager.get_manager",
            AsyncMock(return_value=voice_manager),
        ),
    ):
        response = client.post("/debug/reinitialize")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "device": "cpu",
        "model": "kokoro_v1",
        "voice_packs": 1,
    }
    model_manager.unload_all.assert_called_once()
    model_manager.initialize_with_warmup.assert_called_once_with(voice_manager)
