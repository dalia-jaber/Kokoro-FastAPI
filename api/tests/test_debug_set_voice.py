from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from api.src.main import app

client = TestClient(app)


def test_set_default_voice():
    voice_manager = AsyncMock()
    voice_manager.list_voices.return_value = ["voice1", "voice2"]

    with (
        patch(
            "api.src.routers.debug.get_voice_manager",
            AsyncMock(return_value=voice_manager),
        ),
        patch("api.src.routers.debug.settings") as mock_settings,
    ):
        mock_settings.default_voice = "voice1"
        response = client.post("/debug/voice", json={"voice": "voice2"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "voice": "voice2"}
    assert mock_settings.default_voice == "voice2"
    assert mock_settings.default_voice_code == "v"


def test_set_default_voice_invalid():
    voice_manager = AsyncMock()
    voice_manager.list_voices.return_value = ["voice1", "voice2"]

    with (
        patch(
            "api.src.routers.debug.get_voice_manager",
            AsyncMock(return_value=voice_manager),
        ),
        patch("api.src.routers.debug.settings"),
    ):
        response = client.post("/debug/voice", json={"voice": "unknown"})

    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["error"] == "voice_not_found"
