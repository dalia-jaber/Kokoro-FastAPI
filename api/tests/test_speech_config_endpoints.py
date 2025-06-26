from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.src.main import app
from api.src.routers import openai_compatible

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_config():
    openai_compatible.speech_config = openai_compatible.SpeechConfig()
    yield
    openai_compatible.speech_config = openai_compatible.SpeechConfig()


@pytest.fixture
def mock_tts_service():
    with patch("api.src.routers.openai_compatible.get_tts_service") as mock_get:
        service = AsyncMock()
        service.generate_audio.return_value = openai_compatible.AudioChunk(
            np.zeros(100, np.int16)
        )
        async def mock_stream(*args, **kwargs):
            if False:
                yield b""
        service.generate_audio_stream = mock_stream
        service.list_voices.return_value = ["af_heart", "new_voice"]
        mock_get.return_value = service
        mock_get.side_effect = None
        yield service


def test_update_base_config_affects_defaults(mock_tts_service):
    resp = client.post(
        "/dev/speech/config/base",
        json={"voice": "new_voice", "speed": 1.5},
    )
    assert resp.status_code == 200

    client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": "hi", "stream": False},
    )
    mock_tts_service.generate_audio.assert_called()
    kwargs = mock_tts_service.generate_audio.call_args[1]
    assert kwargs["voice"] == "new_voice"
    assert kwargs["speed"] == 1.5


def test_update_advanced_config_affects_stream(mock_tts_service):
    resp = client.post(
        "/dev/speech/config/advanced",
        json={"stream": False, "response_format": "mp3"},
    )
    assert resp.status_code == 200

    client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": "hello", "stream": False},
    )
    mock_tts_service.generate_audio.assert_called_once()

