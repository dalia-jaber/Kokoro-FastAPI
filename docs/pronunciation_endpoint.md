# Runtime Pronunciation Endpoint

This API exposes development routes for managing a pronunciation dictionary while the service is running.

## Endpoints

- `POST /dev/update_pronunciation` – Add or update a word's phoneme sequence.
- `GET /dev/pronunciations` – View the current dictionary contents.

The dictionary is stored in `pronunciations.json` by default. You can change the path with the `PRONUNCIATION_DICT_PATH` environment variable.

## Example

```python
import requests

# Add a new pronunciation
resp = requests.post(
    "http://localhost:8880/dev/update_pronunciation",
    json={"word": "llama", "phonemes": "l ɑː m ə"}
)
resp.raise_for_status()
print(resp.json())  # {"status": "ok"}

# Inspect all pronunciations
print(requests.get("http://localhost:8880/dev/pronunciations").json())
```
