import requests

# # Add a new pronunciation
# resp = requests.post(
#     "http://localhost:8880/dev/update_pronunciation",
#     json={"word": "byebye", "phonemes": "/həˈloʊ/"}
# )
#
# # "byebye": "/həˈloʊ/",
# resp.raise_for_status()
# print(resp.json())  # {"status": "ok"}
#
# # Inspect all pronunciations
# print(requests.get("http://localhost:8880/dev/pronunciations").json())
#

print(requests.post(url="http://localhost:8880/debug/reinitialize"))


#
# import json
#
# # Open and read the JSON file
# with open('../api/pronunciations.json', 'r') as file:
#     data = json.load(file)
#
# # Print the dictionary
# print(data)
# for item, phoneme in data.items():
#     print(item, phoneme)
#
#     Inspect all pronunciations
#
#
#
