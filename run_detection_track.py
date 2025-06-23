import os
import json
import requests

# Detection endpoint
url = "http://0.0.0.0:7007/detection/detect"

# Explicitly use environment variables provided by Docker scripts
images_dir = os.environ.get("images_dir")
save_dir = os.environ.get("save_dir")

protocol_path = os.path.join(images_dir, "detection_track.txt")
responses = []

# Read protocol file and handle explicitly
with open(protocol_path, "r") as f:
    lines = [line.strip().split()[0] for line in f if line.strip()]  # explicitly extract image filenames
    for image_name in lines:
        image_path = os.path.join(images_dir, image_name)

        if not os.path.isfile(image_path):
            print(f"Warning: Image '{image_path}' not found. Skipping.")
            continue

        params = {"image_path": image_path}
        response = requests.get(url, params=params)

        try:
            response_json = response.json()
        except json.JSONDecodeError:
            response_json = {"image_path": image_path, "score": "ERROR", "decision": "ERROR", "comment": "Invalid JSON response"}

        print(response_json)
        responses.append(response_json)

# Save responses
result_file_path = os.path.join(save_dir, "detection_track.json")
with open(result_file_path, "w") as f:
    json.dump(responses, f, indent=4)

print(f"Results saved to '{result_file_path}'.")
