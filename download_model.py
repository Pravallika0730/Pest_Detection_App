# download_model.py
import os, requests

MODEL_URL = "https://github.com/Pravallika0730/Pest_Detection_App/releases/download/v1-model/pest_model.keras"
DEST_DIR = "models"
DEST_PATH = os.path.join(DEST_DIR, "pest_model.keras")

os.makedirs(DEST_DIR, exist_ok=True)

if not os.path.exists(DEST_PATH):
    print("Downloading model...")
    with requests.get(MODEL_URL, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        with open(DEST_PATH, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    print("Model downloaded to", DEST_PATH)
else:
    print("Model already present:", DEST_PATH)
