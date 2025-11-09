import os
import requests  # NEW: for downloading model at startup
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename  # NEW: safer filenames
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- Paths & config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'upload')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_FILENAME = 'pest_model.keras'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = "https://github.com/Pravallika0730/Pest_Detection_App/releases/download/v1-model/pest_model.keras"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

img_height, img_width = 150, 150  # match training size

def allowed(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def ensure_dirs():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def download_model_if_missing():
    ensure_dirs()
    need = True
    if os.path.exists(MODEL_PATH):
        try:
            size = os.path.getsize(MODEL_PATH)
            print(f"Model file exists, size={size} bytes")
            need = size < 1024  # treat tiny/0-byte file as invalid
        except Exception as e:
            print("Could not stat model file:", e)
            need = True
    if need:
        print("Downloading model from GitHub release...")
        with requests.get(MODEL_URL, stream=True, allow_redirects=True, timeout=300) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"Model downloaded to {MODEL_PATH} (size={os.path.getsize(MODEL_PATH)} bytes)")

download_model_if_missing()

# Load the model using an ABSOLUTE path
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = load_model(MODEL_PATH, compile=False)


# --- Labels & info ---
pest_dict = {
    0: {'name': 'Aphid', 'damage': 'Weakens plants by sucking sap, causing leaves to curl and distort.',
        'control': 'Use insecticidal soap, neem oil, or introduce ladybugs to control them.'},
    1: {'name': 'Armyworm', 'damage': 'Chews on plant leaves, creating irregular holes and defoliating crops.',
        'control': 'Apply Bacillus thuringiensis (Bt) or neem oil to manage armyworm infestations.'},
    2: {'name': 'Beetle', 'damage': 'Feeds on plant foliage and roots, causing holes in leaves and stunted growth.',
        'control': 'Use insecticidal soap, neem oil, or hand-pick beetles off plants.'},
    3: {'name': 'Bollworm', 'damage': 'Feeds on cotton bolls and other crops, leading to reduced yield and damaged fruit.',
        'control': 'Apply biological control methods such as Bt or use insecticides. Hand-picking larvae is also effective.'},
    4: {'name': 'Grasshopper', 'damage': 'Consumes leaves, stems, and flowers, often causing significant crop loss.',
        'control': 'Use insecticides or introduce natural predators like birds or spiders.'},
    5: {'name': 'Mites', 'damage': 'Sucks plant sap, causing leaves to yellow and dry out, often leading to leaf drop.',
        'control': 'Apply miticides or neem oil to control mite infestations.'},
    6: {'name': 'Mosquito', 'damage': 'Lays eggs in standing water, potentially spreading diseases to plants and animals.',
        'control': 'Use larvicides to kill larvae or insect repellents to reduce mosquito presence.'},
    7: {'name': 'Sawfly', 'damage': 'Feeds on leaves, skeletonizing them or creating notches along leaf edges.',
        'control': 'Apply insecticides or manually remove larvae. Remove affected plants to control spread.'},
    8: {'name': 'Stem Borer', 'damage': 'Bores into stems and stalks, causing structural damage and reduced plant vigor.',
        'control': 'Use appropriate insecticides or biological controls such as parasitic wasps.'}
}

# --- Routes ---
@app.route('/')
def home():
    ensure_dirs()  # ensure upload dir exists
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    ensure_dirs()

    if 'pestImage' not in request.files:
        return "No file part", 400

    file = request.files['pestImage']
    if file.filename == '':
        return "No selected file", 400

    if not allowed(file.filename):
        return "Unsupported file type. Please upload a PNG/JPG/JPEG image.", 400

    # Save safely
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Read & preprocess
    img = cv2.imread(filepath)
    if img is None:
        return "Could not read image file.", 400

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_height, img_width))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    pest_type = int(np.argmax(prediction, axis=1)[0])
    pest_info = pest_dict.get(pest_type, {'name': 'Unknown', 'damage': 'N/A', 'control': 'N/A'})

    return render_template('result.html', pest=pest_info)

# --- Entrypoint ---
if __name__ == '__main__':
    # make sure folders exist and bind to platform port
    ensure_dirs()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
