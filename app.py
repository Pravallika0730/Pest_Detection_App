import os
from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Path to store uploaded images
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained pest detection model
model = load_model('models/pest_model.keras', compile=False)  # Load the final model saved

# Define image dimensions
img_height, img_width = 150, 150  # Match the size used during training

# Dictionary to map model output to pests and control measures
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pestImage' not in request.files:
        return "No file part"
    
    file = request.files['pestImage']

    if file.filename == '':
        return "No selected file"
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load and preprocess the image
        img = cv2.imread(filepath)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Keras generators use RGB

        # Resize the image to the input size the model expects
        expected_size = (img_height, img_width)  # Use the same size as in training
        img = cv2.resize(img, expected_size)

        # Normalize the image
        ##img = img / 255.0
        img = img.astype("float32") / 255.0

        # Expand dimensions to match the input shape
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = model.predict(img)
        pest_type = int(np.argmax(prediction, axis=1)[0])
        #pest_type = np.argmax(prediction)

        # Get pest info based on prediction
        pest_info = pest_dict[pest_type]
        print(pest_info)

        return render_template('result.html', pest=pest_info)

if __name__ == '__main__':
    app.run(debug=True)
