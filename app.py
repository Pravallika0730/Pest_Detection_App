from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__, template_folder='templates')

UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = tf.keras.models.load_model('pest_model.h5')

# Pest classes (based on your dataset)
pest_classes = [
    'Aphid', 'Armyworm', 'Beetle', 'Bollworm', 'Grasshopper',
    'Mites', 'Mosquito', 'Sawfly', 'Stem Borer'
]

def identify_pest(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale like during training

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    pest_type = pest_classes[class_index]
    particle_count = float(np.sum(predictions))  # Convert to float
    return pest_type, particle_count

def get_control_measures(pest_type):
    measures = {
        'Aphid': {
            'damage': 'Weakens plants by sucking sap, causing leaves to curl and distort.',
            'control': 'Use insecticidal soap, neem oil, or introduce ladybugs to control them.'
        },
        'Armyworm': {
            'damage': 'Chews on plant leaves, creating irregular holes and defoliating crops.',
            'control': 'Apply Bacillus thuringiensis (Bt) or neem oil to manage armyworm infestations.'
        },
        'Beetle': {
            'damage': 'Feeds on plant foliage and roots, causing holes in leaves and stunted growth.',
            'control': 'Use insecticidal soap, neem oil, or hand-pick beetles off plants.'
        },
        'Bollworm': {
            'damage': 'Feeds on cotton bolls and other crops, leading to reduced yield and damaged fruit.',
            'control': 'Apply biological control methods such as Bt or use insecticides. Hand-picking larvae is also effective.'
        },
        'Grasshopper': {
            'damage': 'Consumes leaves, stems, and flowers, often causing significant crop loss.',
            'control': 'Use insecticides or introduce natural predators like birds or spiders.'
        },
        'Mites': {
            'damage': 'Sucks plant sap, causing leaves to yellow and dry out, often leading to leaf drop.',
            'control': 'Apply miticides or neem oil to control mite infestations.'
        },
        'Mosquito': {
            'damage': 'Lays eggs in standing water, potentially spreading diseases to plants and animals.',
            'control': 'Use larvicides to kill larvae or insect repellents to reduce mosquito presence.'
        },
        'Sawfly': {
            'damage': 'Feeds on leaves, skeletonizing them or creating notches along leaf edges.',
            'control': 'Apply insecticides or manually remove larvae. Remove affected plants to control spread.'
        },
        'Stem Borer': {
            'damage': 'Bores into stems and stalks, causing structural damage and reduced plant vigor.',
            'control': 'Use appropriate insecticides or biological controls such as parasitic wasps.'
        }
    }
    
    return measures.get(pest_type, {
        'damage': 'No damage information available.',
        'control': 'No control measures available.'
    })

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            try:
                pest_type, particle_count = identify_pest(filename)
                control_measures = get_control_measures(pest_type)

                # Pass the results to the result page
                return render_template('result.html', pest={
                                         'name': pest_type, 
                                         'damage': control_measures['damage'], 
                                        'control': control_measures['control']
                                        })

            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
