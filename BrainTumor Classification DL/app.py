import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Initialize Flask app
app = Flask(__name__)

# Load and define the model
def build_model():
    base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
    x = base_model.output
    flat = Flatten()(x)
    class_1 = Dense(4608, activation='relu')(flat)
    drop_out = Dropout(0.2)(class_1)
    class_2 = Dense(1152, activation='relu')(drop_out)
    output = Dense(2, activation='softmax')(class_2)
    model = Model(inputs=base_model.input, outputs=output)
    return model

model = build_model()

# Load your model weights if you have any
weights_path = 'path_to_weights.h5'
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Model weights loaded successfully.")
else:
    print("Model weights not found.")

# Function to get the class name based on the prediction
def get_class_name(class_no):
    return "No Brain Tumor" if class_no == 0 else "Yes Brain Tumor"

# Function to preprocess and predict the result
def get_result(img_path):
    image = cv2.imread(img_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((240, 240))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)

    print("Input image shape:", input_img.shape)  # Debugging line

    result = model.predict(input_img)
    print("Model prediction:", result)  # Debugging line

    result_class = np.argmax(result, axis=1)[0]
    return result_class

# Home route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
        file.save(upload_path)

        print("File saved to:", upload_path)  # Debugging line

        prediction = get_result(upload_path)
        result = get_class_name(prediction)
        print("Prediction result:", result)  # Debugging line

        return result
    return None

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
