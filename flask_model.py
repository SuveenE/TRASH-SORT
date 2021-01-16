from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
from flask_cors import CORS
import cv2
from PIL import Image
import re
import io
import base64



app = Flask(__name__)
CORS(app)

model = load_model('model.h5')

label_dict = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}


def preprocess(img):
    img = np.array(img)
    img = img / 255.0
    resized = cv2.resize(img, (224, 224))
    reshaped = resized.reshape(1,224, 224, 3)
    return reshaped


@app.route('/')
def index():
    return (render_template('index1.html'))


@app.route('/predict', methods=['POST'])
def predict():
     message = request.get_json()
     encoded = message['image']
     decoded = base64.b64decode(encoded)
     dataBytesIO = io.BytesIO(decoded)
     dataBytesIO.seek(0)
     image = Image.open(dataBytesIO)

     test_image = preprocess(image)


     prediction = model.predict(test_image)
     result = np.argmax(prediction, axis=1)[0]
     accuracy = float(np.max(prediction, axis=1)[0])

     label = label_dict[result]
     print(prediction, result, accuracy)

     response = {"prediction": {"result": label, "accuracy": accuracy}}

     return jsonify(response)


app.run(debug=True)
