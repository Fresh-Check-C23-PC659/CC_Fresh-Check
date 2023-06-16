import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import base64
import io
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from flask_restful import Api, Resource
from flask_httpauth import HTTPBasicAuth
from flask import Flask, request, jsonify

model = keras.models.load_model("my_model.h5")

class_labels = [
    'fresh_apple',
    'fresh_banana',
    'fresh_bitter_gourd',
    'fresh_capsicum',
    'fresh_orange',
    'fresh_tomato',
    'stale_apple',
    'stale_banana',
    'stale_bitter_gourd',
    'stale_capsicum',
    'stale_orange',
    'stale_tomato'
]

app = Flask(__name__)
api = Api(app, prefix="/api/v1")
auth = HTTPBasicAuth()

USER_DATA = {
    "admin": "admin123"
}
def generate_basic_auth(username, password):
    credentials = f"{username}:{password}"
    credentials_bytes = credentials.encode('ascii')
    base64_credentials = base64.b64encode(credentials_bytes).decode('ascii')
    return base64_credentials


def transform_image(pillow_image):
    pillow_image = pillow_image.convert('RGB')  # Convert image to RGB
    data = np.asarray(pillow_image)
    data = data / 255.0
    data = tf.image.resize(data, [224, 224])
    data = np.expand_dims(data, axis=0)  # Add a batch dimension
    return data

def predict(x):
    predictions = model.predict(x)
    pred0 = predictions[0]
    label0 = np.argmax(pred0)
    predicted_label = class_labels[label0]
    prediction_parts = predicted_label.split("_")
    prediction = prediction_parts[0]
    name = prediction_parts[1]
    return prediction, name

@auth.verify_password
def verify(username, password):
    if not (username and password):
        return False
    return USER_DATA.get(username) == password

class PrivateResource(Resource):
    @app.route("/predict", methods=["GET", "POST"])
    @auth.login_required
    def index():
        if request.method == "POST":
            file = request.files.get('file')
            if file is None or file.filename == "":
                return jsonify({"error": "no file"})

            try:
                image_bytes = file.read()
                pillow_img = Image.open(io.BytesIO(image_bytes)).convert('L')
                tensor = transform_image(pillow_img)
                predicted_label, name_label = predict(tensor)
                data = {
                    "prediction": predicted_label,
                    "name": name_label
                }
                auth_header = request.headers.get('Authorization')
                if auth_header:
                    auth_type, auth_value = auth_header.split()
                    if auth_type.lower() == 'basic':
                        credentials = base64.b64decode(auth_value).decode('ascii')
                        username, password = credentials.split(':')
                        if verify(username, password):
                            return jsonify(data)
                return jsonify({"error": "Unauthorized"})
                # return jsonify(data)
            except Exception as e:
                return jsonify({"error": str(e)})

        return "OK"

api.add_resource(PrivateResource, '/private')

if __name__ == "__main__":
    app.run(debug=True)
