from flask import Flask
from flask import request
from flask_cors import CORS
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
# from flask_restplusrestplus import Api, Resource, fields
import pickle
import numpy as np
import json
import datetime
import zipfile
app = Flask(__name__)
CORS(app)
model = pickle.load(open('RandomForestGridModel.pkl', 'rb'))
zip_file='appleandbananas.zip'
try:
    with zipfile.ZipFile(zip_file) as z:
        z.extractall()
        print("Extracted all")
except:
    print("Invalid file")

modelImage = load_model('appleandbananas.h5')


@app.route('/')
def home():
    return '<h1>Welcome to flask api</h1>'


@app.route('/heartdisease', methods=['POST'])
def predict_outcome():
    data = request.json
    data_array = np.array([data])
    outcome = model.predict(data_array)
    predicTionResult = 0
    print(outcome)
    if outcome == 1:
        prediction = "You have a heart disease"
        predicTionResult = 1
    else:
        prediction = "You dont a have heart disease"
    dictionary = {'predictionMessage': prediction, 'prediction': predicTionResult}
    json_string = json.dumps(dictionary, indent=4)
    return json_string


@app.route('/binaryimagedetection', methods=['POST'])
def image_api():
    image_file = request.files['image']
    x = datetime.datetime.now()
    file_name = x.strftime("%f")
    image_path = "./images/" + 'test' + file_name + '.jpeg'
    image_file.save(image_path)
    result = predictImage(image_path)
    if result:
        os.remove(image_path)
    return result


def predictImage(filename):
    img1 = image.load_img(filename, target_size=(224, 224))
    Y = image.img_to_array(img1)
    X = np.expand_dims(Y, axis=0)
    val = modelImage.predict(X)
    if val == 1:
        return 'banana'
    elif val == 0:
        return 'apple'


if __name__ == '__main__':
    print("hello main")
    app.run()
