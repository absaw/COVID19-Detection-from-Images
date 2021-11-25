from flask import Flask, render_template, jsonify, redirect, request
from tensorflow.keras.models import load_model
# import joblib
import cv2
import efficientnet.keras as efn
import numpy as np
from PIL import Image

app = Flask(__name__)


@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")


@app.route("/detect", methods=['GET'])
def detectGet():
    return render_template("upload.html")


@app.route('/detect', methods=['POST'])
def detectPost():
    if request.method == 'POST':
        # image = request.files['file']
        # f.save(f.filename)
        # image = np.array(image)
        #read image file string data
        '''
        filestr = request.files['file'].read()
        npimg = np.fromstring(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.CV_LOAD_IMAGE_UNCHANGED)'''
        image = Image.open(request.files['file'])
        image = np.array(image)
        # image = cv2.imread(f.filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.array(image) / 255
        image = np.expand_dims(image, axis=0)

        option = request.form.getlist('options')
        print(option)
        if(option[0] == 'Chest X-Ray'):
            model = load_model("efficientNet.h5")
            print("Efficinet used")
        else:
            model = load_model("efficientNet_CT.h5")
            print("efficientNet_CT used")

        y_pred = model.predict(image)
        y_pred_bin = np.argmax(y_pred, axis=1)
        print(y_pred)
        probability = y_pred[0][0]*100
		
        if probability > 50:
            result = "covid"
        else:
            result = "nonCovid"
            probability = 100-probability
        print(probability)
        print(result)
        return render_template("upload.html", probability=probability, result=result)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
