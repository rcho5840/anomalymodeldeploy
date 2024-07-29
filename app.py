from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img
from keras.preprocessing.image import img_to_array
import time
import pickle

app = Flask(__name__)
TF_ENABLE_ONEDNN_OPTS=1

with open('newcnn', 'rb') as f:
    newcnn = pickle.load(f)

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
        
        
def predict():
    
    start_time = time.time()
# Read Images
    imagefile= request.files['imagefile']
    image_path = "C:/Users/randy/OneDrive/Desktop/anomalymodeldeploy/testimages/" + imagefile.filename
    imagefile.save(image_path)
    image = load_img(image_path, target_size=(256, 256))
    image = tf.image.rgb_to_grayscale(image)
    image = img_to_array(image)
    image = np.expand_dims(image / 255, 0)
    yhat = newcnn.predict(image)
    
    if yhat > 0.5:
        classification = "Normal" 
    else:
        classification = "an Anomaly"

    end_time = time.time()
    inference_time = end_time - start_time

    return render_template('index.html', prediction=yhat, inference_time = inference_time)



if __name__ == '__main__':
    app.run(debug=True)