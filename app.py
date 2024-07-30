from flask import Flask, render_template, request

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
#from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

app = Flask(__name__)
model = ResNet50()

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "testimages/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]

    classification = '%s (%.2f%%)' % (label[1], label[2]*100)


    return render_template('index.html', prediction=classification)


if __name__ == '__main__':
    app.run(port=3000, debug=True)




# from flask import Flask, render_template, request
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.utils import load_img
# from keras.preprocessing.image import img_to_array
# import time
# import pickle

# app = Flask(__name__)
# TF_ENABLE_ONEDNN_OPTS=1

# with open('newcnn.pkl', 'rb') as f:
#     newcnn = pickle.load(f)

# @app.route('/', methods=['GET'])
# def hello_word():
#     return render_template('index.html')

# @app.route('/', methods=['POST'])
        
        
# def predict():
    
#     start_time = time.time()
# # Read Images
#     imagefile= request.files['imagefile']
#     image_path = "C:/Users/randy/OneDrive/Desktop/anomalymodeldeploy/testimages/" + imagefile.filename
#     imagefile.save(image_path)
#     image = load_img(image_path, target_size=(256, 256))
#     image = tf.image.rgb_to_grayscale(image)
#     image = img_to_array(image)
#     image = np.expand_dims(image / 255, 0)
#     yhat = newcnn.predict(image)
    
#     if yhat > 0.5:
#         classification = "Normal" 
#     else:
#         classification = "an Anomaly"

#     end_time = time.time()
#     inference_time = end_time - start_time

#     return render_template('index.html', prediction=yhat, inference_time = inference_time)



# if __name__ == '__main__':
#     app.run(debug=True)