
from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

def model_predict(img_path):
    
    np.set_printoptions(suppress=True)
    
    # Load the model
    model = tensorflow.keras.models.load_model('ripeness.h5')
    
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Replace this with the path to your image
    image = Image.open(img_path)
    
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
    #turn the image into a numpy array
    image_array = np.asarray(image)
    
    # display the resized image
    #image.show()
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array
    
    # run the inference
    preds = ""
    prediction = model.predict(data)
    # max_val = np.amax(prediction)*100
    # max_val = "%.2f" % max_val
    if np.argmax(prediction)==0:
        preds = f"Unripe😑"
    elif np.argmax(prediction)==1:
        preds = f"Overripe😫"
    else :
        preds = f"ripe😄"

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)

#         # Make prediction
#         preds = model_predict(file_path)
        preds = model_predict(f)
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)
