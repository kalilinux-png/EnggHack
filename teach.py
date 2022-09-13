import time
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


cam= cv2.VideoCapture('video.mp4')
# cam = cv2.VideoCapture(0)
model = load_model('keras_model.h5')

while True:
    result, image = cam.read()
    cv2.imshow("Say Cheese", image)
    time.sleep(2)
    cv2.imwrite("shubh.bmp", image)

    print("Image written to file-system : shubh.bmp")

    # Load the model

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open('shubh.bmp').convert("RGB") #
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print("prediction",prediction)
