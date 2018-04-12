from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
#from threading import Thread
import numpy as np
import time
import imutils
import cv2
import os


SANTA=False
MODEL_PATH="./learning/"
MODEL_NAME="newnetowrkmodelcats.h5py"


curr_path = os.getcwd()
os.chdir(MODEL_PATH)
print("[INFO] loading model...")
model = load_model(MODEL_NAME)
os.chdir(curr_path)


print("[INFO] Building ResNet Model...")
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
conv_base = ResNet50(weights='imagenet')

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=1).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)






# loop over the frames from the video stream
while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()

        #frame = imutils.resize(frame, width=400)

        whatisit = decode_predictions(
                conv_base.predict(
                        preprocess_input(
                                np.expand_dims(cv2.resize(frame,
                                                          conv_base.input_shape[1:3]),
                                               axis=0).astype('float32')
                        )))
        print(whatisit)
        time.sleep(1)
        img = cv2.resize(frame, (150, 150))
        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)


        #print(model.predict(image)[0])