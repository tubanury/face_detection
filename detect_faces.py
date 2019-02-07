import argparse
import os
import urllib
from urllib.request import urlopen

import cv2
import gridfs
import numpy as np
import uuid as uuid
from pymongo import MongoClient

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

current_path = os.getcwd()

# create a gridfs instance for storing images
client = MongoClient('localhost', 27017)

# check if db exists
dbnames = client.list_database_names()
# if not 'face_database' in dbnames:
db = client['face_database']

myFaceColl = db['myFaceCollection']
fs = gridfs.GridFS(db)
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# 'http://10.154.3.199:8000/cgi-bin/camera?resolution=320&page=1548066835550&Language=0'

req = urllib.request.urlopen('https://cdn.arstechnica.net/wp-content/uploads/2016/05/person-of-interest-creators-preview-an-exciting-ex_sfxg.640-640x360.jpg')
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
image = cv2.imdecode(arr, -1)

# image = cv2.rotate(image, rotateCode=cv2.ROTATE_180)

(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()
countForNaming = 0

for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > args["confidence"]:
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        # crop the face by rectangle
        crop_face = image[startY:endY, startX:endX]
        # create jpeg name via random number
        jpeg_name = '/home/tugba/Desktop/git-repos/face_detection/face_database/'+ str(uuid.uuid4()) + '.jpg'
        cv2.imwrite(jpeg_name, crop_face)
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
