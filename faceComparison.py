from sklearn import neighbors
import os
import os.path
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

import cv2
import numpy as np
from itertools import chain

from deepface import DeepFace


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
unknown_dir = os.path.join(BASE_DIR, "unknown")
card_dir = os.path.join(BASE_DIR, "card")
face_dir = os.path.join(BASE_DIR, "face")

def compare():

    try:
        card_image = face_recognition.load_image_file('card/card.jpg')
        face_image = face_recognition.load_image_file('face/face.jpg')

        card_encoding = face_recognition.face_encodings(card_image)[0]
        face_encoding = face_recognition.face_encodings(face_image)[0]

        return face_recognition.compare_faces([card_encoding], face_encoding)
    except:
        return ["Face not detected"]
