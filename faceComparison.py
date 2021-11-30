import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
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
    # face_file_path = os.path.join(face_dir , 'face.jpg')
    # card_file_path = os.path.join(face_dir , 'card.jpg')
    try:
        known_image = face_recognition.load_image_file('card/card.jpg')
        unknown_image = face_recognition.load_image_file('face/face.jpg')

        biden_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

        return face_recognition.compare_faces([biden_encoding], unknown_encoding)
    except:
        return ["Face not detected"]

# def compare():
#     try:
#         face_path = face_recognition.load_image_file('face/face.jpg')
#         card_path = face_recognition.load_image_file('card/card.jpg')

#         models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
#         model_name = "Facenet"

#         result = DeepFace.verify(img1_path=face_path, img2_path=card_path, model_name=model_name)

#         print(result)

#         return result
#     except:
#         return ["Face not detected"]