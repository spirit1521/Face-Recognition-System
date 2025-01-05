import cv2
import numpy as np
from PIL import Image
import os
import sys
# Path for face image database
path = 'database'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml");
# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for ip in imagePaths:
        PIL_img = Image.open(ip).convert('L') # convert it to grayscale here L stads for luminance that stores grayscale color values nd P stands for paletties which stores color values
        img_numpy = np.array(PIL_img,'uint8')#uint8 means unsigned(positive) 8bit int no i.e from 0 to 255 nd it converts pillow images to numpy array
        id =int(os.path.split(ip)[-1].split(".")[1])#main path is splitted to get id 
        faces = detector.detectMultiScale(img_numpy)#face detection is done of numpy images
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])#adds the face samples to empty tuple
            ids.append(id)#adds id to tuple
    return faceSamples,ids
print ("\n Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)#functionis called and returning values r stored in given variables
recognizer.train(faces, np.array(ids))#trains samples using obtained faces
# Save the model into trainer/trainer.yml
recognizer.save('trainer.yml') 
print("\n {0} faces trained. Exiting Program".format(len(np.unique(ids))))