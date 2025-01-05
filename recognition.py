import cv2
import numpy as np
import os
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml");
font = cv2.FONT_HERSHEY_SIMPLEX

path = 'unknowns'
sender_mail="infernoredx7@gmail.com"
reciever="parthkharche1521@gmail.com"
password="pwld lhqx hzac ktrw"
subject="Be Carefull(Face Recognition)"
body="Someone is trying to get in the house be carefull!!!!!"
msg = MIMEMultipart()
msg['From'] = sender_mail
msg['To'] = reciever
msg['Subject'] = subject
msg.attach(MIMEText(body,'plain'))
id = 0#for indexing
names = ['None', 'Gaurang', 'Parth','Shweta','Sumit']  
cam = cv2.VideoCapture(0)


cam.set(3, 640)
cam.set(4, 480) 
while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence<= 70):
            
                id = names[id]#uses earlier given id as a index to get the names from the list
                confidence = "  {0}%".format(round(confidence))
                print("Opening Lock")
                #time.sleep(1)
        else:
            id= "unknown"
            confidence=" {0}%".format(round(confidence))
            cv2.imwrite("unknowns/" + str(id) +".jpg",gray)
            filename= 'unknowns/unknown.jpg'
            attachment  =open(filename,'rb')
            part = MIMEBase('application','octet-stream')#used to open the given file to open in application
            part.set_payload((attachment).read())#sets the capacity to send the attachments
            encoders.encode_base64(part)#endcodes the file in base 64 to send the file
            part.add_header('Content-Disposition',"attachment; filename= "+filename)
            msg.attach(part)
            text = msg.as_string()
            server = smtplib.SMTP('smtp.gmail.com',587)
            server.starttls()
            server.login(sender_mail,password)
            server.sendmail(sender_mail,reciever,text)
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h+25), font, 1, (255,255,0), 1)  
    cv2.imshow('camera',img)
    key = cv2.waitKey(100)
    if key == 27:
        print("Esc pressed")
        break
print("Exiting Program")
cam.release()
cv2.destroyAllWindows()