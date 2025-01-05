import cv2
import os
cam = cv2.VideoCapture(0)
cam.set(3, 640) 
cam.set(4, 480) 
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# For each person, enter one numeric face id
face_id = input('Enter User ID: ')
print("Look at camera and wait ")
if not os.path.exists("database"):
    os.makedirs("database")
if not os.path.exists("unknowns"):
    os.makedirs("unknowns")
count = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        cv2.imwrite("database/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    key = cv2.waitKey(100) 
    if key == 27:
        print("Esc pressed")
        break
    elif count >= 20: 
         break
# Do a bit of cleanup
print("Exiting")
cam.release()
cv2.destroyAllWindows()