import cv2
import numpy as np
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);   #video capture object
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#conversion to gray scale for the casscading to work
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,z,h) in faces:
        cv2.rectangle(img,(x,y),(x+z,y+h),(0,0,255),2)
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows();
    
