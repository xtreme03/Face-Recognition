import cv2
import numpy as np
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cam=cv2.VideoCapture(0);   #video capture object
while(True):
    ret,mg=cam.read();
    img=cv2.flip(mg,1);
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#conversion to gray scale for the casscading to work
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,z,h) in faces:
        cv2.rectangle(img,(x,y),(x+z,y+h),(0,0,255),2)
        roi_color = img[y:y+z, x:x+h]
        roi_gray = gray[y:y+h, x:x+h]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows();
    
