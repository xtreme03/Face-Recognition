import cv2
import numpy as np
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);   #video capture object
rec= cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer\\trainningData.yml" )
id=0
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret,img=cam.read();
    ver_img = cv2.flip( img, 1 )
    gray=cv2.cvtColor(ver_img,cv2.COLOR_BGR2GRAY)#conversion to gray scale for the casscading to work
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,z,h) in faces:
        cv2.rectangle(ver_img,(x,y),(x+z,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+z])
        print id
        if(id and conf):
            cv2.putText(ver_img,str(id),(x,y+h),font,3,255);
        else:
            print ("UNKNOWN")
    
    
    cv2.imshow("Face",ver_img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows();
