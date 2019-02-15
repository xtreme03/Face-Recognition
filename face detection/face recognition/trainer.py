import os
import cv2
import numpy as np
from PIL import Image
recognizer = cv2.face.LBPHFaceRecognizer_create()
path='dataset'
def imagewithid(path):
    imagepaths=[os.path.join(path,f) for f in os.listdir(path)]# from path it is listing all the directories
    faces=[]
    IDs=[]
    for imagepath in imagepaths:
        faceimg=Image.open(imagepath).convert('L');
        facenp=np.array(faceimg,'uint8')
        ID=int(os.path.split(imagepath)[-1].split('.')[1])
        faces.append(facenp)
        print ID
        IDs.append(ID)
        cv2.imshow("Training",facenp)
        cv2.waitKey(10)
    return np.array(IDs),faces
Ids,faces=imagewithid(path)
recognizer.train(faces,Ids)
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()
        
        
           
