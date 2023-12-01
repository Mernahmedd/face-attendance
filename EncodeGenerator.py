import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': "https://faceattendancerealtime-b99ec-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendancerealtime-b99ec.appspot.com"
})

folderPath = 'Images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))  # import images
    studentIds.append(os.path.splitext(path)[0]) # split it to take the id ['id','.extention']

    #upload images
    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)


print(len(imgList))

def findEncodings(imagesList): # encode every image of images
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert color from bgr (open cv) to rgb (face reognition)
        encode = face_recognition.face_encodings(img)[0] # encode it 
        encodeList.append(encode)

    return encodeList

print("Encoding Started ...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")

file = open("EncodeFile.p", 'wb') # wb write in pickle file in binary code
pickle.dump(encodeListKnownWithIds, file) # dump the data in the file
file.close()
print("File Saved")

