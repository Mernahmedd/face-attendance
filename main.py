import cv2
import cvzone
import os
import pickle
import face_recognition
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime



cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': "https://faceattendancerealtime-b99ec-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendancerealtime-b99ec.appspot.com"
})

bucket = storage.bucket()

cap = cv2.VideoCapture(0)  # open the video in same computer and if it is 1 will open it in another computer connected to
cap.set(3, 640)  # the highest of the video parameter
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.png') # import background image

folderModePath = 'Resources/Modes'  # take the bath of the photoes
modePathList = os.listdir(folderModePath) # get the list of all files and directories in the folder
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path))) # select the photoes and put it in the array

print(os.listdir(folderModePath))

# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb') # rb for read the file
encodeListKnownWithIds = pickle.load(file) # load the ids from the file 
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds # store it
# print(studentIds)
print("Encode File Loaded")
modeType = 0
counter = 0
id = -1
imgStudent = []


while True:
    success, img = cap.read()  # to read what camera is see

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) # make the image small so we can limit the time to encode it
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) # to make it match the face recogniton

    faceCurFrame = face_recognition.face_locations(imgS) # to get the location of the face 
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame) # to encode the small image with its location


    imgBackground[162:162 + 480, 55:55 + 640] = img  # put the camera in this dimentional in background image
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]  # to add the mode in another side of image

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame): # use zip because we need use them at the same time
        # number of values is equal number of stored images 
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) # it gives true or false as it match or not
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace) # it gives numbers and the lowest is the matched
        # print("matches", matches)
        # print("faceDis", faceDis)

        matchIndex = np.argmin(faceDis)
        # print("Match Index", matchIndex) # the index of the image

        if matches[matchIndex]:
            # print("Known Face Detected")
            # print(studentIds[matchIndex])
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
            id = studentIds[matchIndex]
            if counter == 0:
                cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                cv2.imshow("Face Attendance", imgBackground)
                cv2.waitKey(1)
                counter = 1
                modeType = 1

    if counter != 0:

        if counter == 1:
            # Get the Data
            studentInfo = db.reference(f'Students/{id}').get()
            # print(studentInfo)
            # Get the Image from the storage
            blob = bucket.get_blob(f'Images/{id}.png')
            array = np.frombuffer(blob.download_as_string(), np.uint8)
            imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
            # Update data of attendance
            datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                                   "%Y-%m-%d %H:%M:%S")
            secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
            # print(secondsElapsed)
            if secondsElapsed > 50:
                ref = db.reference(f'Students/{id}')
                studentInfo['total_attendance'] += 1
                ref.child('total_attendance').set(studentInfo['total_attendance'])
                ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            else:
                modeType = 3
                counter = 0
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        if modeType !=3:
            if 10 < counter < 40:
                modeType = 2

            if counter <=10:
                cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(imgBackground, str(id), (1006, 493),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                offset = (414 - w) // 2
                cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
                imgBackground[175:175 + 216, 909:909 + 216] = imgStudent

            counter +=1
            if counter >= 40:
                counter = 0
                modeType = 0
                studentInfo = []
                imgStudent = []
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    else:
        modeType = 0
        counter = 0

    cv2.imshow("Face Attendance", imgBackground) # activate the camera
    cv2.waitKey(1) # waite for it 