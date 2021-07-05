import cv2
import os

#initialize classifier
faceCascade = cv2.CascadeClassifier("C:\\Users\\pauls\\PycharmProjects\\NLP\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("training.yml")

names = []
paths = []

for users in os.listdir("face_images"):
    names.append(users)


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray_img,
                                         scaleFactor=1.2,
                                         minNeighbors=5,
                                         minSize=(50,50))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
        print(id)
        id, _= recognizer.predict(gray_img[y:y+h, x:x+w])
        if id:
            cv2.putText(frame, names[id-1], (x,y-4), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (0,255,0),
                        1, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Unknown', (x, y - 4), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (0, 255, 0),
                        1, cv2.LINE_AA)
    cv2.imshow("Recognize", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()