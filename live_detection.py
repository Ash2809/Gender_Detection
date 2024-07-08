import cv2
import os
import numpy as np
from keras.models import load_model

model_path = r"Gender_classifier.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model Loaded Successfully.")
else:
    print(f"No model was found at{model_path}")



cap = cv2.VideoCapture(0)
color_dict = {1:(255,0,0), 0:(203,192,255)}

while(True):
    r,frame = cap.read()
    if r == True:
        frame = cv2.resize(frame,(600,600))
        frame = cv2.flip(frame,1)

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        clsfr = cv2.CascadeClassifier(r"C:\Users\aashutosh kumar\Downloads\haarcascade_frontalface_default.xml")
        faces = clsfr.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

        for (x,y,w,h) in faces:
            face_rgn = gray[y:y+h, x:x+w]
            face_rgn_rzd = cv2.resize(face_rgn,(150,150))

            face_rgn_rzd = cv2.cvtColor(face_rgn_rzd , cv2.COLOR_GRAY2BGR)
            normalized_face = face_rgn_rzd/255.0

            reshaped = np.reshape(normalized_face, (1, 150, 150, 3))
            
            result = model.predict(reshaped)

            confidence = result[0][0]
            if confidence <= 0.5:
                label = 0
                gender = "FEMALE"
            else:
                label = 1
                gender = "MALE"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_dict[label], 2)
            
            cv2.rectangle(frame, (x, y-40), (x+w, y), color_dict[label], -1)
            cv2.putText(frame,gender,(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("LIVE",frame)

        if cv2.waitKey(25) & 0xff == ord("p"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
