# Task6
Task 6 - Face Recognition
Step 1 - Create Training Data
1st person
In [1]:
import cv2
import numpy as np 

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return None 

    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face

cap = cv2.VideoCapture(0)
count = 0

while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = './faces/user/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)


        cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1)==13 or count==100:
        break

cap.release()
cv2.destroyAllWindows()
print("Samples collected")
<>:11: SyntaxWarning: "is" with a literal. Did you mean "=="?
<>:11: SyntaxWarning: "is" with a literal. Did you mean "=="?
<ipython-input-1-71088aa84e60>:11: SyntaxWarning: "is" with a literal. Did you mean "=="?
  if faces is ():
Samples collected
In [ ]:

Step2 - Train Model
In [2]:
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import time

data_path = './faces/user/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

harshal_model  = cv2.face_LBPHFaceRecognizer.create()

harshal_model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained succesfully")
Model trained succesfully
Step 3- Collecting Data and Model for 2nd Person
In [3]:
import cv2
import numpy as np 
from os import listdir
from os.path import isfile, join

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return None 

    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face

cap = cv2.VideoCapture(0)
count = 0

while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = './faces/user2/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)


        cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1)==13 or count==100:
        break

cap.release()
cv2.destroyAllWindows()
print("Samples collected")








data_path = './faces/user2/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

person_model  = cv2.face_LBPHFaceRecognizer.create()

person_model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained succesfully")
<>:13: SyntaxWarning: "is" with a literal. Did you mean "=="?
<>:13: SyntaxWarning: "is" with a literal. Did you mean "=="?
<ipython-input-3-560f64de6721>:13: SyntaxWarning: "is" with a literal. Did you mean "=="?
  if faces is ():
Samples collected
Model trained succesfully
Step 4- Run the Trained Facial Model
In [4]:
import cv2
import numpy as np
import os
import pywhatkit
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from os.path import isfile, join

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def SendMail(ImgFile):
    with open(ImgFile, 'rb') as f:
        img_data = f.read()


    sender_email = "harshaluser123@gmail.com"
    receiver_email = "harshalkondhalkar245@gmail.com"
    password = "idoq12349"

    message = MIMEMultipart()
    message["Subject"] = "System Alert"
    message["From"] = sender_email
    message["To"] = receiver_email

    text = MIMEText("Hey Harshal, this is your image")
    message.attach(text)
    image = MIMEImage(img_data, name=os.path.basename(ImgFile))
    message.attach(image)


    s = smtplib.SMTP_SSL("smtp.gmail.com", 465) 
    s.login(sender_email, password)
    s.sendmail(sender_email, receiver_email, message.as_string())
    s.quit()

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []


    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)


        results = harshal_model.predict(face)



        if results[1] < 500:
                confidence = int( 100 * (1 - (results[1])/400) )
                display_string = str(confidence) + '% Confident it is User'

        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)

        if confidence > 80:
            cv2.putText(image, "Hey Harshal", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            cv2.imwrite('cap_img.jpg', image)

            SendMail('cap_img.jpg')
            print("mail sent")

            pywhatkit.sendwhatmsg('+918080451271', "Hey Pranav!!!", 14,10)
            print("Message sent to Pranav")
            break


        else:
            results = person_model.predict(face)

            if results[1] < 500:
                confidence = int( 100 * (1 - (results[1])/400) )
                display_string = str(confidence) + '% Confident it is User'

            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)

            if confidence > 80:
                cv2.putText(image, "Hey Man", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Recognition', image )



                os.system("aws ec2 run-instances  --image-id ami-0ad704c126371a549 --instance-type t2.micro --count 1 --subnet-id subnet-7bd28a37 --security-group-ids sg-0a195485ec0f45274 --key-name awsclass2021key")
                print("Instance created...")
                os.system("aws ec2 create-volume --avaibility-zone ap-south-1b --volume-type gp2 --size 5 ")
                print("EBS volume of 5GB created...")
                break

        pass
    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )



    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
<>:41: SyntaxWarning: "is" with a literal. Did you mean "=="?
<>:41: SyntaxWarning: "is" with a literal. Did you mean "=="?
<ipython-input-4-975d01304eb5>:41: SyntaxWarning: "is" with a literal. Did you mean "=="?
  if faces is ():
Instance created...
EBS volume of 5GB created...
