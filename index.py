# ImportING the necessary packages
import cv2 as cv
import face_recognition
import matplotlib.pyplot as plt
 
 
# Load the known image 
known_image = face_recognition.load_image_file("WIN_20231026_11_50_16_Pro.jpg")
known_faces = face_recognition.face_encodings(face_image = known_image,
                                              num_jitters=50,
                                              model='large')[0]
 
# Lanching the live camera
cam = cv.VideoCapture(0)
#Checking camera
if not cam.isOpened():
    print("Camera not working")
    exit()
     
# when camera is opened
while True:
     
    # campturing the image frame-by-frame
    ret, frame = cam.read()
     
    # checking frame is reading or not
    if not ret:
        print("Can't receive the frame")
        break
 
    # Face detection in the frame
    face_locations = face_recognition.face_locations(frame)
 
    for face_location in face_locations:
        top, right, bottom, left = face_location
        # Drawing a rectangle with blue line borders of thickness of 2 px
        frame = cv.rectangle(frame,  (right,top), (left,bottom), color = (0,0, 255), thickness=2)
    # Checking the each faces location in each frame
    try:
        # Frame encoding
        Live_face_encoding = face_recognition.face_encodings(face_image = frame,
                                                              num_jitters=23,
                                                              model='large')[0]
 
        # Matching with the known faces
        results = face_recognition.compare_faces([known_faces], Live_face_encoding)
 
        if results:
            img = cv.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv.putText(img, 'HarshVerma', (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255,0,0), 2, cv2.LINE_AA)
            print('Harsh Verma Enter....')
            plt.imshow(img)
            plt.show()
            break
    except:
        img = cv.putText(frame, 'Not Harsh Verma', (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255,0,0), 2, cv2.LINE_AA)
        # Display the resulting frame
        cv.imshow('frame', img)
        # End the streaming
        if cv.waitKey(1) == ord('q'):
            break
     
 
# Releasing the capture
cam.release()
cv.destroyAllWindows()