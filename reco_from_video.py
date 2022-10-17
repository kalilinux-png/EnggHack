import cv2
import os

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# video_capture = cv2.VideoCapture(0)

# Call the trained model yml file to recognize faces
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C:/Users/shubh/Desktop/folder/face-recognition-opencv/trained.yml")

# Names corresponding to each id
names = ['rishika','shubh','shubh1','shushil','sudarshan','yumna']

for users in os.listdir("dataset"):
    names.append(users)

main_path = "dataset/shubham"
# img = cv2.imread("test/chris.jpeg")

        # _, img = video_capture.read()
        # for images in os.listdir(main_path):
        #     image_path = main_path+"//"+images
        #     print("image path ",image_path)
        # image_path  = ""
        #     img = cv2.imread(image_path)
def start_video(folder_path=None):

    # while True:
            if not folder_path:
                img_path = "C:/Users/shubh/Desktop/folder/EnggDayHackathon/images/yavnika/" # file path 
            else:
                img_path = folder_path
            # print("names",names)
            names = os.listdir("C:/Users/shubh/Desktop/folder/EnggDayHackathon/images/")

            for images in os.listdir(img_path):
                img = cap=cv2.VideoCapture(0).read()[1]
                # img = cv2.imread(img_path+"/"+images)
                print(images)



                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                faces = faceCascade.detectMultiScale(
                    gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)
                )
                

                # Try to predict the face and get the id
                # Then check if id == 1 or id == 2
                # Accordingly add the names
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    id, _ = recognizer.predict(gray_image[y : y + h, x : x + w])
                    print("id returned",id,"other data",_)
                    if id:
                        cv2.putText(
                            img,
                            names[id],
                            (x, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    else:
                        cv2.putText(
                            img,
                            "Unknown",
                            (x, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )

                cv2.imshow("Recognize", img)
                cv2.waitKey(1)

                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break


start_video()

# todo: update  names with unique id and test on images with and then on video
# 