import cv2,numpy as np
import os
from PIL import Image

# file_path = "C:\\Users\\shubh\\Desktop\\folder\\face-recognition-opencv\\dataset\\shubham" # path that contains the file path 
file_path = "C:\\Users\\shubh\\Desktop\\folder\\EnggDayHackathon\\images\\"  # path that contains the file path it must be one level above
# unique_id = 3  #can be used to identify user 
# print(os.listdir(file_path))

sub_folders = [ ]
for unique_id,folder_name in enumerate(os.listdir(file_path)):
    sub_folders.append(file_path+folder_name) # index == unique id for training in opencv2

print(sub_folders)

image_numpy_array = [ ] # conexit()tain array of images
unique_id_for_each_image = [ ] # it will be same for each image in particular folder

for unique_id , images_path in enumerate(sub_folders):
    for images in os.listdir(images_path):
        # full_image_path = "+images_path # create full image path for images
        full_image_path = images_path
        full_image_path = full_image_path+"\\"+images
        black_white_image = Image.open(full_image_path).convert("L") # converting image to black and white 
        image_array = np.array(black_white_image,"uint8")
        # cv2.imshow("Testing",image_array)
        image_numpy_array.append(image_array)
        unique_id_for_each_image.append(unique_id)
        # cv2.waitKey(1)

unique_id_for_each_image = np.array(unique_id_for_each_image) # convert list to numpy array
trainer = cv2.face.LBPHFaceRecognizer_create()
trainer.train(image_numpy_array,unique_id_for_each_image)
trainer.write("trained.yml")

print("Training completed")


# file more than 100 mb train.yml