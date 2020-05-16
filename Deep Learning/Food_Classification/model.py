import os
import shutil
from sklearn.model_selection import train_test_split
import time

start = time.time()
BASE_DIR_ORIG = r'C:\Users\ragha\Desktop\New folder\gym.AI\Deep Learning\Food_Classification\New folder\train'
val_path = r'C:\Users\ragha\Desktop\New folder\gym.AI\Deep Learning\Food_Classification\New folder\val'

labels = os.listdir(BASE_DIR_ORIG)

try:
    os.mkdir(val_path)
except:
    pass

for label in labels:
    try:
        os.mkdir(os.path.join(val_path, label))
    except:
        pass


    image = os.listdir(os.path.join(BASE_DIR_ORIG, label))
    train_x, val = train_test_split(image, test_size=0.25, random_state=25)

    print("STARTED COPYING.....")
    orig_img_path = os.path.join(BASE_DIR_ORIG, label)
    for i in val:
        orig_img = os.path.join(orig_img_path, i)
        new_img_path = os.path.join(val_path, label)
        shutil.move(orig_img, new_img_path)

    print("COPYING ENDED......." + str(label) + " Train")
    print("NO. OF IMAGES = " + str(len(os.listdir(os.path.join(val_path, label)))))
    print(" ")

end = time.time()

print("TIME TAKEN BY PROGRAM TO RUN : "+str(end-start))