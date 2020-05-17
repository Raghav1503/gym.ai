import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize
model = load_model(r'C:\Users\ragha\Desktop\New folder\gym.ai\Deep Learning\Food_Classification\save_model\RESNET-50-FOOD-CLASSIFICATION.h5')
number_to_class = []
for i in  range(101):
    number_to_class.append(i)

def hello():
    print("hello")

def predict(path):
    my_image = plt.imread(path)
    my_image_resized = resize(my_image, (32, 32, 3))
    probabilities = model.predict(np.array([my_image_resized, ]))
    index = np.argsort(probabilities[0, :])
    for i in range(5):
        print(number_to_class[index[i]])
