import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize

model = load_model('save_model\RESNET-50-FOOD-CLASSIFICATION.h5')

number_to_class = []
for i in  range(101):
    number_to_class.append(i)

def predict():
    path = r'data/train/beef_carpaccio/6765.jpg'
    my_image = plt.imread(path)
    my_image_resized = resize(my_image, (32, 32, 3))
    probabilities = model.predict(np.array([my_image_resized, ]))
    index = np.argsort(probabilities[0, :])
    for i in range(5):
        print(number_to_class[index[i]])

predict()