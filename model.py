!pip install kaggle

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle competitions download -c dogs-vs-cats

!ls

from zipfile import ZipFile
dataset = '/content/dogs-vs-cats.zip'
with ZipFile(dataset, 'r') as zip:
  zip.extractall()
  print('The dataset is extracted')

from zipfile import ZipFile
dataset = '/content/train.zip'
with ZipFile(dataset, 'r') as zip:
  zip.extractall()
  print('The dataset is extracted')

import os

file_count = len(files)
print('Number of images: ', file_count)
file_names = os.listdir('/content/train/')
print(file_names)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow

img = mpimg.imread('/content/train/dog.3154.jpg')
imgplt = plt.imshow(img)
# plt.show()

img = mpimg.imread('/content/train/cat.11375.jpg')
imgplt = plt.imshow(img)
# plt.show()

# file_names = os.listdir('/content/train/')
for i in range(8):
  name = file_names[i]
  print(name[0:3])

# file_names = os.listdir('/content/train/')

dog_count = 0
cat_count = 0

for img_file in file_names:

  name = img_file[0:3]

  if name == 'dog':
    dog_count += 1

  else:
    cat_count += 1

print('Number of dog images =', dog_count)
print('Number of cat images =', cat_count)

# os.mkdir('/content/image_resized')

original_folder = '/content/train/'
resized_folder = '/content/image_resized/'

for i in range(2000):

  filename = os.listdir(original_folder)[i]
  img_path = original_folder+filename

  img = Image.open(img_path)
  img = img.resize((224, 224))
  img = img.convert('RGB')

  newImgPath = resized_folder+filename
  img.save(newImgPath)

img = mpimg.imread('/content/image_resized/dog.10132.jpg')
imgplt = plt.imshow(img)
# plt.show()

filenames = os.listdir('/content/image_resized/')
#بندي لابل للكلب بقيمه 1
#بندي لابل للقطه بقيمه 0

labels = []
for i in range(2000):
  file_name = filenames[i]
  label = file_name[0:3]
  if label == 'dog':
    labels.append(1)
  else:
    labels.append(0)

print(filenames[0:5])
print(len(filenames))

print(labels[0:5])
print(len(labels))

values, counts = np.unique(labels, return_counts=True)
print(values)
print(counts)

import cv2
import glob

image_directory = '/content/image_resized/'
image_extension = ['png', 'jpg']

files = []
[files.extend(glob.glob(image_directory + '*.' + e)) for e in image_extension]
dog_cat_images = np.asarray([cv2.imread(file) for file in files])

print(dog_cat_images)

type(dog_cat_images)

print(dog_cat_images.shape)

X = dog_cat_images
Y = np.asarray(labels)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

X_train_scaled = X_train/255
X_test_scaled = X_test/255

print(X_train_scaled)

import tensorflow as tf

pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False, 
    weights='imagenet' 
)

num_of_classes = 2
model = tf.keras.Sequential([
    pretrained_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_of_classes, activation='softmax') 
])

model.summary()

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics = ['acc']
)

model.fit(X_train_scaled, Y_train, epochs=4)

score, acc = model.evaluate(X_test_scaled, Y_test)
print('Test Loss =', score)
print('Test Accuracy =', acc)

input_image_path = input('Path of the image to be predicted: ')

input_image = cv2.imread(input_image_path)

cv2_imshow(input_image)

input_image_resize = cv2.resize(input_image, (224,224))

input_image_scaled = input_image_resize/255

image_reshaped = np.reshape(input_image_scaled, [1,224,224,3])

input_prediction = model.predict(image_reshaped)

print(input_prediction)

input_pred_label = np.argmax(input_prediction)

print(input_pred_label)

if input_pred_label == 0:
  print('The image represents a Cat')

else:
  print('The image represents a Dog')

import requests
image_url = input('Enter the URL of the image to be predicted: ')
response = requests.get(image_url, stream=True)
if response.status_code == 200:
    image = Image.open(response.raw)
    plt.imshow(image)
    image.show()
    image_np = np.array(image)
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    input_image_resize = cv2.resize(image_np, (224, 224))
    input_image_scaled = input_image_resize / 255.0
    image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])
    input_prediction = model.predict(image_reshaped)
    print(input_prediction)
    input_pred_label = np.argmax(input_prediction)
    if input_pred_label == 0:
        print('The image represents a Cat')
    else:
        print('The image represents a Dog')

from tensorflow import keras
from tensorflow import lite
converter=tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel=converter.convert()
open("animal.tflite","wb").write(tfmodel)
