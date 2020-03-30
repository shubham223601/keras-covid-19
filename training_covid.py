from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import cv2
import os



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='dataset', help ="path to Input Dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="covid_19_model.h5", help="path to output the model")
args = vars(ap.parse_args())

initial_lr = 0.001
epochs = 20
batch_size = 8

imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []


print("Loading Images")
for imagePath in imagePaths:
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	data.append(image)
	labels.append(label)

data = np.array(data)/255.0
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest", zoom_range=0.15, width_shift_range=0.2,
                                height_shift_range=0.2, shear_range=0.15, horizontal_flip=True)

model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3))
)

base_line = model.output
base_line = AveragePooling2D(pool_size=(4,4))(base_line)
base_line = Flatten(name="flatten")(base_line)
base_line = Dense(128, activation="relu")(base_line)
base_line = Dense(64, activation="relu")(base_line)
base_line = Dropout(0.5)(base_line)
base_line = Dense(2, activation='softmax')(base_line)

final_model = Model(inputs = model.input, outputs = base_line)

for layer in model.layers:
    layer.trainable=False

opt = Adam(lr = initial_lr, decay = initial_lr/epochs)
final_model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

print("training the model")
model_fit = final_model.fit_generator(
    trainAug.flow(
        train_x, train_y, batch_size = batch_size
    ),
    steps_per_epoch = len(train_x)//batch_size,
    validation_data = (test_x, test_y),
    validation_steps = len(test_x)//batch_size,
    epochs = epochs
)

pred_idx = final_model.predict(test_x, batch_size=batch_size)
predIdxs = np.argmax(pred_idx, axis=1)
print(classification_report(test_y.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

cm = confusion_matrix(test_y.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])


print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), model_fit.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), model_fit.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), model_fit.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), model_fit.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# serialize the model to disk
print("[INFO] saving COVID-19 detector model...")
model.save(args["model"])