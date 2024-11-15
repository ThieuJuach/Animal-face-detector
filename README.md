Animal Faces Classification Project

Overview

This project uses a convolutional neural network (CNN) built on ResNet50, a pretrained model, to classify animal faces into three categories: cats, dogs, and wild animals. The project involves data preprocessing, model fine-tuning, evaluation, and visualization of results.

Features

Data Loading: Loads and preprocesses the Animal Faces dataset.
Data Augmentation: Applies transformations like rotation, zoom, and horizontal flips.
Model Architecture: Fine-tunes ResNet50 for the classification task.
Evaluation: Generates confusion matrices and classification reports.
Visualization: Plots training and validation accuracy and loss.
Requirements
Python 3.x
TensorFlow
NumPy
scikit-learn
Matplotlib
Steps

1. Import Required Libraries
   
python
Copy code
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

2. Load and Prepare Dataset
   
The dataset is automatically downloaded and extracted. Training and validation directories are prepared.

python
Copy code
_URL = 'https://datasetdownloadslinks.s3.us-east-1.amazonaws.com/Download%20Datasets/Animal%20Faces.zip'
path_to_zip = tf.keras.utils.get_file('Animal Faces.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'afhq')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'val')

3. Data Augmentation
   
Apply transformations to enhance model robustness.

python
Copy code
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1.0/255)

4. Build and Compile the Model
   
Use ResNet50 as the base model.
Add a global average pooling layer, dense layers, and an output layer.
python
Copy code
Base_Model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in Base_Model.layers:
    layer.trainable = False

X = Base_Model.output
X = GlobalAveragePooling2D()(X)
X = Dense(1024, activation='relu')(X)
predictions = Dense(3, activation='softmax')(X)
Final_Model = Model(inputs=Base_Model.input, outputs=predictions)
Final_Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy', metrics=['accuracy'])
                    
5. Train the Model
   
Train the model using the fit method and fine-tune for better accuracy.

python
Copy code
History = Final_Model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir="./logs")]
)

6. Evaluate and Visualize
   
Evaluate the model and plot accuracy and loss graphs.

python
Copy code
val_generator.reset()
Preds = Final_Model.predict(val_generator)
Y_Pred = np.argmax(Preds, axis=1)
Y_True = val_generator.classes

print(classification_report(Y_True, Y_Pred, target_names=val_generator.class_indices.keys()))
Conf_Matrix = confusion_matrix(Y_True, Y_Pred)
print("Confusion Matrix:", Conf_Matrix)

plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

Results

Training and validation metrics plotted.
Classification report and confusion matrix generated for further analysis.

Notes

Ensure the dataset is correctly structured with train/ and val/ folders.
Use a GPU for faster training.
