import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import pickle

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

class CNNModel():
    def __init__(self,TRAIN_DIR='',IMG_SIZE=180,LR=0.1,batch_size=10,validation_split=0.1,epochs=10):
        self.__TRAIN_DIR=TRAIN_DIR
        self.__IMG_SIZE=IMG_SIZE
        self.__LR=LR
        self.__batch_size=batch_size
        self.__validation_split=validation_split
        self.__epochs=epochs


    def __label_img(self):
        data_dir = pathlib.Path(self.__TRAIN_DIR)
#        image_count = len(list(data_dir.glob('*/*.jpg')))
 #       print(image_count)
  
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(self.__IMG_SIZE,self.__IMG_SIZE),
        batch_size=self.__batch_size)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(self.__IMG_SIZE,self.__IMG_SIZE),
        batch_size=self.__batch_size)
        
        return train_ds, val_ds

    def create_model(self,activation,optimizer):
        train_ds, val_ds = self.__label_img()
        class_names = train_ds.class_names
   #     print(class_names)


  #      plt.figure(figsize=(10, 10))
   ##        for i in range(9):
     #           ax = plt.subplot(3, 3, i + 1)
      #          plt.imshow(images[i].numpy().astype("uint8"))
       #         plt.title(class_names[labels[i]])
        #        plt.axis("off")

        #for image_batch, labels_batch in train_ds:
         #   print(image_batch.shape)
          #  print(labels_batch.shape)
           # break

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))

        num_classes = 7

        model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.__IMG_SIZE, self.__IMG_SIZE, 3)),
        layers.Conv2D(64, 3, padding='same', activation=activation),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation=activation),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation=activation),
        layers.Dense(num_classes)
        ])

        model.compile(optimizer=optimizer,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        model.summary()

        
        history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=self.__epochs
        )

        # serialize model to JSON
        model_json = model.to_json()
        with open("Denomination_Model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")


        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

#        plt.figure(figsize=(8, 8))
 #       plt.subplot(1, 2, 1)
  #      plt.plot(epochs_range, acc, label='Training Accuracy')
   #     plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    #    plt.legend(loc='lower right')
     #   plt.title('Training and Validation Accuracy')

#        plt.subplot(1, 2, 2)
 #       plt.plot(epochs_range, loss, label='Training Loss')
  #      plt.plot(epochs_range, val_loss, label='Validation Loss')
   #     plt.legend(loc='upper right')
    #    plt.title('Training and Validation Loss')
     #   plt.show()


    def Denomation_Detector(self, imgPath):
                # load json and create model
        json_file = open('Denomination_Model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
#        print("Loaded model from disk")

        img = keras.preprocessing.image.load_img(
            imgPath, target_size=(self.__IMG_SIZE, self.__IMG_SIZE)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = loaded_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_names = ['10','100','1000','20','50','500','5000']
        
        #print(
        #    "This image most likely belongs to {} with a {:.2f} percent confidence."
        #    .format(class_names[np.argmax(score)], 100 * np.max(score))
        #)

        return class_names[np.argmax(score)]
