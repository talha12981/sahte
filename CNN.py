import numpy as np
import tensorflow as tf

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
        image_count = len(list(data_dir.glob('*/*.jpg')))
        print(image_count)
  
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

        model.save("CNN_Model")

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(self.__epochs)

    def Denomation_Detector(self, imgPath):
        loaded_model = tf.keras.models.load_model('CNN_Model')

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


#if __name__=="__main__":
 #   cnn= CNNModel()
  #  print(cnn.Denomation_Detector('fake1_3_1.jpg'))
