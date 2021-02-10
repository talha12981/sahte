from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, datasets
import pickle
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.io import imread
import skimage
import cv2
from ImageProcess import ImageProcess

class SVM():

    def __init__(self, dir=''):
            if(dir != ''):
                self.__dir=dir
            else:
                pkl_watermark ="watermark_model.pkl"
                # Load from file
                with open(pkl_watermark, 'rb') as file:
                    self.__water_model = pickle.load(file)

                pkl_micro = "micro_model.pkl"
                # Load from file
                with open(pkl_micro, 'rb') as file:
                    self.__micro_model = pickle.load(file)

                pkl_latent = "latent_model.pkl"
                # Load from file
                with open(pkl_latent, 'rb') as file:
                    self.__latent_model = pickle.load(file)

    def __load_data(self):
        image_dir = Path(self.__dir)
        folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
        categories = [fo.name for fo in folders]
        print(categories)
        descr = "A image classification dataset"
        images = []
        flat_data = []
        target = []
        for i, direc in enumerate(folders):
            for file in direc.iterdir():
                img = skimage.io.imread(file)
                flat_data.append(img.flatten()) 
                images.append(img)
                target.append(i)
        flat_data = np.array(flat_data)
        target = np.array(target)
        images = np.array(images)

        return Bunch(data=flat_data,
                    target=target,
                    target_names=categories,
                    images=images,
                    DESCR=descr)

    def create_model(self,C,kernel,gamma):
        image_dataset = self.__load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            image_dataset.data, image_dataset.target, test_size=0.2,random_state=109)

        clf = svm.SVC(C=C,kernel=kernel,gamma=gamma)
        clf.fit(X_train,y_train)

        pkl_filename = "latent_model.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(clf, file)

        y_pred = clf.predict(X_test)
        print("Precision:",metrics.precision_score(y_test, y_pred))
        print("Classification report for - \n{}:\n{}\n".format(
        clf, metrics.classification_report(y_test, y_pred)))


    def predict(self,features):
        
            
        # Calculate the accuracy score and predict target values
        wat = self.__water_model.predict(features[0].flatten().reshape(1,-1))
        
        
        
            
        # Calculate the accuracy score and predict target values
        mic = self.__micro_model.predict(features[1].flatten().reshape(1,-1))
        
        
        
            
        # Calculate the accuracy score and predict target values
        lat = self.__latent_model.predict(features[2].flatten().reshape(1,-1))

        if(wat+mic+lat>1):
            return "Original Note"
        else:
            return "Fake Note"
