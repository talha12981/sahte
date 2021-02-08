import cv2
import os
from ImageProcess import ImageProcess

def load_images_from_folder(folder):
    i=1
    water_file="Dataset/ORIGINAL/watermark/"
    micro_file="Dataset/ORIGINAL/micro/"
    latent_file="Dataset/ORIGINAL/latent/"
    print("Yes")
    ip = ImageProcess('1000')
    for filename in os.listdir(folder):
        filePath = os.path.join(folder,filename)
        img = cv2.imread(filePath)
        if img is not None:
            feat = ip.features_extraction(filePath)
            cv2.imwrite(water_file+str(i)+'.jpg',feat[0])
            cv2.imwrite(micro_file+str(i)+'.jpg',feat[1])
            cv2.imwrite(latent_file+str(i)+'.jpg',feat[2])
            print(i)
            i=i+1


if __name__ == "__main__":
    folder = "Dataset/DATA/original1000"
    load_images_from_folder(folder)