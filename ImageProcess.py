from PIL import Image
import cv2
import numpy as np
import os
from Hog_Descriptor import Hog_descriptor

class ImageProcess():
    def __init__(self,denomination):
        self.__denomination = denomination

    def features_extraction(self,imgPath):
        width = 0
        height = 245
        if(self.__denomination=="500"):
            width = 555
        elif(self.__denomination=="1000"):
            width = 589
        elif(self.__denomination=="5000"):
            print("5000")
            width = 616
        
        image = cv2.imread(imgPath)
        #cv2.imshow("10",img)
        #cv2.waitKey(0)  
        res_img = self.__resize_image(image,width,height)
        #cv2.imshow("Resized",res_img)
        #cv2.waitKey(0)

        features = self.__extract(res_img)
        



        return features

    def __resize_image(self,image,width,height):
        
        # dsize
        dsize = (width, height)

        # resize image
        res_img = cv2.resize(image, dsize)

        
        cv2.imwrite('resize_img.png',res_img) 

        return res_img

    def __extract(self,res_img):
        features = []
        features.append(self.__extract_watermark(res_img))
        features.append(self.__extract_micro_lettering(res_img))
        features.append(self.__extract_latent_image(res_img))

        cv2.imwrite("watermark.png",features[0])
        #cv2.imshow("watermark",features[0])
        #cv2.waitKey(0)
        cv2.imwrite("micro_lettering.png",features[1])
        #cv2.imshow("micro_lettering",features[1])
        #cv2.waitKey(0)
        cv2.imwrite("latent_image.png",features[2])
        #cv2.imshow("latent_image",features[2])
        #cv2.waitKey(0)


        return features
    
    def __extract_watermark(self,img):
        cropped=''
        if(self.__denomination=='500'):
            cropped = img[57:206, 18:121]
            #cv2.imshow("Watermark", cropped)
            #cv2.waitKey(0)
        elif(self.__denomination=='1000'):
            cropped = img[ 72:205,20:127]
            #cv2.imshow("Watermark", cropped)
            #cv2.waitKey(0)
        elif(self.__denomination=='5000'):
            print("5000")
            cropped = img[60:216, 10:122]
            #cv2.imshow("Watermark", cropped)
            #cv2.waitKey(0)
        
        img_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) 
        sharp_img = self.__unsharp_mask(img_gray)
        #cv2.imshow("Sharp", sharp_img)
        #cv2.waitKey(0)

        canny_img = self.__canny_detector(sharp_img)
        #cv2.imshow("Canny", canny_img)
        #cv2.waitKey(0)
    
        hog = Hog_descriptor(canny_img, cell_size=7, bin_size=7)
        hog_img=hog.extract()
        #cv2.imshow("Hog", hog_img)
        #cv2.waitKey(0)

        return hog_img
    
    def __extract_micro_lettering(self,img):
        cropped=''
        if(self.__denomination=='500'):
            cropped = img[134:229, 132:273]
            #cv2.imshow("Micro", cropped)
            #cv2.waitKey(0)
        elif(self.__denomination=='1000'):
            cropped = img[ 150:231,142:283]
            #cv2.imshow("Micro", cropped)
            #cv2.waitKey(0)
        elif(self.__denomination=='5000'):
            cropped = img[138:239, 145:299]
            #cv2.imshow("Micro", cropped)
            #cv2.waitKey(0)


        img_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) 
        
        canny_img = self.__canny_detector(img_gray)
        #cv2.imshow("Canny", canny_img)
        #cv2.waitKey(0)

        
        hog = Hog_descriptor(canny_img, cell_size=6, bin_size=6)
        hog_img=hog.extract()
        #cv2.imshow("Hog", hog_img)
        #cv2.waitKey(0)

        return hog_img
    
    def __extract_latent_image(self,img):
        cropped=''
        if(self.__denomination=='500'):
            cropped = img[38:145, 410:480]
            #cv2.imshow("Latent", cropped)
            #cv2.waitKey(0)
        elif(self.__denomination=='1000'):
            cropped = img[ 42:151,422:497]
            #cv2.imshow("Latent", cropped)
            #cv2.waitKey(0)
        elif(self.__denomination=='5000'):
            cropped = img[41:147, 434:518]
            #cv2.imshow("Latent", cropped)
            #cv2.waitKey(0)

        img_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) 
                
        hog = Hog_descriptor(img_gray, cell_size=7, bin_size=6)
        hog_img=hog.extract()
        #cv2.imshow("Hog", hog_img)
        #cv2.waitKey(0)
    
        return hog_img

    def __canny_detector(self,img, weak_th = None, strong_th = None): 
        
        # Noise reduction step 
        img = cv2.GaussianBlur(img, (3, 3), 1.4) 
        
        # Calculating the gradients 
        gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3) 
        gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3) 
        
        # Conversion of Cartesian coordinates to polar  
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True) 
        
        # setting the minimum and maximum thresholds  
        # for double thresholding 
        mag_max = np.max(mag) 
        if not weak_th:weak_th = mag_max * 0.1
        if not strong_th:strong_th = mag_max * 0.5
        
        # getting the dimensions of the input image   
        height, width = img.shape 
        
        # Looping through every pixel of the grayscale  
        # image 
        for i_x in range(width): 
            for i_y in range(height): 
                
                grad_ang = ang[i_y, i_x] 
                grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang) 
                
                # selecting the neighbours of the target pixel 
                # according to the gradient direction 
                # In the x axis direction 
                if grad_ang<= 22.5: 
                    neighb_1_x, neighb_1_y = i_x-1, i_y 
                    neighb_2_x, neighb_2_y = i_x + 1, i_y 
                
                # top right (diagnol-1) direction 
                elif grad_ang>22.5 and grad_ang<=(22.5 + 45): 
                    neighb_1_x, neighb_1_y = i_x-1, i_y-1
                    neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
                
                # In y-axis direction 
                elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90): 
                    neighb_1_x, neighb_1_y = i_x, i_y-1
                    neighb_2_x, neighb_2_y = i_x, i_y + 1
                
                # top left (diagnol-2) direction 
                elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135): 
                    neighb_1_x, neighb_1_y = i_x-1, i_y + 1
                    neighb_2_x, neighb_2_y = i_x + 1, i_y-1
                
                # Now it restarts the cycle 
                elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180): 
                    neighb_1_x, neighb_1_y = i_x-1, i_y 
                    neighb_2_x, neighb_2_y = i_x + 1, i_y 
                
                # Non-maximum suppression step 
                if width>neighb_1_x>= 0 and height>neighb_1_y>= 0: 
                    if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]: 
                        mag[i_y, i_x]= 0
                        continue
    
                if width>neighb_2_x>= 0 and height>neighb_2_y>= 0: 
                    if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]: 
                        mag[i_y, i_x]= 0
    
        weak_ids = np.zeros_like(img) 
        strong_ids = np.zeros_like(img)               
        ids = np.zeros_like(img) 
        
        # double thresholding step 
        for i_x in range(width): 
            for i_y in range(height): 
                
                grad_mag = mag[i_y, i_x] 
                
                if grad_mag<weak_th: 
                    mag[i_y, i_x]= 0
                elif strong_th>grad_mag>= weak_th: 
                    ids[i_y, i_x]= 1
                else: 
                    ids[i_y, i_x]= 2
        
        
        # finally returning the magnitude of 
        # gradients of edges 
        return mag 


    def __unsharp_mask(self,image, kernel_size=(5,5), sigma=10.0, amount=10.0, threshold=1):
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened
