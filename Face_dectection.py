import cv2

class faceDetection():
    def face_detection(self,imgPath):
        # Load the cascade
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Read the input image
        img = cv2.imread(imgPath)
        img_90 = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        img_180 = cv2.rotate(img_90, cv2.cv2.ROTATE_90_CLOCKWISE)
        img_270 = cv2.rotate(img_180, cv2.cv2.ROTATE_90_CLOCKWISE)
        
        width = 1920
        height = 720
        dim = (width, height)
        
        # resize image
        resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        # Convert into grayscale
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces_img = face_cascade.detectMultiScale(gray_img, 1.1, 4)


        
        # resize image
        resized_img_90 = cv2.resize(img_90, dim, interpolation = cv2.INTER_AREA)
        # Convert into grayscale
        gray_img_90 = cv2.cvtColor(resized_img_90, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces_90 = face_cascade.detectMultiScale(gray_img_90, 1.1, 4)

        
        # resize image
        resized_img_180 = cv2.resize(img_180, dim, interpolation = cv2.INTER_AREA)
        # Convert into grayscale
        gray_img_180 = cv2.cvtColor(resized_img_180, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces_180 = face_cascade.detectMultiScale(gray_img_180, 1.1, 4)

        
        # resize image
        resized_img_270 = cv2.resize(img_270, dim, interpolation = cv2.INTER_AREA)
        # Convert into grayscale
        gray_img_270 = cv2.cvtColor(resized_img_270, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces_270 = face_cascade.detectMultiScale(gray_img_270, 1.1, 4)
        if(len(faces_img)==1):
            return img
        elif(len(faces_90)==1):
            img = cv2.rotate(img_90, cv2.cv2.ROTATE_90_CLOCKWISE)
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
            return img
        elif(len(faces_180)==1):
            img = cv2.rotate(img_180, cv2.cv2.ROTATE_90_CLOCKWISE)
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
            return img
        elif(len(faces_270)==1):
            img = cv2.rotate(img_270, cv2.cv2.ROTATE_90_CLOCKWISE)
            return img
        else:
            return "NO"
        