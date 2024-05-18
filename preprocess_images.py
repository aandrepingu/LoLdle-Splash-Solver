import cv2
import os
def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Resize the image to the target size
    image = cv2.resize(image, (300,177))
    
    # Normalize pixel values to range [0, 1]
    #image = image.astype(np.float32) / 255.0
    
    cv2.imwrite(filename=image_path, img=image)


if __name__ == "__main__":
    for image in os.listdir("../splash"):
        preprocess_image("../splash/"+image)
        print(image)