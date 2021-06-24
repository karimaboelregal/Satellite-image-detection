import numpy as np
import cv2
import matplotlib.pyplot as plt
import os



def main():
    while(1):
        path, dirs, files = next(os.walk("dataset"))
        file_count = len(files)
        val = input("Please choose a satellite image(1,"+str(file_count)+"): ")
        currentImage = 'dataset/'+val+'.jpg'
        if (not val.isnumeric() or int(val) == 0 or int(val) > file_count):
            print("Thank you for using this program")
            break;
        image = cv2.pow((cv2.imread(currentImage)/255.0),1)
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        k = 6
        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        labels = labels.flatten()
        table = imagecopyreshape(image)
        colorandshow(image, table, labels)


def imagecopyreshape(image):
    masked_image = []
    for i in range(5):
        masked_image.insert(i, np.copy(image));
        masked_image[i] = masked_image[i].reshape((-1, 3))
    return masked_image

def colorandshow(image, masked_image, labels):
    f, axarr = plt.subplots(1, 6)
    axarr[0].imshow(image)
    for i in range(5):
        for j in range(6):
            if (i == j):
                masked_image[i][labels == j] = [1,1,1]
            else:
                masked_image[i][labels == j] = [0,0,0]
        masked_image[i] = masked_image[i].reshape(image.shape)
        axarr[i+1].imshow(masked_image[i])
    plt.show()







if __name__ == "__main__":
    main()