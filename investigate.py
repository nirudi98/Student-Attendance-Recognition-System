from skimage.metrics import structural_similarity as compare_ssim
import sys
import imutils
import cv2
import numpy as np


# reading and resizing the image
def read_resize_image(file_path, folder_name, image):
    print(file_path + folder_name + "/" + image)
    sample = cv2.imread(file_path + folder_name + "/" + image)
    print("image old size :", sample.shape)

    # resizing the image
    img = cv2.resize(sample, (350, 100))
    print("image new size :", img.shape)
    return img


# displaying an image
def show_image(name, image):
    cv2.imshow(name, image)
    cv2.moveWindow(name, 500, 0)
    cv2.waitKey(0)


# signature recognition, comparison function
def sign_recognition(image_path, imageA_folder, imageB_folder, img_name):
    img_A = read_resize_image(image_path, imageA_folder, img_name)
    print(image_path+imageB_folder+img_name)
    img_B = read_resize_image(image_path, imageB_folder, img_name)

    res = np.hstack((img_A, img_B))
    show_image("resized images", res)

# converting two signatures to gray
    grayA = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
    res = np.hstack((grayA, grayB))
    show_image("Gray images", res)

# removing noise of the two signatures
    blurredA = cv2.GaussianBlur(grayA, (5, 5), 0)
    blurredB = cv2.GaussianBlur(grayB, (5, 5), 0)
    res = np.hstack((blurredA, blurredB))
    show_image("blurred images", res)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(blurredA, blurredB, full=True)
    diff = (diff * 255).astype("uint8")
    print("Structural Similarity Index (SSIM): {}".format(score))
    print("difference between the two images", diff)

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the bounding box on both images to show the difference between them
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img_A, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(img_B, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # show the output images
    cv2.imshow("Image A", img_A)
    cv2.imshow("Image B", img_B)
    cv2.imshow("Diff", diff)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)


if __name__ == '__main__':
    image_name = sys.argv[1]
    folder_one_name = sys.argv[2]
    folder_two_name = sys.argv[3]
    print(sys.argv[0])
    print(sys.argv[1])
    print(sys.argv[2])
    sign_image_path = 'CGVAssignment/signatures/'

    sign_recognition(sign_image_path, folder_one_name, folder_two_name, image_name)


#    image_name = '10009303.jpeg'

# to run code with similar images use the default signature folder paths given
# python investigate.py 10009303.jpeg Three Three/signRecognition

# Three/signRecognition folder contains a copy of 10009303.jpeg signature in signing sheet 3

# to run code with different images, use the command below
# python investigate.py 10009303.jpeg One Three
























































