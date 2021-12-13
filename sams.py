from xml.dom import minidom

import cv2
import sys
import numpy as np
from PIL import Image
import pandas as pd
from imutils import contours
import imutils
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
import os


# reading and resizing the image
def read_resize_image(image):
    print(image)
    sample = cv2.imread(image)
    print("image old size :", sample.shape)

    # resizing the image
    img = cv2.resize(sample, (610, 800))
    print("image new size :", img.shape)
    cv2.imwrite('CGVAssignment/zoomed.jpeg', img)
    return img


# displaying an image
def show_image(name, image):
    cv2.imshow(name, image)
    cv2.moveWindow(name, 500, 0)
    cv2.waitKey(0)


# gaussian kernel
def gaussian_kernel(size, sigma=1):
    size = int(size)
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


# creating histogram for an image
def create_hist(img_hist):
    print(img_hist.flatten().size)
    hist, bins = np.histogram(img_hist.flatten(), 256, [0, 256])
    print("bins", bins)
    print("hist", hist)
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(img_hist.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()


# creating equalized histogram for an image
def equ_hist(equalized):
    hist, bins = np.histogram(equalized.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(equalized.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()


# function containing basic image processing to the input image
def image_processing(img_name):
    signing_sheet = read_resize_image(img_name)
    show_image("signing sheet", signing_sheet)

# converting image to grayscale
    gray_image = cv2.cvtColor(signing_sheet, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('CGVAssignment/grayed_image.jpeg', gray_image)
    grey_graph = plt.imshow(gray_image, cmap='gray')
    plt.show()

# remove noises by image filtering - applying gaussian filter
#    gau = gaussian_kernel(5)
#    gau = cv2.GaussianBlur(gray_image, (5, 5), 0)
    print(gray_image.shape)
    filter_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
#    res = np.hstack((filter_image1, filter_image2, filter_image3))
    show_image("filtered image", filter_image)

    # histogram - creating a histogram
    create_hist(gray_image)

    # equalizing the histogram
    equ = cv2.equalizeHist(gray_image)
    res = np.hstack((gray_image, equ))
    show_image("equalized image", res)
    cv2.imwrite('CGVAssignment/hist_equalized_image.jpg', equ)

    # equalized histogram
    equ_hist(equ)

    # basic thresholding applied to filtered image
    ret, thresh1 = cv2.threshold(filter_image, 127, 255, cv2.THRESH_BINARY)
    show_image("binary thresholding", thresh1)
    cv2.imwrite('CGVAssignment/binarized_image.jpg', equ)

    # comparing different types of thresholding
    thresh2 = cv2.adaptiveThreshold(filter_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh3 = cv2.adaptiveThreshold(filter_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding',
              'Adaptive Gaussian Thresholding']
    images = [filter_image, thresh1, thresh2, thresh3]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()

    # edge detection using canny
    edges = cv2.Canny(filter_image, 50, 200, 255)
    res = np.hstack((gray_image, edges))
    show_image("edges detected", res)
    return edges, gray_image, signing_sheet


# creating a csv to store student attendance
def create_csv():
    student_df = pd.DataFrame(columns=["Index", "Student Name", "Date", "Attendance"])
    return student_df


# signature detection function
def signature_detection(img_name, edges, sign_path, gray_im, original_img, info_students):
    cropped_signature = None
    screenCnt = None
    count = 0

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, 2, 4, 8)
    contour = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    contour = sorted(contour, key=cv2.contourArea, reverse=True)[:5]

    # looping over the contours to find a rectangle
    for c in contour:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print(len(approx))

        # if the approximated contour has four points, we can come to the assumption, we found the table
        if len(approx) == 4:
            screenCnt = approx
            break

    warped = four_point_transform(gray_im, screenCnt.reshape(4, 2))
    output = four_point_transform(original_img, screenCnt.reshape(4, 2))
    cv2.imwrite('CGVAssignment/' + img_name + '_output_table.jpg', output)
    print(output.shape)

# creating a csv file to keep the attendance
    student_details = create_csv()
    print(student_details)

# accessing the info.xml to get student name and index number
    Date = '27-08-2021'
    info_doc = minidom.parse(info_students)
    index = info_doc.getElementsByTagName("indexNumber")
    student_name = info_doc.getElementsByTagName("studentName")

# looping through the output table to crop the signature cells
    for i in range(15, 120, 17):
        cropped_signature = output[i:i + 19, 340:340 + 92]
        gray = cv2.cvtColor(cropped_signature, cv2.COLOR_BGR2GRAY)

        blur_signature = cv2.GaussianBlur(gray, (5, 5), 0)

        inv_bin_signature = cv2.threshold(blur_signature, 195, 255, cv2.THRESH_BINARY)[1]
        signature_image = imutils.resize(inv_bin_signature, width=612, height=800)

        show_image("cropped signature", signature_image)
        cv2.imwrite('Code_Images/signature.jpeg', inv_bin_signature)
        signature_TEST = Image.open("Code_Images/signature.jpeg")

        # count the white pixels in the signature image by using countNonZero() function
        data = np.array(signature_TEST.getdata(), np.uint8)
        nzCount = cv2.countNonZero(data)

        print("white pixel in image count :", nzCount)

# increase the counter by one since the student list index starts from 1
        count += 1

# if number of white pixels less than thousand, we can assume there is a signature present
        if 10 < nzCount < 1000:
            crop_index = output[i:i + 19, 1:1 + 50]
            show_image("index of cropped image :", crop_index)
            print("count: " + str(count))

            index_number = index[count - 1].firstChild.data
            stu_name = student_name[count - 1].firstChild.data
            student_details.loc[len(student_details.index)] = [index_number, stu_name, Date, "Present"]

# store the image of sign with the name on created path
            store_signature_path = "Code_Images/" + "signature_" + index_number + ".jpeg"

# show the cropped sign on screen
            cv2.imwrite(store_signature_path, cropped_signature)
            cv2.imwrite(sign_path + "/" + index_number + ".jpeg", cropped_signature)

        else:
            index_number = index[count - 1].firstChild.data
            name = student_name[count - 1].firstChild.data
            student_details.loc[len(student_details.index)] = [index_number, name, Date, "Absent"]

        if count >= 6:
            break

    return student_details


# creating CSV file to store student attendance - use date to name the csv files
def write_csv(attendance_sheet):
    filename = 'Attendance_Sheet' + attendance_sheet['Date'].iloc[0]
    attendance_count = attendance_sheet.value_counts('Attendance')
    attendance_sheet.to_csv(filename + '.csv', index=False, header=True)
    return 'saved'


# main function containing basic file paths and other function callings
if __name__ == '__main__':
    image_name = sys.argv[1]  # grabbing the image file from arguments passed
    info = sys.argv[2]  # grabbing the xml file from arguments passed

    sign_image_path = 'CGVAssignment/signatures/Three'

    edge_detected_image, gray_img, original_image = image_processing(image_name)

    students_attendance_sheet = signature_detection(image_name, edge_detected_image, sign_image_path, gray_img, original_image, info)

    print(students_attendance_sheet)

    status = write_csv(students_attendance_sheet)

    if status == 'saved':
        print("attendance sheet successfully saved")
    else:
        print("attendance sheet saving unsuccessful")


# python sams.py 3.jpeg info.xml

# when running the code for other signing sheets change the code in line 234 as below
# sign_image_path = 'CGVAssignment/signatures/Four'
# sign_image_path = 'CGVAssignment/signatures/One'

