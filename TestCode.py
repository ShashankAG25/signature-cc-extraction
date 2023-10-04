import cv2
import numpy as np

# image_input = cv2.imread("dataset/MC/m_chal (101).jpg") # in 1
image_input = cv2.imread("dataset/certificates/c2.jpg")  # in 2
# print("number of channels in original in : " + str(len(image_input.shape)))
cv2.namedWindow('original', cv2.WINDOW_NORMAL)
cv2.imshow("original", image_input)
cv2.waitKey(0)

# creating threshold mask
grey_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
ret, bin_mask = cv2.threshold(grey_image, 60, 255, 0)  # for certificates
# ret, bin_mask = cv2.threshold(grey_image, 10, 255, 0) #for macksheets
# bin_mask = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2RGB)
# bin_mask = cv2.bitwise_not(bin_mask)
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.imshow("mask", bin_mask)
cv2.waitKey(0)

# subtract mask from grey_image
bit_op = cv2.subtract(bin_mask, grey_image)
process_img = cv2.bitwise_not(bit_op)
process_img = cv2.cvtColor(process_img, cv2.COLOR_GRAY2RGB)
cv2.namedWindow('PreProcessing', cv2.WINDOW_NORMAL)
cv2.imshow("PreProcessing", process_img)
cv2.waitKey(0)

# sub process_img from original input image(testing)
image_diff = cv2.subtract(process_img, image_input)
image_diff_final = cv2.bitwise_not(image_diff)
cv2.namedWindow('final', cv2.WINDOW_NORMAL)
cv2.imshow("final", image_diff_final)
cv2.waitKey(0)


def detect_signatures(image):
    # Apply edge detection to find potential signature regions
    edges = cv2.Canny(image, threshold1=30, threshold2=100)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store detected signature regions
    signature_regions = []

    # Set a threshold for signature area (adjust this as needed)
    min_signature_area = 1000

    # Iterate through the detected contours
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # If the area is above the threshold, consider it as a signature
        if area > min_signature_area:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Extract the signature region from the original image
            signature = image[y:y + h, x:x + w]

            # Append the signature region to the list
            signature_regions.append(signature)

    return signature_regions


signature_regions = detect_signatures(image_diff_final)

# Display the detected signatures (optional)
for i, signature in enumerate(signature_regions):
    cv2.namedWindow('results', cv2.WINDOW_NORMAL)
    cv2.imshow("results", signature)
    cv2.waitKey(0)
