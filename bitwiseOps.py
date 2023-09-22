import cv2
import numpy as np

image_input = cv2.imread("dataset/MC/m_chal (43).jpg")


def process_image(image):
    result = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 38, 0])
    upper = np.array([145, 255, 255])
    mask = cv2.inRange(image, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    img_erosion = cv2.erode(mask, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    opening = cv2.morphologyEx(img_dilation, cv2.MORPH_OPEN, kernel, iterations=1)

    result[opening== 0] = (255, 255, 255)
    return result


mask_image= process_image(image_input)
mask_image =cv2.bitwise_not(mask_image)
# bit_op = cv2.add(image_input,mask_image)
bit_op = cv2.bitwise_xor(image_input,mask_image)
cv2.namedWindow('PreProcessing', cv2.WINDOW_NORMAL)
cv2.imshow("PreProcessing",bit_op )
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
            signature = image[y:y+h, x:x+w]

            # Append the signature region to the list
            signature_regions.append(signature)

    return signature_regions

#
signature_regions = detect_signatures(bit_op)

# Display the detected signatures (optional)
for i, signature in enumerate(signature_regions):
    cv2.imshow("results",signature)
    cv2.waitKey(0)
