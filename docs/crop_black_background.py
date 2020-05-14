import cv2
import numpy as np


def crop_image(img_name='selfie.png'):
    """Crop unwanted pixel borders from image
    takes parameter - name of the image
    return None
    save image in the same directory where it was"""
    img = cv2.imread('../images/' + img_name)
    blur = cv2.GaussianBlur(img, (15, 15), 2)
    lower_green = np.array([50, 100, 0])
    upper_green = np.array([120, 255, 120])
    mask = cv2.inRange(blur, lower_green, upper_green)
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    img = cv2.imread('../images/' + img_name)
    blur = cv2.GaussianBlur(img, (15, 15), 2)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_green = np.array([37, 0, 0])
    upper_green = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    img = cv2.imread('../images/' + img_name)
    blur = cv2.GaussianBlur(img, (15, 15), 2)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_green = np.array([37, 0, 0])
    upper_green = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    masked_img = cv2.bitwise_and(img, img, mask=opened_mask)

    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + +cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = img[y:y + h, x:x + w]
    cv2.imwrite('../images/' + img_name, crop)

    print('CROPPED. see ' + img_name)
