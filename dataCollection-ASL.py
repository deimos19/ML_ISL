import math
import time
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

offset = 20
imgSize = 300
counter = 0
folder = "Data/A"

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
while True:
    success1, img1 = cap.read()
    hands, img1 = detector.findHands(img1)
    # crop hand
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # setting image matrix which is of fixed size
        img_croped = img1[y - offset:y + h + offset, x - offset:x + w + offset]  # setting padding

        imgCropShape = img_croped.shape
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            widthCalculated = math.ceil(k * w)
            # img resize
            imgResize = cv2.resize(img_croped, (widthCalculated, imgSize))
            imgResizeShape = imgResize.shape
            widthGap = math.ceil((imgSize - widthCalculated) / 2)  # setting vertical images at the centre
            # overlay croped img on the top of white image to form a fixed size img
            imgWhite[:, widthGap:widthCalculated + widthGap] = imgResize

        else:
            k = imgSize / w
            heightCalculated = math.ceil(k * h)
            # img resize
            imgResize = cv2.resize(img_croped, (imgSize, heightCalculated))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((imgSize - heightCalculated) / 2)  # setting vertical images at the centre
            # overlay croped img on the top of white image to form a fixed size img
            imgWhite[heightGap:heightCalculated + heightGap, :] = imgResize

        # cv2.imshow("Cropped", img_croped)
        cv2.imshow("Image_Matrix", imgWhite)

    cv2.imshow("Image", img1)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

