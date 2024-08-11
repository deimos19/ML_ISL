import math
import time

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

offset = 20
imgSize = 300
counter = 0
folder = "landmark_data/Z"
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    # Create a blank image for displaying landmarks
    landmark_image = np.zeros_like(img)

    if hands:
        for hand in hands:
            # Get the landmarks
            lmList = hand["lmList"]

            # Draw the landmarks and lines on the landmark_image
            for i, lm in enumerate(lmList):
                # Draw the landmarks
                cv2.circle(landmark_image, (lm[0], lm[1]), 5, (0, 255, 0), cv2.FILLED)


            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),        # Index
                (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
                (0, 13), (13, 14), (14, 15), (15, 16), # Ring
                (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
                (5, 9), (9, 13), (13, 17)              # Palm
            ]

            for start, end in connections:
                cv2.line(landmark_image, (lmList[start][0], lmList[start][1]), (lmList[end][0], lmList[end][1]), (0, 255, 0), 2)

        # Find the bounding box that encompasses both hands
        x_min = min(hand['bbox'][0] for hand in hands)
        y_min = min(hand['bbox'][1] for hand in hands)
        x_max = max(hand['bbox'][0] + hand['bbox'][2] for hand in hands)
        y_max = max(hand['bbox'][1] + hand['bbox'][3] for hand in hands)

        # Define the combined bounding box
        x = x_min - offset
        y = y_min - offset
        w = x_max - x_min + 2 * offset
        h = y_max - y_min + 2 * offset

        # Crop the region defined by the combined bounding box
        imgCropped = landmark_image[y:y + h, x:x + w]

        # Create a white image of fixed size
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Resize the cropped image to fit within imgWhite
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            widthCalculated = math.ceil(k * w)
            imgResize = cv2.resize(imgCropped, (widthCalculated, imgSize))
            widthGap = math.ceil((imgSize - widthCalculated) / 2)
            imgWhite[:, widthGap:widthCalculated + widthGap] = imgResize

        else:
            k = imgSize / w
            heightCalculated = math.ceil(k * h)
            imgResize = cv2.resize(imgCropped, (imgSize, heightCalculated))
            heightGap = math.ceil((imgSize - heightCalculated) / 2)
            imgWhite[heightGap:heightCalculated + heightGap, :] = imgResize
            # hsv_image = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2HSV)
            #



        cv2.imshow("Image_Matrix", imgWhite)
        cv2.imshow("cropped",imgCropped)
        # cv2.imshow('Landmark and Line Detection', landmark_image)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        # setting timer to read images
        seconds = 5
        if hands:
            for i in range(seconds):
                print(f"{i + 1} second(s) have passed")
                time.sleep(1)  # Pause for 1 second

            # Recheck for hands and capture images if hands are detected
            hands_detected = False
            for i in range(500):
                success, img = cap.read()
                hands, img = detector.findHands(img)

                if hands:
                    for hand in hands:
                        # Get the landmarks
                        lmList = hand["lmList"]

                        # Draw the landmarks and lines on the landmark_image
                        for i, lm in enumerate(lmList):
                            # Draw the landmarks
                            cv2.circle(landmark_image, (lm[0], lm[1]), 5, (0, 255, 0), cv2.FILLED)

                        # Draw the connections (lines) between landmarks
                        connections = [
                            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                            (5, 9), (9, 13), (13, 17)  # Palm
                        ]
                        for start, end in connections:
                            cv2.line(landmark_image, (lmList[start][0], lmList[start][1]),
                                     (lmList[end][0], lmList[end][1]), (0, 255, 0), 2)

                    hands_detected = True

                    # Find the bounding box that encompasses both hands
                    x_min = min(hand['bbox'][0] for hand in hands)
                    y_min = min(hand['bbox'][1] for hand in hands)
                    x_max = max(hand['bbox'][0] + hand['bbox'][2] for hand in hands)
                    y_max = max(hand['bbox'][1] + hand['bbox'][3] for hand in hands)

                    # Define the combined bounding box
                    x = x_min - offset
                    y = y_min - offset
                    w = x_max - x_min + 2 * offset
                    h = y_max - y_min + 2 * offset

                    # Crop the region defined by the combined bounding box
                    imgCropped = landmark_image[y:y + h, x:x + w]

                    # Create a white image of fixed size
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                    # Resize the cropped image to fit within imgWhite
                    aspectRatio = h / w
                    if aspectRatio > 1:
                        k = imgSize / h
                        widthCalculated = math.ceil(k * w)
                        imgResize = cv2.resize(imgCropped, (widthCalculated, imgSize))
                        widthGap = math.ceil((imgSize - widthCalculated) / 2)
                        imgWhite[:, widthGap:widthCalculated + widthGap] = imgResize
                    else:
                        k = imgSize / w
                        heightCalculated = math.ceil(k * h)
                        imgResize = cv2.resize(imgCropped, (imgSize, heightCalculated))
                        heightGap = math.ceil((imgSize - heightCalculated) / 2)
                        imgWhite[heightGap:heightCalculated + heightGap, :] = imgResize

                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    counter += 1
                    print(counter)
                else:
                    print("No hands detected, cannot save image.")
                    break  # Exit the loop if no hands are detected during the capture process


                cv2.imshow("Image_Matrix", imgWhite)
                cv2.imshow("cropped", imgCropped)
                # cv2.imshow('Landmark and Line Detection', landmark_image)
        cv2.imshow("Image", img)