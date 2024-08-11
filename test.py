import math
import time
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

offset = 20
imgSize = 300
counter = 0

model_path = 'model/ISL_model1_Full5.h5'
model = load_model(model_path)

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=2)

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    imgOut = img.copy()
    hands, img = detector.findHands(img)
    landmark_image = np.zeros_like(img)

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
        imgCropShape = imgCropped.shape
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            widthCalculated = math.ceil(k * w)
            imgResize = cv2.resize(imgCropped, (widthCalculated, imgSize))
            imgResizeShape = imgResize.shape
            widthGap = math.ceil((imgSize - widthCalculated) / 2)
            imgWhite[:, widthGap:widthCalculated + widthGap] = imgResize
        else:
            k = imgSize / w
            heightCalculated = math.ceil(k * h)
            imgResize = cv2.resize(imgCropped, (imgSize, heightCalculated))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((imgSize - heightCalculated) / 2)
            imgWhite[heightGap:heightCalculated + heightGap, :] = imgResize

        # Convert imgWhite to grayscale
        # imgWhiteGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)


        # Preprocess the frame (resize and normalize as per your training setup)
        input_size = (300, 300)  # Update input size to match the training size
        frame_resized = cv2.resize(imgWhite, input_size)
        frame_normalized = frame_resized / 255.0

        frame_expanded = np.expand_dims(frame_normalized, axis=-1)  # Add channel dimension for grayscale image
        frame_expanded = np.expand_dims(frame_expanded, axis=0)  # Add batch dimension

        # Get predictions
        predictions = model.predict(frame_expanded)
        predicted_class = np.argmax(predictions)
        predicted_label = labels[predicted_class]
        confidence = np.max(predictions)

        # Display the predictions on the frame
        cv2.putText(imgWhite, f'{predicted_label}: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

        # Show the frame
        cv2.imshow('Real-time Predictions', imgWhite)
        print(predicted_label, confidence)

        cv2.rectangle(imgOut, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (230, 230, 0), cv2.FILLED)
        cv2.putText(imgOut, predicted_label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOut, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (230, 230, 0), 4)

    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)
