import pytesseract
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from helpers import calc_geo, geometry_data, merge_boxes
import re

config_tesseract = '--oem 3 --psm 6'
image = cv2.imread('./src/foto.jpg')
image = cv2.imread('./src/placarj.webp')

cv2.imshow('original image', image)

model = "./src/frozen_east_text_detection.pb"
layers = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

modelW, modelH = 320,320
image_resized = cv2.resize(image, (modelW, modelH))
blob = cv2.dnn.blobFromImage(image_resized, 1.0, (modelW, modelH), swapRB=True, crop=False)
neural_network = cv2.dnn.readNet(model)

neural_network.setInput(blob)

scores, geometry = neural_network.forward(layers)

lines, columns = scores.shape[2:4]
print(scores.shape)
     
confidence_level = 0.8
boxes = []
confidences = []
for y in range(lines):
    data_scores = scores[0,0,y]
    
    dtop, dright, dbottom, dleft, angle = geometry_data(geometry, y)
    
    for x in range(columns):
        if(data_scores[x] >= confidence_level):
            
            intiX, initY, endX, endY = calc_geo(dtop[x], dright[x], dbottom[x], dleft[x], angle[x], x, y)
            confidences.append(data_scores[x])
            boxes.append((intiX, initY, endX, endY))

detections = non_max_suppression(np.array(boxes), probs=confidences)

proportionH = image.shape[0]/float(modelH)
proportionW = image.shape[1]/float(modelW)

processed_boxes = []
roiImages = []
for (initX, initY, endX, endY) in detections:
    initX = int(initX*proportionW * 0.9)
    endX = int(endX*proportionW  * 1.05)
    initY = int(initY*proportionH * 0.9)
    endY = int(endY*proportionH * 1.05)
    
    """ processed_boxes.append((initX, initY, endX, endY)) """
    if endX > initX and endY > initY:  # Verifica se o ROI é válido
        roi = image[initY:endY, initX:endX]
        if roi.size > 0:  # Verifica se o ROI não está vazio
            processed_boxes.append((initX, initY, endX, endY))
    
merged_boxes = merge_boxes(processed_boxes, 100)

for (initX, initY, endX, endY) in merged_boxes:
    
    roi = image[initY:endY, initX:endX]
    roiImages.append(roi)
    cv2.rectangle(image, (initX, initY, endX, endY), (0,255,0), 1)

strings = []

for i, roi in enumerate(roiImages):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    print(roi.shape)
    print(f'imaggem {i} média:{np.mean(roi)}')

    if(roi.shape[0] < 200 or roi.shape[1] < 200):
        roi = cv2.resize(roi, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
        roi = cv2.dilate(roi, np.ones((3,3), np.uint8))
        roi = cv2.erode(roi, np.ones((3,3), np.uint8))
    
    if(np.mean(roi) <= 85):
        """ roi = cv2.threshold(roi, 80, 255, cv2.THRESH_BINARY)[1] """
        #roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 9)
        """ _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) """
        roi = cv2.bitwise_not(roi)
        roi = _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    elif(np.mean(roi) <= 170):
        roi = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)[1]
    else:
        roi = 255 - roi
        roi = cv2.threshold(roi, 70, 255, cv2.THRESH_BINARY)[1]


    #roi = cv2.threshold(roi, 80, 255, cv2.THRESH_BINARY)[1]
    
    """ roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] """
    
    
    cv2.imshow(f'roi {i}', roi)
    
    string = pytesseract.image_to_string(roi, 'por', config = config_tesseract)
    strings.append(string)

print(strings)

cv2.waitKey(0)
cv2.destroyAllWindows()