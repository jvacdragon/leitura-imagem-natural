import pytesseract
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from helpers import calc_geo, geometry_data, merge_boxes

config_tesseract = '--oem 3 --psm 6'
image = cv2.imread('./src/foto.jpg')

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
for (initX, initY, endX, endY) in detections:
    initX = int(initX*proportionW - int(initX*proportionW)/20)
    endX = int(endX*proportionW + int(endX*proportionW)/20)
    initY = int(initY*proportionH)
    endY = int(endY*proportionH)
    
    roi = image[initY:endY, initX:endX]
    
    print(initX, initY, endX, endY)
    processed_boxes.append((initX, initY, endX, endY))

for box in merge_boxes(processed_boxes, 100):
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)

cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()