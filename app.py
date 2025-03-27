import pytesseract
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

from helpers import calc_geo, geometry_data

#from helpers import calc_geo, geometry_data #para identificação dos melhores boxes que nao se sobrepoem


config_tesseract = '--oem 3 --psm 6'
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

print(confidences)
print(boxes)

detections = non_max_suppression(np.array(boxes), probs=confidences)
print(detections)

proportionH = image.shape[0]/float(modelH)
proportionW = image.shape[1]/float(modelW)

for (initX, initY, endX, endY) in detections:
    initX = int(initX*proportionW)
    endX = int(endX*proportionW)
    initY = int(initY*proportionH)
    endY = int(endY*proportionH)
    
    roi = image[initY:endY, initX:endX]
    cv2.rectangle(image, (initX, initY), (endX, endY), (0,255,0), 2)

cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()