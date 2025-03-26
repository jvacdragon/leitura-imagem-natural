import pytesseract
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression #para identificação dos melhores boxes que nao se sobrepoem


config_tesseract = '--oem 3 --psm 6'
image = cv2.imread('./src/fake-traffic-sign.jpg')
cv2.imshow('original image', image)

model = "./src/frozen_east_text_detection.pb"
layers = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

modelW, modelH = 320,320
blob = cv2.dnn.blobFromImage(image, 1.0, (modelW, modelH), swapRB=True, crop=False)
neural_network = cv2.dnn.readNet(model)

neural_network.setInput(blob)

scores, geometry = neural_network.forward(layers)

lines, columns = scores.shape[2:4]
print(scores.shape)




cv2.waitKey(0)
cv2.destroyAllWindows()