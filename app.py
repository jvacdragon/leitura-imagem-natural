import pytesseract
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from helpers import calc_geo, geometry_data, merge_boxes
import re

config_tesseract = '--oem 3 --psm 6'
image = cv2.imread('./src/placarj.webp')

model = "./src/frozen_east_text_detection.pb"
layers = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

modelW, modelH = 320,320
image_resized = cv2.resize(image, (modelW, modelH))
blob = cv2.dnn.blobFromImage(image_resized, 1.0, (modelW, modelH), swapRB=True, crop=False)
neural_network = cv2.dnn.readNet(model)
neural_network.setInput(blob)
scores, geometry = neural_network.forward(layers)

lines, columns = scores.shape[2:4]
confidence_level = 0.8
boxes = []
confidences = []

#Passando por todos os scores para definir quais geometry serçao utilizados para formação dos ROIs
for y in range(lines):
    data_scores = scores[0,0,y]
    
    #Pegando distancias para formar bounding boxes do ROI referentes aos scores que estçao sendo analizados no momento
    dtop, dright, dbottom, dleft, angle = geometry_data(geometry, y)
    
    for x in range(columns):
        #verificando se no nivel de confiança a ser olhado está no aceitável ou nao. Caso esteja, é calculada as coordenadas para formar o bounding box da área referente a esse nivel de confiança utilizando a função calc_geo e então essas coordenadas sao adicionadas ao array de boxes
        if(data_scores[x] >= confidence_level):
            
            intiX, initY, endX, endY = calc_geo(dtop[x], dright[x], dbottom[x], dleft[x], angle[x], x, y)
            confidences.append(data_scores[x])
            boxes.append((intiX, initY, endX, endY))

#Utilizando non-max-supression para determinar os melhores dados a serem usados para criar os ROIs
detections = non_max_suppression(np.array(boxes), probs=confidences)

#Proporção de altura e largura para usar de calculo na criação das bounding boxes na imagem original
proportionH = image.shape[0]/float(modelH)
proportionW = image.shape[1]/float(modelW)

processed_boxes = []
roiImages = []

#Delimitando as coordenadas da bounding boxes dos ROIs. As margens são aumentadas para evitar possiveis erros de detecção
for (initX, initY, endX, endY) in detections:
    initX = int(initX*proportionW * 0.9)
    endX = int(endX*proportionW  * 1.05)
    initY = int(initY*proportionH * 0.9)
    endY = int(endY*proportionH * 1.05)
    
    if endX > initX and endY > initY:
        roi = image[initY:endY, initX:endX]
        if roi.size > 0:
            processed_boxes.append((initX, initY, endX, endY))

#Identificando onde se deve fazer o merge das boxes que irçao delimitar o ROI, com base na proximidade dessas boxes e na sobreposição delas umas com as outras
merged_boxes = merge_boxes(processed_boxes, 100)

#Delimitando a área de todos os ROI (Region of Interest) e adicionando no array dedicado para esses cortes de imagem
for (initX, initY, endX, endY) in merged_boxes:
    roi = image[initY:endY, initX:endX]
    roiImages.append(roi)
    cv2.rectangle(image, (initX, initY), (endX, endY), (0,255,0), 1)

#Onde serão armazenadas os textos da imagem
strings = []

#Fazendo processamentod a imagem para cada ROI (Region of Interest) encontrado na imagem original
for i, roi in enumerate(roiImages):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    #Aumentando tamanho da imagem caso ela tenha altura ou largura menor que 200 pixels para facilitar a leitura dela pelo tessseract
    if(roi.shape[0] < 200 or roi.shape[1] < 200):
        roi = cv2.resize(roi, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
        roi = cv2.dilate(roi, np.ones((3,3), np.uint8))
        roi = cv2.erode(roi, np.ones((3,3), np.uint8))
    
    mean = (np.percentile(roi, 25) + np.percentile(roi, 75))/2
    
    #Fazendo o processamento da imagem e aplicando métodos de thresholding e dilatação/erosão basedo na média de pixels da imagem entre os percentis de 25% e 75%
    if(mean <= 50 ):       
        _, roi = cv2.threshold(roi, 75, 255, cv2.THRESH_BINARY)
        roi = cv2.erode(roi, np.ones((3,3), np.uint8))
        
    elif(mean <= 100):        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        roi = clahe.apply(roi)
        
        roi = cv2.threshold(roi, 100 - np.mean(roi)/2, 255, cv2.THRESH_BINARY)[1]
        
        roi = cv2.erode(roi, np.ones((5,5), np.uint8))
        roi = cv2.dilate(roi, np.ones((3,3), np.uint8))
    
    elif(np.mean(roi) <= 170):
        roi = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)[1]
    else:
        roi = 255 - roi
        roi = cv2.threshold(roi, 70, 255, cv2.THRESH_BINARY)[1]    
    
    string = pytesseract.image_to_string(roi, 'por', config = config_tesseract)
    
    #Pós processamento, corrigindo possíveis erros na hora de guardar as informações da imagem
    string = re.sub(r'[^a-zA-ZÀ-ÿ0-9"\s]', '', string)
    string = string.replace("\n", " ")
    string = re.sub(r'^\s*[a-zA-Z]\s', ' ', string)
    string = re.sub(r'\s[a-zA-Z]\s*$', ' ', string)
    string = re.sub(r'\s+', ' ', string).strip()
    strings.append(string)

print(strings)

cv2.waitKey(0)
cv2.destroyAllWindows()