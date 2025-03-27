import numpy as np

def geometry_data(geometry, y):
    dtop = geometry[0,0,y]
    dright = geometry[0,1,y]
    dbottom = geometry[0,2,y]
    dleft = geometry[0,3,y]
    angle = geometry[0,4,y]
    
    return dtop, dright, dbottom, dleft, angle

def calc_geo(data_dtop, data_dright, data_dbottom, data_dleft, data_angle, x, y):
    (offsetx, offsety) = (x*4, y*4)
    angles = data_angle
    cos = np.cos(angles)
    sin = np.sin(angles)
    h = data_dtop + data_dbottom
    w = data_dleft + data_dright
    
    #fórmula de rotação em plano (2 dimensões)
    endX = int(offsetx + (cos * data_dright) + (sin * data_dbottom))
    endY = int(offsety - (sin * data_dright) + (cos * data_dbottom))
    
    initX = int(endX - w)
    initY = int(endY - h)
    
    return initX, initY, endX, endY
    

def merge_boxes(boxes, threshold_distance=20):
    merged = []
    for box in boxes:
        x1, y1, x2, y2 = box
        found = False
        for i, m in enumerate(merged):
            mx1, my1, mx2, my2 = m
            # Verifica se a caixa atual está próxima da caixa merged
            if (abs(x1 - mx1) < threshold_distance and abs(y1 - my1) < threshold_distance):
                # Combina as coordenadas
                new_x1 = min(x1, mx1)
                new_y1 = min(y1, my1)
                new_x2 = max(x2, mx2)
                new_y2 = max(y2, my2)
                merged[i] = (new_x1, new_y1, new_x2, new_y2)
                found = True
                break
        if not found:
            merged.append(box)
    return merged

