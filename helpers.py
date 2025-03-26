def geometry_data(geometry, y):
    dtop = geometry[0,0,y]
    dright = geometry[0,1,y]
    dbottom = geometry[0,2,y]
    dleft = geometry[0,3,y]
    angle = geometry[0,4,y]
    
    return dtop, dright, dbottom, dleft, angle

def calc_geo(dtop, dright, dbottom, dleft, angle):
    