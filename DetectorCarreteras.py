# -*- coding: utf-8 -*-

from pathlib import Path
from skimage import io, color, morphology, feature, util
import numpy

#TODO: pre-processing image before using Canny so that u get only the road's edges
#Clipping the skeletonized image so that u only have the road and not the branches
#(after that u can dilate it and consider it the detected road)

#Lista de imágenes: (el formato es .tiff para las fotos y .tif para ground truth)
#10078675_15.tiff       1
#10078690_15.tiff       2
#10678720_15.tiff       3
#10678735_15.tiff       4
#10828735_15.tiff       5
#10828750_15.tiff       6
#11428690_15.tiff       7
#11428705_15.tiff       8
#12478735_15.tiff       9
#12478750_15.tiff       10

if __name__ == '__main__':
    imageName = '10078675_15.tiff'
    GTName = '10078675_15.tif' #IMPORTANTE a la hora de escoger el ground truth, recordar que el formato es ".tif" y no ".tiff"
    try:
        imageDir = Path(__file__).parent / 'imag' / imageName  
    except Exception as e:
        print(e, "Image not found, please make sure it's on the same directory as the program files")
        
    try:
        groundTruthDir = Path(__file__).parent / 'gt' / GTName
    except Exception as e:
        print(e, "Image not found, please make sure it's on the same directory as the program files")
        
    image = io.imread(imageDir) #imagen en RGB
    image_hsv = color.rgb2hsv(image) #imagen en hsv
    image_hue = image_hsv[:,:,0]
    image_sat = image_hsv[:,:,1]
    image_val = image_hsv[:,:,2]
    
    groundTruth = util.img_as_float(io.imread(groundTruthDir))
            
    edgesCanny = feature.canny(color.rgb2gray(image), low_threshold=0.3, high_threshold=0.4) #Buscamos los bordes suficientemente notables como para pertenecer a la carretera
    disk_elem = morphology.disk(10)   
    edges = morphology.closing(edgesCanny, disk_elem)   #Cerramos los bordes, de esta forma se rellena la carretera, por lo que detectará todos los pixels de la carretera y no solo los bordes  
    #Obviamente esto también detecta cosas que no son la carretera, pero nos deshacemos de esas cosas más adelante
    image_thresh = numpy.zeros((image_hsv.shape[0],image_hsv.shape[1]))
    for y in range (0, image_hsv.shape[1]):
        for x in range(0,image_hsv.shape[0]):
            image_thresh[x,y] = (image_sat[x,y]<0.2 and edges[x,y] and image_hue[x,y] >0.2 and image_hue[x,y] <0.8)
            #seleccionamos aquellos pixels con una saturación y brillo adecuados, y que formen parte de la imagen de bordes calculada anteriormente 
    
    filtered = morphology.remove_small_objects(image_thresh.astype(bool), min_size = 700)
    #Nos deshacemos de todo lo que sea inferior a cierto tamaño, inferior al tamaño de las carreteras

    disk_elem = morphology.disk(1)  
    closed = morphology.closing(filtered,disk_elem)
    
    cleanRoads = numpy.zeros((image_hsv.shape[0],image_hsv.shape[1]))
    
    for y in range (0, image_hsv.shape[1]):
        for x in range(0,image_hsv.shape[0]):
            cleanRoads[x,y] = (closed[x,y] and not edgesCanny[x,y]) #Al excluir los bordes, deberíamos "cortar" las cosas que están pegadas a la carretera
            #(Esto funciona a medias, ciertamente corta algunas cosas, pero no todas)
            
            
            
    cleanRoads = morphology.remove_small_objects(cleanRoads.astype(bool), min_size = 200)
    #Nos deshacemos de todo lo que sea inferior a cierto tamaño, inferior al tamaño de las carreteras

    disk_elem = morphology.disk(2)  
    cleanRoads = morphology.closing(cleanRoads,disk_elem)
    
    imgDifference = cleanRoads * groundTruth
    
    roadsImgDif = numpy.sum(imgDifference)
    roadsGT = numpy.sum(groundTruth)
    roadsResult = numpy.sum(cleanRoads)
    
    groundImgDif = imgDifference.size - roadsImgDif
    groundGT = groundTruth.size - roadsGT
    groundResult = cleanRoads.size - roadsResult
    
    falseRoads = roadsResult - roadsImgDif #carreteras detectadas que no están en el ground truth
    falseGround = roadsGT - roadsImgDif #Carreteras del ground truth que no fueron detectadas
    
    truePos = roadsImgDif / roadsGT #Fracción de verdaderos positivos
    trueNeg = (groundResult-falseGround) / groundGT #fracción de verdaderos negativos (negativos obtenidos menos los falsos)
    falsePos = falseRoads / groundGT #fracción de falsos positivos. = 1-trueNeg
    falseNeg = falseGround / roadsGT #fracción de falsos negativos. = 1-truePos
    
    print('True positive fraction', truePos)
    print('True negative fraction', trueNeg)
    print('False positive fraction', falsePos)
    print('False negative fraction', falseNeg)
    
    skeleton = morphology.skeletonize(cleanRoads)
    roadLength = numpy.sum(skeleton)
    
    print('Road length:',roadLength, 'px')
    print('')
    
    io.imshow(cleanRoads)
    io.show()
    
    io.imsave("edges.png",util.img_as_ubyte(edgesCanny))
    io.imsave("edgesClosed.png",util.img_as_ubyte(edges))
    io.imsave("thresholded.png",util.img_as_ubyte(image_thresh))
    io.imsave("filtered.png",util.img_as_ubyte(filtered))
    io.imsave("closed.png",util.img_as_ubyte(closed))
    io.imsave("final.png",util.img_as_ubyte(cleanRoads))
    io.imsave("correctRoads.png",util.img_as_ubyte(imgDifference))
    io.imsave("skeleton.png",util.img_as_ubyte(skeleton))
    
