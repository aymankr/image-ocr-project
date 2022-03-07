from skimage import io
from skimage.transform import resize
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from typing import Iterator

SIZE = 60
THRESHOLD = 150
FIRST = 0
SECOND = 0
confusionMatrix = [[0 for col in range(10)] for row in range(10)]


def bold(str):
    return '\033[1m' + str + '\033[0m'

def MapToHProfile(matrix: np.ndarray):
    return SIZE - matrix.sum()
    
def getHorizontaleProfile(img):
    img = resize(img, (SIZE, SIZE), preserve_range=True).astype('uint8')
    binarized = 1.0 * (img > THRESHOLD)
    binarized = binarized[:,:,0]
    return (list(map(MapToHProfile, binarized)),binarized)

def getAllVectors():
    vectors = []
    for i in range(0,10):
        for j in range(1,11):
            name = f'baseProjetOCR/{i}_{j}.png'
            img = io.imread(name)
            vector = getHorizontaleProfile(img)[0]
            vectors.append((name,vector))
    return vectors

ALL_VECTORS = getAllVectors()
    

# read in image as 8 bit grayscale
# img = io.imread(f'baseProjetOCR/{FIRST}_{SECOND}.png')

# hProfile = getHorizontaleProfile(img)
# vector = hProfile[0]
# img = hProfile[1]

def start():
    global FIRST, SECOND
    for i in range(0,10):
        for j in range(1,11):
            FIRST = i
            SECOND = j
            unknownImageFile = f'baseProjetOCR/{i}_{j}.png'
            print(f'reading file { bold(unknownImageFile.split("/")[1]) }', end=' ')
            unknownIMG = io.imread(unknownImageFile)
            unknownHProfile = getHorizontaleProfile(unknownIMG)
            found = guessNumber(unknownHProfile[0], ALL_VECTORS)
            print('-- ' + ("success" if int(found) == i else ("wrong (found " + found + " instead of " + str(i) + ")")) + " !")
            confusionMatrix[i][int(found)] += 1
    print(np.matrix(confusionMatrix))
            
            
            



def findMatch(aVector, vectors):
    global FIRST,SECOND
    highest = 'null'
    highScore = 0
    for item in vectors:
        filename = item[0]
        numbers = numbersFromFile(filename)
        if (numbers[0] == FIRST and numbers[1] == SECOND):
            continue
        vals = item[1]
        score = 0
        for index in range(SIZE):
            difference = max(aVector[index],vals[index]) - min(aVector[index],vals[index])
            score += difference if (difference <= 3) else 0
        if (score > highScore):
            highest = item[0]
            highScore = score
    print(f'-- best match: { bold(highest.split("/")[1]) }', end=' ')
    return highest

def numbersFromFile(name):
    exactName = name.split('/')[1]
    first = exactName[0]
    second = exactName[2]
    return (first, second)

def guessNumber(aVector, vectors):
    return numbersFromFile(findMatch(aVector, vectors))[0]
    
def getSuccessRate():
    sum = 0
    for i in range(10):
        sum += confusionMatrix[i][i]
    print(f'Taux de succès : {sum} %')
    return sum

start()
getSuccessRate()