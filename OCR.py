import math
from skimage import io
from skimage.transform import resize
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np
from art import *

class OCR():
    guesser = {}
    def __init__(self, *args):
        self.ph = ProfilH()
        self.phConMat = self.ph.getProfileVHConfusionMatrix()
        self.knn = KNN()
        self.knnConMat = self.knn.getKnnConfusionMatrix()
        self.zn = Zoning()
        self.znConMat = self.zn.getZoningConfusionMatrix()
        self.finalConfMatrix = np.zeros((10,10))
        self.guesser = {"knn": self.guessFromKnn, "zn": self.guessFromZoning, "ph": self.guessFromHProfile}
        self.stats = {"knn": 0, "zn": 0, "ph": 0}
        self.run()
        
        
    def run(self):
        for nb in range(0,10):
            for nth in range(1,11):
                name = f'baseProjetOCR/{nb}_{nth}.png'
                alg = self.bestAlgoForNb(nb)
                self.stats[alg] += 1
                foundNb = self.guesser[alg](name)
                self.finalConfMatrix[nb][foundNb] += 1
        print(self.finalConfMatrix)
        print(self.stats)
        print(f'Succès : {np.trace(self.finalConfMatrix)} % ')
            
    def guessFromKnn(self, file):
        return self.knn.guessFromFilename(file)
    
    def guessFromHProfile(self, file):
        return int(self.ph.guessFromFileName(file))
    
    def guessFromZoning(self, file):
        return self.zn.guessFromFilename(file)
                

    def bestAlgoForNb(self, nb):
        knn = self.knnConMat[nb][nb]
        zn = self.znConMat[nb][nb]
        ph = self.phConMat[nb][nb]
        if (knn > zn and knn > ph):
            return "knn"
        elif (zn >= knn and zn >= ph):
            return "zn"
        else:
            return "ph"
        
        
        
                

class ProfilH:
    SIZE = 60
    THRESHOLD = 150
    
    def __init__(self, *args):
        self.FIRST = 0,
        self.SECOND = 0
        self.initConfusionMatrix()
        self.initScoreMatrix()
        self.getAllVectors()
        #self.getProfileVHConfusionMatrix()
        
    def getScoreTab(self, number, nth):
        return self.scoreMatrix[f'{number}-{nth}']
    
    def initConfusionMatrix(self):
        self.confusionMatrix = [[0 for col in range(10)] for row in range(10)]
        
    def initScoreMatrix(self):
        self.scoreMatrix = {}
        
        
    def getAllVectors(self):
        vectors = []
        for i in range(0,10):
            for j in range(1,11):
                name = f'baseProjetOCR/{i}_{j}.png'
                img = io.imread(name)
                vector = self.getHorizontaleProfile(img)[0]
                vectors.append((name,vector))
        self.ALL_VECTORS = vectors
        
    def getProfileVHConfusionMatrix(self):
        self.initConfusionMatrix()
        for i in range(0,10):
            for j in range(1,11):
                self.FIRST = i
                self.SECOND = j
                unknownImageFile = f'baseProjetOCR/{i}_{j}.png'
                #print(f'reading file { self.bold(unknownImageFile.split("/")[1]) }', end=' ')
                unknownIMG = io.imread(unknownImageFile)
                unknownHProfile = self.getHorizontaleProfile(unknownIMG)
                found = self.guessNumber(unknownHProfile[0], self.ALL_VECTORS)
                #print('-- ' + ("success" if int(found) == i else ("wrong (found " + found + " instead of " + str(i) + ")")) + " !")
                self.confusionMatrix[i][int(found)] += 1
        return self.confusionMatrix
        
    def bold(self, str):
        return '\033[1m' + str + '\033[0m'

    def MapToHProfile(matrix: np.ndarray):
        return ProfilH.SIZE - matrix.sum()
        
    def getHorizontaleProfile(self,img):
        img = resize(img, (self.SIZE, self.SIZE), preserve_range=True).astype('uint8')
        binarized = 1.0 * (img > self.THRESHOLD)
        binarized = binarized[:,:,0]
        o = self
        return (list(map(ProfilH.MapToHProfile, binarized)),binarized)
    
    def findMatch(self, aVector, vectors):
        scoreTab = [[0 for col in range(10)] for row in range(10)]
        highest = 'null'
        highScore = 0
        for item in vectors:
            filename = item[0]
            numbers = self.numbersFromFile(filename)
            if ( (int(numbers[0]) == self.FIRST) and (int(numbers[1]) == self.SECOND) ):
                continue
            vals = item[1]
            score = 0
            for index in range(self.SIZE):
                difference = max(aVector[index],vals[index]) - min(aVector[index],vals[index])
                score += difference if (difference <= 3) else 0
            scoreTab[int(numbers[0])][int(numbers[1]) - 1] = np.round( (score / 180.0)*100, 1)
            if (score > highScore):
                highest = item[0]
                highScore = score
        #print(f'-- best match: { self.bold(highest.split("/")[1]) }')
        self.scoreMatrix[f'{self.FIRST}-{self.SECOND}'] = scoreTab
        return (highest, scoreTab)

    def numbersFromFile(self, name):
        exactName = name.split('/')[1]
        first = exactName[0]
        second = exactName[2]
        return (first, second)
    
    def guessNumber(self, aVector, vectors):
        scores = self.findMatch(aVector, vectors)
        #print(np.matrix(scores[1]))
        return self.numbersFromFile(scores[0])[0]
    
    def guessFromFileName(self, filename):
        unknownIMG = io.imread(filename)
        unknownHProfile = self.getHorizontaleProfile(unknownIMG)
        return self.guessNumber(unknownHProfile[0], self.ALL_VECTORS)
        
        
    def getSuccessRate(self):
        sum = 0
        for i in range(10):
            sum += self.confusionMatrix[i][i]
        print(f'Taux de succès : {sum} %')
        return sum
    
class KNN():
    
    def __init__(self, *args):
        pass
        
    def getTrainingDistanceForTestSample(self, X_train, test_sample):
        return [euclideanDistance(train_sample,test_sample) for train_sample in X_train]

    def get_most_frequent_element(self, l):
        return max(l, key=l.count)
    
    def guessFromFilename(self, file):
        fileNames = []
        imageData = {}

        for i in range(0,10):
            for x in range(1,11):
                    name = f'baseProjetOCR/{i}_{x}.png'
                    fileNames.append(name)


        for name in fileNames:
            image = io.imread(name)
            imMatrix= imgToBinaryMatrix(image, 35, 6)
            meanKernel = np.full((3, 3), 1.0/9)
            imMatrix = ndi.correlate(imMatrix, meanKernel)
            imageData[name]=np.concatenate(imMatrix)

        
        imageDataTest = self.getValuesWithoutSample(imageData, file)
        return self.knn(imageDataTest.values(),
        [getNumberOfFileName(file) for name in imageDataTest.keys()],
        imageData[file],5)

    def knn(self, X_train, Y_train, sample, k=3):

        training_disance = self.getTrainingDistanceForTestSample(X_train, sample)
        sorted_distance_indices = [ 
            pair[0]
            for pair in sorted(enumerate(training_disance), key=lambda x: x[1])
        ]

        candidates = [
            Y_train[idx]
            for idx in sorted_distance_indices[:k]
        ]
        top_candidate = self.get_most_frequent_element(candidates)
        return top_candidate

    def getValuesWithoutSample(self, dictionnary, key):
        dic = {}
        dic = dictionnary.copy()
        dic.pop(key)
        return dic
    
    def getKnnConfusionMatrix(self):
        fileNames = []
        imageData = {}
        mat = np.zeros(shape=(10, 10), dtype=np.uint8)

        for i in range(0,10):
            for x in range(1,11):
                    name = f'baseProjetOCR/{i}_{x}.png'
                    fileNames.append(name)


        for name in fileNames:
            image = io.imread(name)
            imMatrix= imgToBinaryMatrix(image, 35, 6)
            meanKernel = np.full((3, 3), 1.0/9)
            imMatrix = ndi.correlate(imMatrix, meanKernel)
            imageData[name]=np.concatenate(imMatrix)

        for key in imageData:
            imageDataTest = self.getValuesWithoutSample(imageData, key)
            n = self.knn(imageDataTest.values(),
            [getNumberOfFileName(name) for name in imageDataTest.keys()],
            imageData[key],5)
            mat[getNumberOfFileName(key)][n]+=1
        return mat
        
class Zoning():
    
    
    def __init__(self, *args):
        self.initScoreMatrices()
        return

    def initScoreMatrices(self):
        self.scoreMatrices = {}
    
    def convertMatrixTo4x4(self,matrix):
        xs = matrix.shape[0]//4  # division lines for the picture
        ys = matrix.shape[1]//4

        # now slice up the image (in a shape that works well with subplots)
        newMatrix = [[matrix[0:xs, 0:ys], matrix[0:xs, ys:ys*2], matrix[0:xs, ys*2:ys*3], matrix[0:xs, ys*3:ys*4]],
                    [matrix[xs:xs*2, 0:ys], matrix[xs:xs*2, ys:ys*2],
                        matrix[xs:xs*2, ys*2:ys*3], matrix[xs:xs, ys*3:ys*4]],
                    [matrix[xs*2:xs*3, 0:ys], matrix[xs*2:xs*3, ys:ys*2],
                        matrix[xs*2:xs*3, ys*2:ys*3], matrix[xs*2:xs*3, ys*3:ys*4]],
                    [matrix[xs*3:xs*4, 0:ys], matrix[xs*3:xs*4, ys:ys*2],
                    matrix[xs*3:xs*4, ys*3:ys*4], matrix[xs*3:xs*4, ys*3:ys*4]]
                    ]
        return newMatrix

    def convert4x4ToVector(self,matrix):
        vector = []
        for i in range(4):
            for j in range(4):
                vector.append(np.sum(matrix[i][j]))
        return vector
    
    def getComparedVectorsExcept(self, number, nth):
        vectors = []
        for i in range(10):
            for j in range(1, 11):
                if i == number and j == nth:
                    continue
                name = f'baseProjetOCR/{i}_{j}.png'
                matrixImg = imgToBinaryMatrix(io.imread(name))
                matrixImg = self.convertMatrixTo4x4(matrixImg)
                vectors.append((name, self.convert4x4ToVector(matrixImg)))
        return vectors



    def getFileNameCorrespondingTo(self,img, comparedVectors):
        matrixImg = imgToBinaryMatrix(img)
        meanKernel = np.full((3, 3), 1.0/9)
        matrixImg = ndi.correlate(matrixImg, meanKernel)
        matrixImg = self.convertMatrixTo4x4(matrixImg)
        sourceVector = self.convert4x4ToVector(matrixImg)

        allDistancesZoning = [euclideanDistance(sourceVector, comparedVectors[i][1]) for i in range(len(comparedVectors))]
        
            

        indexOfMinDistance = allDistancesZoning.index(min(allDistancesZoning))
        return (comparedVectors[indexOfMinDistance][0], allDistancesZoning)
    
    def getZoningConfusionMatrix(self):
        self.initScoreMatrices()
        mat = np.zeros(shape=(10, 10), dtype=np.uint8)
        for i in range(10):
            for j in range(1,11):

                vectors = self.getComparedVectorsExcept(i, j)
                fnAndScores = self.getFileNameCorrespondingTo(io.imread(f'baseProjetOCR/{i}_{j}.png'), vectors)
                index = i * 10 + (j-1)
                fnAndScores[1].insert(index, 0)
                scoreMat = np.reshape(fnAndScores[1], (10,10))
                self.scoreMatrices[f'{i}-{j}'] = scoreMat
                n = getNumberOfFileName(fnAndScores[0])
                mat[i][n] += 1
        return mat

    def guessFromFilename(self, file):
        i = file.split('/')[1][0]
        j = file.split('/')[1][2]
        vectors = self.getComparedVectorsExcept(i, j)
        fnAndScores = self.getFileNameCorrespondingTo(io.imread(f'baseProjetOCR/{i}_{j}.png'), vectors)
        return getNumberOfFileName(fnAndScores[0])
        
    
    def getScoreMatrix(self, nb, nth):
        return self.scoreMatrices[f'{nb}-{nth}']
        
def imgToBinaryMatrix(image, sizeX=20, sizeY=20):
    image = resize(image, (sizeX, sizeY), preserve_range=True).astype('uint8')
    threshold = 150
    # make all pixels < threshold black
    binarized = 1.0 * (image > threshold)

    # convert 3d array to 2d array
    return binarized[:, :, 0]


def getNumberOfFileName(s):
    return int(s.split("/", 1)[1][0])

def euclideanDistance(v1, v2):
    distance = 0
    for i in range(len(v1)):
        distance += math.pow((v1[i] - v2[i]), 2)
    return math.sqrt(distance)



def main():
    print("\n" + "="*50 + "\n")
    tprint("OCR")
    print("\n CHARASSON Gabin, KACHMAR Ayman, MARTIN Hugo\n")
    print("="*50 + "\n")
    OCR()
    
    print("--- Profil HORIZONTAL ---", end="\n\n")
    ph = ProfilH()
    ph_cm = ph.getProfileVHConfusionMatrix()
    print(np.matrix(ph_cm))
    print("\n\n--- FIN PROFIL HORIZONTAL ---", end="\n\n\n")

    print("--- ZONING ---", end="\n\n")
    zn = Zoning()
    zn_cm = zn.getZoningConfusionMatrix()
    print(zn_cm)
    print("\n\n--- FIN ZONING ---", end="\n\n\n")

    print("--- KNN ---", end="\n\n")
    knn = KNN()
    knn_cm = knn.getKnnConfusionMatrix()
    print(knn_cm)
    print("\n\n--- FIN KNN ---", end="\n\n\n")

if __name__ == "__main__":
    main()