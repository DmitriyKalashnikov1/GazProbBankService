import numpy as np
import joblib
import torch
from hummingbird.ml import load

class CpuClusterizator():
    categoriesNames = ["Привилегии и сервис", "Карты, переводы и операции", "Акции и бонусы",
                   "Поддержка и связь", "Доставка и встречи", "Качество обслуживания",
                   "Счета, вклады и закрытие", "Коммуникация и документы", "Прочие",
                   "Пусто"]
    categoriesClusters = [[1,10,32], [2,4,9,12,17,21,22,30], [3,20,31,34], [5,15,28,33],
                      [16,25], [6,7,8,26,27], [18,19,29], [13,23], [11,24], [14]]
    idfVectorizer = None
    mlClusterizator = None

    def __init__(self, pathToClusterizer:str):
        self.mlClusterizator = joblib.load(pathToClusterizer)

    def clusterizeTextFragmentToBagOfWords(self, textFargment:str):
        bow = np.zeros(shape=(1, len(self.categoriesClusters)))

        cluster = self.mlClusterizator.predict([textFargment])[0]

        for index, list in enumerate(self.categoriesClusters):
            if cluster in list:
                bow[0, index] = 1
        return np.array(bow, dtype=np.integer)

    def clusterizeListOfTextsToBagOfWords(self, listsOfTexts: list):
        outVectors = []

        clusters = self.mlClusterizator.predict(listsOfTexts)

        for f in range(len(listsOfTexts)):
            bow = np.zeros(shape=(1, len(self.categoriesClusters)))
            cluster = clusters[f]

            for index, list in enumerate(self.categoriesClusters):
                if cluster in list:
                    bow[0,index] = 1
            outVectors.append(bow)

        return np.any(np.array(outVectors), axis=0).astype(np.integer)

    def getThemes(self, bow: np.array):
        output = []
        for f in range(bow.shape[0]):
            for index, c in enumerate(bow[f]):
                if c:
                    output.append(self.categoriesNames[index])
            return output


class GpuClusterizator():
    categoriesNames = ["Привилегии и сервис", "Карты, переводы и операции", "Акции и бонусы",
                   "Поддержка и связь", "Доставка и встречи", "Качество обслуживания",
                   "Счета, вклады и закрытие", "Коммуникация и документы", "Прочие",
                   "Пусто"]
    categoriesClusters = [[1,10,32], [2,4,9,12,17,21,22,30], [3,20,31,34], [5,15,28,33],
                      [16,25], [6,7,8,26,27], [18,19,29], [13,23], [11,24], [14]]
    idfVectorizer = None
    mlClusterizator = None

    def __init__(self, pathToClusterizer:str, pathToVectorizer:str):
        self.idfVectorizer = joblib.load(pathToVectorizer)
        self.mlClusterizator = load(pathToClusterizer, override_flag=True)
        if (torch.cuda.is_available()):
            self.mlClusterizator.to("cuda")

    def clusterizeTextFragmentToBagOfWords(self, textFargment:str):
        bow = np.zeros(shape=(1, len(self.categoriesClusters)))

        cluster = self.mlClusterizator.predict(self.idfVectorizer.transform([textFargment]).todense().A)[0]

        for index, list in enumerate(self.categoriesClusters):
            if cluster in list:
                bow[0, index] = 1

        return np.array(bow, dtype=np.integer)

    def clusterizeListOfTextsToBagOfWords(self, listsOfTexts: list):
        outVectors = []

        clusters = self.mlClusterizator.predict(self.idfVectorizer.transform(listsOfTexts).todense().A)

        for f in range(len(listsOfTexts)):
            bow = np.zeros(shape=(1, len(self.categoriesClusters)))
            cluster = clusters[f]

            for index, list in enumerate(self.categoriesClusters):
                if cluster in list:
                    bow[0,index] = 1
            outVectors.append(bow)

        return np.any(np.array(outVectors), axis=0).astype(np.integer)

    def getThemes(self, bow: np.array):
        output = []
        for f in range(bow.shape[0]):
            for index, c in enumerate(bow[f]):
                if c:
                    output.append(self.categoriesNames[index])
            return output
