import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

categoriesNames = ["Привилегии и сервис", "Карты, переводы и операции", "Акции и бонусы",
                   "Поддержка и связь", "Доставка и встречи", "Качество обслуживания",
                   "Счета, вклады и закрытие", "Коммуникация и документы", "Прочие",
                   "Пусто"]
categoriesClusters = [[1,10,32], [2,4,9,12,17,21,22,30], [3,20,31,34], [5,15,28,33],
                      [16,25], [6,7,8,26,27], [18,19,29], [13,23], [11,24], [14]]


def clusterizeTextToBagOfWords(listsOfTexts: list, pathToClusterizer:str):
    outVectors = []

    mlClusterizator = joblib.load(pathToClusterizer)

    clusters = mlClusterizator.predict(listsOfTexts)
    del mlClusterizator
    for f in range(len(listsOfTexts)):
        bow = np.zeros(shape=(1, len(categoriesClusters)))
        cluster = clusters[f]

        for index, list in enumerate(categoriesClusters):
            if cluster in list:
                bow[0,index] = 1
        outVectors.append(bow)

    return np.any(np.array(outVectors), axis=0).astype(np.integer)

def getThemes(bow: np.array):
    output = []
    for f in range(bow.shape[0]):
        for index, c in enumerate(bow[f]):
            if c:
                output.append(categoriesNames[index])
        return output





