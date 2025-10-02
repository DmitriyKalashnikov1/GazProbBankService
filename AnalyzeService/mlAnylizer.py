import numpy as np
import re
from textPreprocessing import textToLemmas
from textClusterisation import CpuClusterizator, GpuClusterizator
from emotionDetector import EmotionDetector

class MlTextAnalyzer():

    clusterizator = None
    detector = None

    def __init__(self):
        self.clusterizator = CpuClusterizator(pathToClusterizer='../data/mlClusterizatorLemmText.joblib')
        self.detector = EmotionDetector()

    def process(self, text:str):
        parts = re.split(r'[.!?:;]+', text)
        lemmParts = [textToLemmas(part) for part in parts]

        bows = [self.clusterizator.clusterizeTextFragmentToBagOfWords(lPart) for lPart in lemmParts]
        em_bows = []

        for f in range(len(bows)):
            em_bows.append(self.detector.get_emotional_bow(parts[f], bows[f][0]))

        finalBow = np.any(np.array(bows), axis=0).astype(np.integer)
        finalEmBow = np.array(em_bows).mean(axis=0)
        themes = self.clusterizator.getThemes(finalBow)

        return {"clusters": finalBow,
                "emotions": finalEmBow,
                "themes": themes}


if __name__ == "__main__":
    import pandas as pd

    textAnylizer = MlTextAnalyzer()

    df = pd.read_csv("../data/dataset.csv")

    texts = df["text"].dropna().to_list()

    print(texts[32])
    print(textAnylizer.process(texts[32]))

