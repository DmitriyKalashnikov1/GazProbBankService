import pandas as pd
import re
from tqdm import tqdm
from textPreprocessing import textToLemmas
from textClusterisation import CpuClusterizator
from time import time

tC = CpuClusterizator(pathToClusterizer='../data/mlClusterizatorLemmText.joblib')

def textClustering(text:str, vectorOrName: bool):
    global tC
    # split text by part
    parts = re.split(r'[.!?:;]+', text)
    lemmParts = [textToLemmas(part) for part in parts]

    if (vectorOrName):
        return tC.clusterizeListOfTextsToBagOfWords(lemmParts)[0]
    else:
        bow = tC.clusterizeListOfTextsToBagOfWords(lemmParts)
        return tC.getThemes(bow)


df = pd.read_csv("../data/dataset.csv")

texts = df["text"].dropna().to_list()

print(texts[32])
t0 = time()
print(textClustering(texts[32], True))
print(textClustering(texts[32], False))
print(f"time of one clusterisation: {(time() - t0)/2} s")

print("Adding clusterVector to dataset...")
df['clusterVector'] = tqdm([textClustering(x, True) for x in df["text"]])
print("Done")
print("Adding textThemes to dataset...")
df['textThemes'] = tqdm([textClustering(x, False) for x in df["text"]])
df.to_csv("../data/dataset.csv")
print("Done")