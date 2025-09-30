import pandas as pd
import re
from tqdm import tqdm
tqdm.pandas()
from pandarallel import pandarallel
from textPreprocessing import textToLemmas
from textClusterisation import clusterizeTextToBagOfWords, getThemes
from time import time

def textClustering(text:str, vectorOrName: bool):
    # split text by part
    parts = re.split(r'[.!?:;]+', text)
    lemmParts = [textToLemmas(part) for part in parts]

    if (vectorOrName):
        return clusterizeTextToBagOfWords(lemmParts,'../data/mlClusterizatorLemmText.joblib')[0]
    else:
        bow = clusterizeTextToBagOfWords(lemmParts,'../data/mlClusterizatorLemmText.joblib')
        return getThemes(bow)

pandarallel.initialize(nb_workers=1, use_memory_fs=True, progress_bar=True)

df = pd.read_csv("../data/dataset.csv")

texts = df["text"].dropna().to_list()

print(texts[32])
t0 = time()
print(textClustering(texts[32], True))
print(textClustering(texts[32], False))
print(f"time of one clusterisation: {(time() - t0)/2} s")

print("Adding information to dataset...")
df['clusterVector'] = df["text"].parallel_apply(lambda x: textClustering(x, True))
#df['textThemes'] = df['text'].apply(lambda x: textClustering(x, False))
df.to_csv("../data/dataset.csv")
print("Done")