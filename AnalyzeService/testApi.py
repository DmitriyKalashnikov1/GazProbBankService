import requests
import json
import pandas as pd

df = pd.read_csv("../data/dataset.csv")

texts = df["text"].dropna().to_list()

test = texts[32]

print(test)

jsonTest = json.dumps([{
    "id": 1,
    "text": test
}])


resp = requests.post("http://0.0.0.0:8000/predict/", data=jsonTest, headers={"Content-type": "application/json"})

print(resp.status_code)
print(resp.text)
