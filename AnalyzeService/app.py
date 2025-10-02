from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import torch
from typing import Union, List

from mlAnylizer import MlTextAnalyzer

analyzer = MlTextAnalyzer()


class Item(BaseModel):
    id: int
    text: str

app = FastAPI()

@app.post("/predict/")
async def predict_json(input_data: List[Item]):
    try:

        results = []
        for item in input_data:
            _id = item.id
            _text = item.text
            if _text is None or _id is None:
                results.append({"id": _id,
                                "topics": "Error"})
                continue
            res = analyzer.process(_text)
            # Возвращаем только нужные поля
            results.append({
                "id": _id,
                "topics": res["themes"],                # переименовано с themes
                "sentiments": res["emotions"].tolist()           # переименовано с emotions
            })
        return results
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/info/")
async def info():
    return {
        "isCudaOn": torch.cuda.is_available(),
        "categoricalNames": analyzer.clusterizator.categoriesNames
    }