import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EmotionDetector:
    model_checkpoint = 'cointegrated/rubert-tiny-sentiment-balanced'
    tokenizer = None
    model = None
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint)

        if (torch.cuda.is_available()):
            self.model.cuda()

    def get_sentiment(self, text:str, return_type='score'):
        """ Calculate sentiment of a text. `return_type` can be 'label', 'score' or 'proba' """
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.model.device)
            proba = torch.nn.functional.softmax(self.model(**inputs).logits, dim=1).cpu().numpy()[0]
        if return_type == 'label':
            return self.model.config.id2label[proba.argmax()]
        elif return_type == 'score':
            return proba.dot([-1, 0, 1])
        elif return_type == 'all':
            return {"label": self.model.config.id2label[proba.argmax()], "score": proba.dot([-1, 0, 1]),
                    "proba": proba.tolist()}
        return proba

    def get_emotional_bow(self, textFragment:str, bow: np.array):
        score = self.get_sentiment(textFragment)
        em_bow = bow.copy().astype(np.float64)

        for f in range(len(bow)):
            if (bow[f] == 1):
                em_bow[f] = score
        return em_bow


if __name__ == "__main__":
    detector = EmotionDetector()

    text = 'Какая гадость эта ваша заливная рыба!'
    # classify the text
    print(detector.get_sentiment(text, 'label'))  # negative
    # score the text on the scale from -1 (very negative) to +1 (very positive)
    print(detector.get_sentiment(text, 'score'))  # -0.5894946306943893
    # calculate probabilities of all labels
    print(detector.get_sentiment(text, 'proba'))  # [0.7870447  0.4947824  0.19755007]


