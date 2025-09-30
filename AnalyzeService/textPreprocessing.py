import re
import nltk
from nltk.corpus import stopwords
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc

def clean_text(text: str):
    months = [
        'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
        'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря'
    ]
    russian_stopwords = set(stopwords.words('russian'))

    if ((text == None)or(len(text) == 0)):
        return ""
    text = text.lower()  # переводим в нижний регистр
    # Формат дд.мм.гггг и дд.мм.гг
    text = re.sub(r'\b\d{2}\.\d{2}\.\d{4}\b', '', text)
    text = re.sub(r'\b\d{2}\.\d{2}\.\d{2}\b', '', text)
    # Формат дд месяца и дд месяца YYYY
    for month in months:
    # дд месяца YYYY
        text = re.sub(rf'\b\d{{1,2}} {month} \d{{4}}\b', '', text)
    # дд месяца
        text = re.sub(rf'\b\d{{1,2}} {month}\b', '', text)

    # Удаляем знаки препинания (кроме букв, цифр и пробелов)
        text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)

    # Удаление стоп-слов
    words = [w for w in text.split() if w.lower() not in russian_stopwords]
    text = ' '.join(words)
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str):
    # NLTK токенизация, либо просто text.split()
    return text.split() if text else []

def lemmatize_natasha(text: str):
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    morph_vocab = MorphVocab()

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    lemmas = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemmas.append(token.lemma)
    return ' '.join(lemmas)

def textToLemmas(text: str):
    # Скачайте один раз, если еще не делали:
    nltk.download('stopwords')
    nltk.download('punkt')

    text = text.lower()

    # remove greetings
    greetings = ['добрый день!', 'добрый день.', 'добрый день', 'добрый день,' 
                          'доброе утро', 'доброе утро,', 'доброе утро.', 'доброе утро!',
                          'добрый вечер!', 'добрый вечер.', 'добрый вечер', 'добрый вечер,' 
                           'здравствуйте.', 'здравствуйте!','здравствуйте', 'здравствуйте,']
    for g in greetings:
        text = text.replace(g, "")

    return lemmatize_natasha(clean_text(text))