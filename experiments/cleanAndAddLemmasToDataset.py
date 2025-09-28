import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc

PATH_TO_DF = "../data/dataset.csv"

# Скачайте один раз, если еще не делали:
nltk.download('stopwords')
nltk.download('punkt')

# Список всех месяцев по-русски
months = [
    'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
    'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря'
]
russian_stopwords = set(stopwords.words('russian'))
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

def clean_text(text):
    if pd.isnull(text):
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
    text = re.sub(r'\s+', ' ', text).strip()    
    return text

def tokenize(text):
    # NLTK токенизация, либо просто text.split()
    return text.split() if text else []

def lemmatize_natasha(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    lemmas = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemmas.append(token.lemma)
    return ' '.join(lemmas)

df = pd.read_csv(PATH_TO_DF)

# Фильтрация непустых строк
df = df[df['text'].notnull() & (df['text'].str.strip() != '')]

# Исключение приветствий
df = df[~df['text'].isin(['Добрый день!', 'Добрый день.', 'Добрый день', 'Добрый день,' 
                          'Доброе утро', 'Доброе утро,', 'Доброе утро.', 'Доброе утро!',
                          'Добрый вечер!', 'Добрый вечер.', 'Добрый вечер', 'Добрый вечер,' 
                           'Здравствуйте.', 'Здравствуйте!','Здравствуйте', 'Здравствуйте,'])]

# 4. лемманизация
df['textLemmas'] = df['text'].apply(lambda x: lemmatize_natasha(clean_text(x)))

print('Кол-во строк:', df.shape[0])
print(df[['text', 'textLemmas']].iloc[0:30])

# Сохраняем в файл
df.to_csv(PATH_TO_DF, index=False)
print(f'Result saved to file: {PATH_TO_DF}')
