import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from string import punctuation
from textblob import TextBlob, Word
import nltk
from nltk.stem import PorterStemmer
import re
import html
from urllib import parse
import requests
import json
from langdetect import detect
from langdetect import DetectorFactory


GOOGLE_TRANSLATE_URL = 'https://translate.google.be/m?q=%s&tl=%s&sl=%s'


def days_calculator(text):
    days = dict({"today": 0, "yesterday": 1, "d": 1, "w": 7, "m": 30, "never": 1000})
    tl = text.split()
    if len(tl) == 1:
        return days[tl[0]]
    else:
        if tl[0] == "a":
            return days[tl[1][0]]
        else:
            return int(tl[0]) * days[tl[1][0]]


def translate(text, to_language="en", text_language="auto"):
    text = parse.quote(text)
    url = GOOGLE_TRANSLATE_URL % (text, to_language, text_language)
    response = requests.get(url)
    data = response.text
    expr = r'(?s)class="(?:t0|result-container)">(.*?)<'
    result = re.findall(expr, data)
    if len(result) == 0:
        return " "

    return html.unescape(result[0])


def translate_to_en(text):
    if len(text) <= 1:
        return " "
    else:
        if detect(text) == "en":
            return text
    return translate(text, to_language="en")


def process_text(data, text_col):
    stop = stopwords.words("english")
    DetectorFactory.seed = 0
    st = PorterStemmer()
    punc = punctuation+"★✩☀️"
    for tc in text_col:
        #  1. Lowercase
        # data[tc] = data[tc].apply(lambda x: " ".join(x.lower() for x in x.split()))
        #  2. Removing punctuation
        # data[tc] = data[tc].str.replace(r"[{}]+".format(punc), "")
        #  3. Translating into English
        # data[tc] = data[tc].apply(lambda x: translate(x))
        for idx in range(6000, len(data)):
            trans = translate(data.loc[idx, tc])
            print(idx, type(eval(trans)))
            data.loc[idx, tc] = eval(trans)
    # for tc in text_col:
    #     #  4. Removing Stop Words
    #     data[tc] = data[tc].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    #     #  5. Correcting spelling
    #     data[tc] = data[tc].apply(lambda x: str(TextBlob(x).correct()))
    #     #  6. Tokenizing text
    #     data[tc] = data[tc].apply(lambda x: nltk.word_tokenize(x))
    #     #  7. Stemming
    #     data[tc] = data[tc].apply(lambda x: " ".join([st.stem(word) for word in x]))
    #     #  8. Lemmatizing
    #     data[tc] = data[tc].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    #     print("{} is processed".format(tc))

    return data
