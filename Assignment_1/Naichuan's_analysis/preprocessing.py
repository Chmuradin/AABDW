import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
from textblob import TextBlob, Word
import ast
import nltk
from nltk.stem import PorterStemmer
import socket
import emoji
from urllib3.connection import HTTPConnection
from langdetect import detect
from langdetect import DetectorFactory

from utils import days_calculator, translate

pd.set_option("display.max_columns", None)


# pd.set_option("display.max_rows", None)


def load_data(path):
    data = pd.read_csv(path)
    return data


def reviews(data):  # Only return "reviews" features
    cols = data.columns.values
    col_reviews = [col for col in cols if col.startswith("reviews")]

    data["reviews_per_month"] = data["reviews_per_month"].fillna(value=0)
    data.loc[:, col_reviews] = data.loc[:, col_reviews].fillna(method="ffill")

    return data


def property_fill_missing(data):
    drops = ["property_sqfeet"]
    data.drop(columns=drops, inplace=True)  # Missing rate = 97.5%

    # fill text data with " "
    data.iloc[:, 1:11] = data.iloc[:, 1:11].fillna(value=" ")

    # fill numerical data with 1
    fill = ["property_bathrooms", "property_bedrooms", "property_beds"]
    data.loc[:, fill] = data.loc[:, fill].fillna(value=1)

    return data


def property_encode_cat(train, test):
    # encode categorical data
    cal = ["property_type", "property_room_type", "property_bed_type"]
    enc = OneHotEncoder(sparse_output=False)
    enc.fit(train.loc[:, cal])
    train_cal = pd.DataFrame(enc.transform(train.loc[:, cal]))
    train.drop(columns=cal, inplace=True)
    train = pd.concat([train, train_cal], axis=1)
    test_cal = pd.DataFrame(enc.transform(test.loc[:, cal]))
    test.drop(columns=cal, inplace=True)
    test = pd.concat([test, test_cal], axis=1)

    # process property last updated
    train["property_last_updated"] = train["property_last_updated"].apply(lambda x: days_calculator(x))
    test["property_last_updated"] = test["property_last_updated"].apply(lambda x: days_calculator(x))

    return train, test


def property_translate_text(data):
    text_col = ["property_name", "property_summary", "property_space", "property_desc", "property_neighborhood",
                "property_notes", "property_transit", "property_access", "property_interaction", "property_rules"]
    # for tc in text_col:
    #     data[tc] = data[tc].apply(lambda x: re.sub(r'[^\w\s]', '', x))\
    for idx in range(len(data)):
        try:
            trans = translate(str(list(data.loc[idx, text_col].values)))
            ltrans = ast.literal_eval(trans)
            atrans = np.array(ltrans)
            data.loc[idx, text_col] = atrans
        except SyntaxError:
            print(idx)
            with open("data/manul.txt", "a+") as f:
                f.write(str(idx) + "\n")
        except ValueError:
            print(idx)
            with open("data/manul.txt", "a+") as f:
                f.write(str(idx) + "\n")
    return data


def property_text(data):
    # text processing
    text_col = ["property_name", "property_summary", "property_space", "property_desc", "property_neighborhood",
                "property_notes", "property_transit", "property_access", "property_interaction", "property_rules"]
    stop = stopwords.words("english") + stopwords.words("french")
    st = PorterStemmer()
    for tc in text_col:
        #  1. Lowercase
        data[tc] = data[tc].str.lower()
        #  2. Removing emoji
        data[tc] = data[tc].apply(lambda x: emoji.demojize(str(x)))
        #  3. Removing Stop Words
        data[tc] = data[tc].apply(lambda x: " ".join(w for w in str(x).split() if w not in stop))
        #  4. Correcting spelling
        # data[tc] = data[tc].apply(lambda x: str(TextBlob(x).correct()))
        #  5. Tokenizing text
        data[tc] = data[tc].apply(lambda x: nltk.word_tokenize(x))
        #  6. Stemming
        data[tc] = data[tc].apply(lambda x: " ".join([st.stem(word) for word in x]))
        #  7. Lemmatizing
        data[tc] = data[tc].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        print("{} is processed".format(tc))

    return data


if __name__ == '__main__':
    train_path = "data/train_processed0.csv"
    test_path = "data/test_processed0.csv"

    HTTPConnection.default_socket_options = (
            HTTPConnection.default_socket_options + [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
        (socket.SOL_TCP, socket.TCP_KEEPALIVE, 45),
        (socket.SOL_TCP, socket.TCP_KEEPINTVL, 10),
        (socket.SOL_TCP, socket.TCP_KEEPCNT, 6)
    ]
    )

    train_data = load_data(train_path)
    test_data = load_data(test_path)

    train_data = reviews(train_data)
    test_data = reviews(test_data)

    train_data = property_fill_missing(train_data)
    test_data = property_fill_missing(test_data)
    train_data, test_data = property_encode_cat(train_data, test_data)
    #
    # train_data = property_translate_text(train_data)
    # test_data = property_translate_text(test_data)

    train_data = property_text(train_data)
    test_data = property_text(test_data)

    train_data.to_csv("data/train_processed.csv", index=False)
    test_data.to_csv("data/test_processed.csv", index=False)
