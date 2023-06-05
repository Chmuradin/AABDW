import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(path):
    data = pd.read_csv(path)
    return data


# TF-IDF
def tfidf(data: pd.DataFrame, text_columns):
    vectorizer = TfidfVectorizer()
    text = data[text_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    data = data.drop(columns=text_columns)
    vectorizer.fit(text)
    text_features = vectorizer.transform(text).toarray()
    data = pd.concat([data, pd.DataFrame(text_features)], axis=1)
    return data


if __name__ == '__main__':
    train_path = "data/train_processed.csv"
    test_path = "data/test_processed.csv"
    text_col = ["property_name", "property_summary", "property_space", "property_desc", "property_neighborhood",
                "property_notes", "property_transit", "property_access", "property_interaction", "property_rules"]

    train_data = load_data(train_path)
    test_data = load_data(test_path)
    train_data = tfidf(train_data, text_col)
    test_data = tfidf(test_data, text_col)

    train_data.to_csv("data/train_featured1.csv", index=False)
    test_data.to_csv("data/test_featured1.csv", index=False)