import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1
        )
        self.model = LogisticRegression()

    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, text):
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

    def save(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump((self.vectorizer, self.model), f)

    def load(self, path="model.pkl"):
        with open(path, "rb") as f:
            self.vectorizer, self.model = pickle.load(f)