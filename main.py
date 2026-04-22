import pandas as pd
from model import TextClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Wczytaj dane
data = pd.read_csv("data.csv")

texts = data["text"]
labels = data["label"]

# Podział na train/test
texts_train, texts_test, labels_train, labels_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Trening
classifier = TextClassifier()
classifier.train(texts_train, labels_train)

# 🔍 TEST DIAGNOSTYCZNY (TU DODAJEMY)
print("Wagi modelu:")
print(classifier.model.coef_)

# (opcjonalnie – lepszy wgląd)
print("\nSłowa i ich wagi:")
feature_names = classifier.vectorizer.get_feature_names_out()
coefficients = classifier.model.coef_[0]

for word, coef in zip(feature_names, coefficients):
    print(word, coef)

# Ewaluacja
predictions = [classifier.predict(t) for t in texts_test]
print("\nAccuracy:", accuracy_score(labels_test, predictions))

# Interfejs użytkownika
print("\nModel wytrenowany! Wpisz zdanie (lub 'exit'):\n")

while True:
    user_input = input("> ")
    if user_input.lower() == "exit":
        break

    prediction = classifier.predict(user_input)
    print(f"Wynik: {prediction}\n")