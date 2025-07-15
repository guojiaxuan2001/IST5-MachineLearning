from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

(X_train_int, y_train), (X_test_int, y_test) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

def decode_review(text_int):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in text_int])

X_train_text = [decode_review(text) for text in X_train_int]
X_test_text = [decode_review(text) for text in X_test_int]

print("Decoding Complete")

print("Vectorizing...")

vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

print("Vectorizing Complete")

print("Training...")
clf = LinearSVC(random_state=42)
clf.fit(X_train_tfidf, y_train)
print("Training Complete")

y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy on Validation Set: {accuracy:.4f}")

