import re
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


class DocumentVectorizer:
    def __init__(self,max_features):
        self.max_features = max_features
        self.vetorizador = TfidfVectorizer(max_features=max_features)    
        nltk.download("punkt")


    def preprocess_text(self, text):
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        # Remove mentions and hashtags
        text = re.sub(r"\@\w+|\#", "", text)
        # Remove punctuation and numbers
        text = re.sub(r"[^A-Za-zÀ-ÿ\s]", "", text)
        # Convert to lowercase
        text = text.lower()
        
        palavras = text.split()
        palavras = [palavra for palavra in palavras if palavra not in stop_words]
        palavras = " ".join(palavras)
        return palavras

    def fit_transform(self, train_text):
        cleaned_text = train_text.apply(self.preprocess_text)        
        X_train_tfid = self.vetorizador.fit_transform(cleaned_text)
        return X_train_tfid

    def transform(self, text):
        cleaned_text = text.apply(self.preprocess_text) if hasattr(text, 'apply') else [self.preprocess_text(t) for t in text]
        X_test_tfid = self.vetorizador.transform(cleaned_text)
        return X_test_tfid
   
    
    def save_vectorizer(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.vetorizador, f)
        print(f" Vetorizador salvo em: {path}")
    
    def load_vectorizer(self, path):
        with open(path, "rb") as f:
            self.vetorizador = pickle.load(f)
        print(f" Vetorizador carregado de: {path}")
