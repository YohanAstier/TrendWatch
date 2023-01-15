import pandas as pd
import joblib

# Load Data Viz Pkgs
import seaborn as sns

# Load Text Cleaning Pkgs
import neattext.functions as nfx

# Build Pipeline
from sklearn.pipeline import Pipeline

# Load ML Pkgs
# Estimators
from sklearn.linear_model import LogisticRegression
# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer


def Train_AI():
    df = pd.read_csv("datasets/dataset.csv", delimiter=';')
    dir(nfx)
    print(df.columns)
    df['text'] = df['text'].fillna("").astype('string')
    df['Clean_Text'] = df['text'].apply(nfx.remove_userhandles)
    df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

    vectorizer = TfidfVectorizer()
    Xfeatures = vectorizer.fit_transform(df['Clean_Text'].values.astype('str'))
    ylabels = df['emotion']

    x_train, x_test, y_train, y_test = train_test_split(Xfeatures, ylabels, test_size=0.1, random_state=42)

    model = LogisticRegression(max_iter=1_000_000)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = model.score(x_test, y_test)
    print(f"Précision du modèle: {str('%.2f' % (accuracy * 100))}%")


Train_AI()
