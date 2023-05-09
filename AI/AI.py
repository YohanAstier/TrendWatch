import pandas as pd
import pickle
from enum import Enum
import nltk
import string 

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

class Mode(Enum):
    NONE = 0
    POSORNEG = 1
    EMOTIONCLASS = 2

class AI():
    def __init__(self, model_path = None, mode = Mode.NONE):
        self.load_model(model_path, mode)
        
    
    def train_AI():
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
        model = model.fit(x_train, y_train)

        accuracy = model.score(x_test, y_test)
        print(f"Précision du modèle: {str('%.2f' % (accuracy * 100))}%")

        pipeline_file = open("/models/bad_or_good_ai.pkl", "wb")
        pickle.dump(model, pipeline_file)
        pipeline_file.close()

        return model

    def train_Emotion_AI():
        df = pd.read_csv("datasets/emotion_dataset_raw.csv")
        dir(nfx)
        df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
        df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)
        Xfeatures = df['Clean_Text']
        ylabels = df['Emotion']
        x_train, x_test, y_train, y_test = train_test_split(Xfeatures, ylabels, test_size=0.3, random_state=42)
        
        pipe_lr = Pipeline(steps=[('cv', CountVectorizer()), ('lr', LogisticRegression(max_iter=1_000_000))])
        pipe_lr.fit(x_train, y_train)
        accuracy = pipe_lr.score(x_test, y_test)
        print(f"Précision du modèle: {str('%.2f' % (accuracy * 100))}%")


        pipeline_file = open("models/class_emotion_ai.pkl", "wb")
        pickle.dump(pipe_lr, pipeline_file)
        pipeline_file.close()

    """Load another AI model"""
    def load_model(self, model_path, target_mode):
        self.mode = target_mode            
        if target_mode == Mode.NONE :
            self.model = None

        else :
            self.model = pickle.load(open(model_path, 'rb'))

    """Predict the value of a message"""
    def predict_message(self, message):
        if self.mode != Mode.NONE: 
            return self.model.predict_proba([message])
        return []
    
    """Return the names of the class of the AI"""
    def emotions_label(self):
        if self.mode != Mode.NONE:
            return list(self.model.classes_)
        return []
    
    def preprocess(sentence):
        stop_words = nltk.corpus.stopwords.words('english') + list(string.punctuation)
        words = nltk.word_tokenize(sentence.lower())
        return [word for word in words if word not in stop_words]
    
    def word_cloud(messages, words_number):
        clean_messages = AI.preprocess(" ".join(messages))
        frequency = nltk.FreqDist(clean_messages).most_common(words_number)
        return [duo[0] for duo in frequency]




