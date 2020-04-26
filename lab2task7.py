from string import punctuation
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import joblib


def process_message(message):
    message = [x for x in message if x not in punctuation]
    message = ''.join(message)
    message = [x for x in message.split() if x.lower() not in stopwords.words('english')]
    return message


def train():
    df = pd.read_csv('SPAM text message 20170820 - Data.csv')
    X_train, X_test, y_train, y_test = \
        train_test_split(df.Message, df.Category, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=process_message)),
        ('tfidf', TfidfTransformer()),
        ('bayes', MultinomialNB())
    ])
    pipeline.fit(X_train, y_train)
    print(pipeline.score(X_test, y_test))
    joblib.dump(pipeline, 'bayes_spam_filter.sav')


def classify(message):
    spam_filter = joblib.load('bayes_spam_filter.sav')
    message = pd.Series(message)
    print(f'Your message is {spam_filter.predict(message)[0]}')
