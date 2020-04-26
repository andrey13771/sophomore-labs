import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


df = pd.read_csv('SPAM text message 20170820 - Data.csv')
X_train, X_test, y_train, y_test = train_test_split(df['Message'],
                                                    df['Category'], test_size=0.2, random_state=1)
pipe = Pipeline([
        ('count', CountVectorizer()),
        ('tfid', TfidfTransformer()),
        ('mnb', MultinomialNB())
    ])
pipe.fit(X_train, y_train)
print('Score: {}'.format(pipe.score(X_test, y_test)))
print(pipe.predict(['hello this is for Rachel, are you free tommorow at 5?'])[0])
