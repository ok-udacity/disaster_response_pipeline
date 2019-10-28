import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """ Loads data from database
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages_categories',engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names = Y.columns
    return X,Y,category_names
    
def tokenize(text):
    """ Normalizes case, removes punctuation, tokenizes text, lemmatizes and removes stop words
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def build_model():
    """ Builds model (finds best parameters using grid search)
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__learning_rate': [0.75, 0.87, 1.0]
    }  
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=10)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """ Compares predicted values vs. test values
    """
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred,columns=category_names)
    for column in category_names:
        print('Category: ', column)
        print(classification_report(Y_test[column],Y_pred_df[column]))

def save_model(model, model_filepath):
    """ Saves model to pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()