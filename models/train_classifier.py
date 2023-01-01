# import libraries
import sys
import pandas as pd
import numpy as np
import re
import nltk
nltk.download(['punkt', 'wordnet','omw-1.4'])
import warnings
warnings.simplefilter('ignore')
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report,confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score

def load_data(database_filepath):
    '''
    Returns: X, y, category_names
    
    input:
          database_filename: path of database filename for sqlite database   
          
    output:
           X: pandas dataframe containing messages as input
           y: pandas dataframe containing the response target as output
           category_names: pandas Series with classes of target response y
    ''' 
    #load data from database setting up the engine connection sql 
    engine = create_engine('sqlite:///'+database_filepath)
    #load sql table using pandas 
    df = pd.read_sql_table('DisasterResponse_Database', engine)
    #set the design matrix X using messages as input
    X = df['message']
    #set the response target y using it as output
    y = df[df.columns[4:]]    
    #outputs the classes of target response y
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    '''
    Returns: clean_tokens
    
    input:
          text: raw text as string   
          
    output:
           clean_tokens: raw text cleaned    
    ''' 
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Returns: model
    
    input:
          None  
          
    output:
           model: model pipeline (pipeline object class)    
    ''' 
    #Set the model based on the pipeline class object
    model = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),
            ('tfidf',TfidfTransformer()),
            ('clf',MultiOutputClassifier(RandomForestClassifier()))
            ])
    
    #specify parameters for grid search
    parameters = {
        'clf__estimator__n_estimators': [10,20],
         'clf__estimator__max_depth':[4,5]
}

    #set some standard metrics of multi-classification based on make_score() function
    scorers = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average = 'macro'),
           'recall': make_scorer(recall_score, average = 'macro'),
           'f1': make_scorer(f1_score, average = 'macro')}

    #set model with gridsearch method based on crossvalidation (default=3) and scoring f1-metric
    cv = GridSearchCV(estimator=model, param_grid=parameters, cv=3, scoring=scorers, verbose=3,refit='f1')
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Returns: None
    
    input:
          model: model using pipeline previously  
          X_test: numpy array containing unseen messages as input in the test set
          Y_test: numpy array containing the response target categories in the test set of new messages
          category_names: pandas Series with classes of target response y

          
    output:
           X: pandas dataframe containing messages as input
           y: pandas dataframe containing the response target as output
           category_names: pandas Series with classes of target response y
    '''     
    #compute predictions of the target variable y using the test set's design matrix X_test
    Y_pred = model.predict(X_test)
    #iterating through the columns and calling sklearn's classification_report on each y_pred and y_test
    #iterating through the columns and calling sklearn's classification_report on each y_pred and y_test
    #outputs in dictionary the classes of target response y and the corresponding values
    reversefactor = dict(zip(range(35),Y_test.columns))
    #y_test will be converted into y_true for purposes of illustration metrics
    Y_true = np.vectorize(reversefactor.get)(Y_test)
    #y_pred will be converted into y_est for purposes of illustration metrics
    Y_est = np.vectorize(reversefactor.get)(Y_pred)
    for i in range(35):
        print('target response class',Y_test.columns[i], ':')
        print(classification_report(np.array(Y_test)[:,i], Y_pred[:,i],target_names=category_names), '**********************************************************')
        print('Confusion Matrix : \n' + str(pd.crosstab(np.array(Y_true)[:,i],Y_est[:,i], rownames=['Actual Classes'], colnames=['Predicted Classes'])))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:        
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        database_filepath='data/DisasterResponse_Database.db'
        model_filepath='models/my_classifier.pkl'
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.3, random_state=2022)
        
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