import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    returns df dataframe containing merged data from disaster_messages.csv and disaster_categories.csv
    
    input:
          messages_filepath: path of disaster_messages.csv file
          categories_filepath: path of disaster_categories.csv file
           
    output:
           df: pandas dataframe containing combined data of disaster_messages.csv and disaster_categories.csv
    ''' 
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets, python understands that the reference key column is the id to 
    df =pd.merge(messages,categories)
    df.head()
    
    #Split values in the categories column based on the ; delimeter in order to    separate columns and pick up the first row
    categories=df['categories'].str.split(';',expand=True)
    # create a dataframe of the 36 individual category columns
    
    #Split values in the categories column based on the ; delimeter in order to separate columns and pick up the first row
    category_colnames=df['categories'].str.split(';')[0]
    #Convert the list into a pandas series in order to manipulate with str.split()  metjpd
    category_colnames=pd.Series(category_colnames)
    #Split the categories column based on the - delimeter
    category_colnames=category_colnames.str.split('-',expand=True)
    
    #Pick up the column 0 that contains all the appropriate names of categories, column 1 is useless
    category_colnames=category_colnames[0]
    categories.columns = category_colnames
    
    for column in categories:        
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
        
    #category disaster related converted to binary
    categories['related'] = categories['related'].astype('str').str.replace('2', '1')
    categories['related'] = categories['related'].astype('int')
        
    # drop the original categories column from `df`
    df.drop('categories', axis=1,inplace=True)    
    
    # concatenate the original dataframe with the new `categories` dataframe
    df =pd.concat([df,categories],axis=1)
    
    return df

def clean_data(df):
    '''
    returns df dataframe in cleaned version 
    
    input:
          df: pandas dataframe of merged data 
           
    output:
           df: pandas dataframe of merged data with clean data
    ''' 
    
    # check number of duplicates
    df.duplicated().sum()
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    # check number of duplicates
    df.duplicated().sum()
    
    #Remove child_alone class (all 0's)
    df.drop('child_alone', axis=1,inplace=True)
    return df
    


def save_data(df, database_filename):
    '''
    Returns: None
    
    input:
          df: pandas dataframe of merged data with clean data
          path of disaster_categories.csv file
          database_filename: path of database filename for sqlite database   
          
    output:
           No output
    ''' 

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse_Database', engine,if_exists = 'replace', index=False)  
      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        messages_filepath='data/disaster_messages.csv'
        categories_filepath='data/disaster_categories.csv'
        database_filepath='data/DisasterResponse_Database.db'

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()