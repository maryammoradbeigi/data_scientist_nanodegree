import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''This function loads data sources as pandas dataframe
    Input: files path
    Output: Pandas dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='outer')
    return df


def clean_data(df):
    '''This function cleans the data
    by spliting the categories and droping duplicates.
    The target variables are binary. Therefore, I drop the row
    if the value is anything except 0 or 1
    
    Input: Pandas dataframe
    Output: Cleaned Pandas dataframe
    '''
    categories = df.categories.str.split(";", expand=True)
    
    raw_category_colnames = []
    for column in categories.columns:
        raw_category_colnames.append(categories.iloc[0, categories.columns.get_loc(column)])
    
    category_colnames = [i.split('-')[0] for i in raw_category_colnames]
    
    categories.columns = category_colnames
    
    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-").str[1]
    
    df = df.drop('categories', axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')
    
    df = df.drop_duplicates()
    df = df[df.related != 2]
    
    return df



def save_data(df, database_filename):
    '''This function saves the dataframe in sqllit database
    Input: dataframe, path
    Output: sqllit table
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')   


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

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
