import pandas as pd
import numpy as np
import sys
import re

# reading a csv file

def process_arguments(args):
    if len(args) < 2:
        sys.stderr.write("Usage: python reader.py <filename>\n")
        sys.exit(1)
    else:
        filename = args[1]
    return filename
    
def read_csv(filename):
    df = pd.read_csv(filename)
    return df


def clean_data(df):
    df['review'] = df['review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x))   #remove non-alphabets
    df['review'] = df['review'].apply(lambda x: x.lower())  #convert to lowercase
    df['review'] = df['review'].apply(lambda x: re.sub(r'br', '', x))  #remove pesky leftover <br> tags
    
    return df

def process(args):
    filename = process_arguments(args)
    df = read_csv(filename)
    df = clean_data(df)
    return df


