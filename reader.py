import pandas as pd
import numpy as np
import sys

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
