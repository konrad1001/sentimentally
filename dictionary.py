import sys
import numpy as np

class Dictionary:
    FILE_PATH = "data/dictionary.txt"
    def __init__(self):
        self.dict = np.array([])
        self.load(self.FILE_PATH)

    def load(self, filename):
        self.dict = np.loadtxt(filename, dtype=str)
    
    def get(self, index):
        return self.dict[index]
    
    def get_index(self, word):
        try:
            return np.where(self.dict == word)[0][0]
        except IndexError:
            sys.stderr.write("Error: %d not found in dictionary\n", word)
            return -1
        

