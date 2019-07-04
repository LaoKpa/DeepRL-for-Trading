import pandas as pd
import glob
import re

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def readInFiles(path, num):
    print("ok")
    findata = []
    all_files = glob.glob(path + "/*.txt")
    fileNum = 0
    for filename in all_files:
        if fileNum < num:
            # print(re.search(r'\d', filename))
            try:
                if not(hasNumbers(filename)):
                    print(filename)
                    df = pd.read_csv(filename)
                    x = df.values #returns a numpy array
                    findata.append(x)
                    fileNum += 1
            except pd.io.common.EmptyDataError:
                print("file is empty")
    return findata