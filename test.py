import pandas as pd
import numpy as np

def read_data():
    df=pd.read_csv("glass.csv")
    y=df["Type"]
    return y

def calc_gini():
    y=read_data()
    category=np.unique(y)
    gini=1
    for i in category:
        g=np.sum(y==i)/len(y)
        gini-=np.square(g)
    return gini

def main():
    print(calc_gini())

if __name__ == '__main__':
    main()