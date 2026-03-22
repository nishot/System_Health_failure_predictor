import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path):
    df=pd.read_csv(path)

    df=df.drop(columns=['UDI', 
                        'Product ID','TWF', 'HDF', 'PWF', 'OSF','RNF'])
    

    le=LabelEncoder()

    df['Type']=pd.Series(le.fit_transform(df['Type']))


    return df



# df=load_and_preprocess("D:/system_failure_p1/ML/dataset/data.csv")

# print(df.head())