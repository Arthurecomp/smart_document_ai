import pandas as pd

def load_raw_data(pathtrain: str, pathtest: str):
    """
    Carrega os dados, e ja faz a transformação de colunas
    """
    df_train = pd.read_csv(pathtrain)
    df_test = pd.read_csv(pathtest)

    df_train.rename(columns= {"Class Index" : "class", "Title":"title", "Description": "description"}, inplace = True)
    df_test.rename(columns= {"Class Index" : "class", "Title":"title", "Description": "description"}, inplace = True)

    df_train['text']  = df_train['title']+ df_train['description']
    df_train = df_train.drop(columns=['title'])
    df_train = df_train.drop(columns=['description'])


    df_test['text']  = df_test['title']+ df_test['description']
    df_test = df_test.drop(columns=['title'])
    df_test = df_test.drop(columns=['description'])

    df_train['class'] = df_train['class'].apply(lambda x : x-1)
    df_test['class'] = df_test['class'].apply(lambda x : x-1)

    X_train = df_train['text']
    y_train = df_train['class']

    X_test = df_test['text']
    y_test = df_test['class']

    return X_train, y_train, X_test, y_test, df_train, df_test


