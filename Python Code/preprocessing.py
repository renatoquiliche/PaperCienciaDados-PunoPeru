def preprocessing(df, umbral: float):
    import pandas as pd
    umbral = 0.02
    x = df.iloc[:,6:]
    print("Old data dimension", x.shape)
    # 1
    frequencies = df.iloc[:,7:-1].mean()
    index_features = frequencies[frequencies > umbral].index
    print("Features removed :", len(frequencies)-len(index_features))

    # 2
    final_data = x.loc[:,index_features]
    descriptive_cat = pd.concat([final_data.mean(), final_data.sum()], axis=1)

    #descriptive_cat.to_excel("descriptive_cat.xlsx")

    final_data[["gpc", "altura"]] = df[["gpc", "altura"]]
    
    # 3
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    final_data[["altura", "gpc"]] = scaler.fit_transform(x[["altura", "gpc"]])
    
    print("New data dimension", final_data.shape)
    return final_data