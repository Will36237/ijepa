import pandas as pd 

df = pd.read_csv('data/trading/data.csv')
df = df.drop(['time','zigzag_large','candle_type'], axis= 1)

grouped_dfs = {
    f"{symbol}_{tf}": group.reset_index(drop=True)
    for (symbol, tf), group in df.groupby(['symbol', 'timeframe'])
}

processed_dfs = {}
for key, sub_df in grouped_dfs.items():
    sub_df = sub_df.drop(columns=['symbol','timeframe'])

    total_len = len(sub_df)
    train_end = int(total_len * 0.8)  
    val_end = int(total_len * 0.9)     

    df_train = sub_df.iloc[:train_end].reset_index(drop=True)
    df_val   = sub_df.iloc[train_end:val_end].reset_index(drop=True)
    df_test  = sub_df.iloc[val_end:].reset_index(drop=True)

    target_col = 'zigzag_small'

    processed_dfs[key] = {
        'train': {'X': df_train.drop(columns=[target_col]), 'y': df_train[target_col]},
        'val'  : {'X': df_val.drop(columns=[target_col]),   'y': df_val[target_col]},
        'test' : {'X': df_test.drop(columns=[target_col]),  'y': df_test[target_col]},
    }