from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd

NUM_FOLD=4
data_path = "/home/tungthanhlee/CGIAR/data"
data_csv = ""
def split(data_path, data_csv):
    
    df = pd.read_csv(data_csv)
    skf = StratifiedKFold(n_splits=NUM_FOLD, random_state=42, shuffle=True)
    X, y = df['ID'], df['label']
    data = data_path
    folds_path = os.path.join(data, 'Folds')
    if not os.path.isdir(folds_path):
        os.mkdir(folds_path)
        
    for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        df_train = pd.DataFrame(zip(X_train, y_train), columns=('id', 'label'))
        df_val = pd.DataFrame(zip(X_val, y_val), columns=('id', 'label'))
        df_train.to_csv(os.path.join(folds_path, f'train_fold{fold_idx}.csv'), index=False)
        df_val.to_csv(os.path.join(folds_path, f'valid_fold{fold_idx}.csv'), index=False)