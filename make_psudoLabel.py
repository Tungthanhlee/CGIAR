import numpy as np
import pandas as pd




df = pd.read_csv("submission/submission.csv")
df_values = df.values
# print(df_values)
trust_threshold = 0.98
indices = []
for i in range(df_values.shape[0]):
    if np.any(df_values[i,1:] > trust_threshold):
        indices.append(i)
# print(len(indices))
trust_names = df.iloc[indices,0].values
# print(trust_names)
trust_output = np.argmax(df.iloc[indices,1:].values, axis=1)
trust_data = [[n]+[o] for n,o in zip(trust_names, trust_output)]
df = pd.DataFrame(data=trust_data, columns=['id', 'label'])
print(df.shape)
##########################################
fold = 0 #Remember to set this fold
labeled_data = pd.read_csv(f"/home/tungthanhlee/CGIAR/data/Folds/train_fold{fold}.csv")
df = df.append(labeled_data)
print(df.shape)
df.to_csv('/home/tungthanhlee/CGIAR/data/Folds/pseudo.csv', index=False)
