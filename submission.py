import numpy as np
import pandas as pd

names = np.load("submission/names.npy")
models = [
    # "submission/test_exp8_fold0.npy",
    # "submission/test_exp8_fold1.npy",
    # "submission/test_exp8_fold2.npy",
    # "submission/test_exp8_fold3.npy",
    # "submission/test_exp8_fold4.npy".
    "/home/tungthanhlee/CGIAR/src/submission/test_exp14_fold0.npy",
    "/home/tungthanhlee/CGIAR/src/submission/test_exp14_fold1.npy",
    "/home/tungthanhlee/CGIAR/src/submission/test_exp14_fold2.npy",
    "/home/tungthanhlee/CGIAR/src/submission/test_exp14_fold3.npy"

]

output = 0
for m in models:
    output += np.load(m) / len(models)
    

data = []
for n,o in zip(names, output):
    o /= np.sum(o)
    # o = np.clip(o, 0.1, 0.9)
    data.append([n.split(".")[0]] + list(o))
df = pd.DataFrame(data=data, columns=["ID", "leaf_rust", "stem_rust", "healthy_wheat"])
df.to_csv("submission/submission.csv", index=False)

# leak_df = pd.read_csv("submission/leakgps.csv")
# norm_df = pd.read_csv("submission/submission.csv")
# leak_value = leak_df.values[:,1:]
# norm_value = norm_df.values[:,1:]
# names = norm_df.values[:,0]
# for i in range(leak_value.shape[0]):
#     for j in range(leak_value.shape[1]):
#         if leak_value[i,j] == 0. or leak_value[i,j] == 1.:
#             norm_value[i,j] = leak_value[i,j]
# for i in range(norm_value.shape[0]):
#     norm_value[i,:] /= np.sum(norm_value[i,:])
# data = [[n]+list(v) for n,v in zip(names, norm_value)]
# df = pd.DataFrame(data=data, columns=norm_df.columns)
# df.to_csv("submission/submission.csv", index=False)