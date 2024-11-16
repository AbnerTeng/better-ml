from sklearn.datasets import fetch_california_housing

import pickle

california = fetch_california_housing()

with open('../pytorch_tutorial/data/reg_task.pkl', 'wb') as f:
    pickle.dump(california, f)
