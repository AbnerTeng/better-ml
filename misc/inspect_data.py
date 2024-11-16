import pickle

with open('../pytorch_tutorial/data/reg_task.pkl', 'rb') as f:
    california = pickle.load(f)

print(california.data.shape)
print(california.target.shape)
print(california.feature_names)
