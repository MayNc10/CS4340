import pandas as pd

main_folder = False
path_prepend = "Project 4/" if main_folder else ""

path = path_prepend + "banknote_data.csv"
data = pd.read_csv(path)
train = data.sample(frac=0.8, random_state=7)
test = data.drop(train.index)

train.to_csv(path_prepend + "banknote_train.csv", index=False)
test.to_csv(path_prepend + "banknote_test.csv", index=False)