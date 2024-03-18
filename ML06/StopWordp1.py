import numpy as np
import pandas as pd



file = "Data/EmailSubjects.csv"
data = pd.read_csv(file)
first_column = data.columns[0]
second_column = data.columns[1]

stop_words_f = open("Data/stop_words.txt")
stop_words = stop_words_f.read().split()

def filter_stop(word):
    return not word in stop_words

new_sentences = []

for idx in range(len(data[first_column])):
    new_sentences.append( (" ".join(list(filter(filter_stop, data[first_column][idx].split() ))), data[second_column][idx]) ) 

new_df = pd.DataFrame(new_sentences, columns=[first_column, second_column])

new_df.to_csv("Data/EmailSubjectsFiltered.csv", index=False)

stop_words_f.close()