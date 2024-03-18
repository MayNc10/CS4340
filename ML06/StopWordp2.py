import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

file = "Data/EmailSubjects.csv"
data = pd.read_csv(file)
first_column = data.columns[0]
second_column = data.columns[1]

def remove_stop_words_nltk(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

new_sentences = []

for idx in range(len(data[first_column])):
    new_sentences.append(( remove_stop_words_nltk(data[first_column][idx]), data[second_column][idx] )) 

new_df = pd.DataFrame(new_sentences, columns=[first_column, second_column])

new_df.to_csv("Data/EmailSubjectsFilteredNLTK.csv", index=False)