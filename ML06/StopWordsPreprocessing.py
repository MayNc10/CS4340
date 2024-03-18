import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

file = "Data/EmailSubjects.csv"
data = pd.read_csv(file)
first_column = data.columns[0]
second_column = data.columns[1]

words_dict = {}

def remove_stop_words_nltk(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

for idx in range(len(data[first_column])):
    filtered = remove_stop_words_nltk(data[first_column][idx])
    word_set = set()
    for word in filtered.split():
        word = word.lower()
        word = word.translate(str.maketrans('', '', string.punctuation))
        if word == '': continue
        word_set.add(word)
    for word in word_set:
        idx = data[second_column][idx]
        if word in words_dict:
            words_dict[word][idx] += 1
        else:
            base = [0, 0]
            base[idx] += 1
            words_dict[word] = base
