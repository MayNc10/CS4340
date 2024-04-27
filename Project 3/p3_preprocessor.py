import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
fin = open("Project 3/SpamTrain.txt", "r", encoding = "unicode-escape")
stop_words_file = open("Project 3/StopWords.txt", "r", encoding = "unicode-escape")
words = fin.readlines()
stop_words = set(stop_words_file.readlines())

words_dict = {}

def remove_stop_words(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

results = [0, 0]

for idx in range(len(words)):
    label = int(words[idx][0])
    filtered = remove_stop_words(words[idx][2:])
    word_set = set()
    for word in filtered.split():
        word = word.lower()
        word = word.translate(str.maketrans('', '', string.punctuation))
        if word == '': continue
        word_set.add(word)
    for word in word_set:
        if word in words_dict:
            words_dict[word][label] += 1
        else:
            base = [0, 0]
            base[label] += 1
            words_dict[word] = base
        results[label] += 1

k = 1 # the lab didn't say what to set this as, so I'm just going with this
num_hams = results[0]
num_spams = results[1]
prob_dict = {}
for word in words_dict.keys():
    p_spam = (k + words_dict[word][1]) / (2 * k + num_spams)
    p_not_spam = (k + words_dict[word][0]) / (2 * k + num_hams)
    prob_dict[word] = [p_not_spam, p_spam]

# write out file
prob_list = [(word, prob_dict[word][0], prob_dict[word][1]) for word in prob_dict.keys()]
new_df = pd.DataFrame(prob_list, columns=["Vocabulary", "P(not Spam)", "P(Spam)"])
new_df.to_csv("Project 3/vocabulary_list.csv", index=False)

results_df = pd.DataFrame([tuple(results)], columns=["Number of Hams", "Number of Spams"])
results_df.to_csv("Project 3/hams_spams.csv", index=False)

fin.close()
stop_words_file.close()