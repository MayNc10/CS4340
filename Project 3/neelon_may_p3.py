import numpy as np
import pandas as pd

def print_cm(con_mat):
    #print(f"{con_mat}")
    accuracy = (con_mat[0][0] + con_mat[1][1]) / (con_mat[0][0] + con_mat[1][1] + con_mat[1][0] + con_mat[0][1]) 
    precision = con_mat[1][1] / (con_mat[1][1] + con_mat[0][1])
    recall = con_mat[1][1] / (con_mat[1][1] + con_mat[1][0])
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"TP: {con_mat[1][1]}, TN: {con_mat[0][0]}, FP: {con_mat[0][1]}, FN: {con_mat[1][0]} Accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")

def calculate_probs(line, dict, counts):
    words = line.split()
    # the probability that a group of words appears in a spam email
    # is p(spam) * product(p(spam | word))
    # over p(spam) * product(p(spam | word)) +  p(not spam) * product(p(not spam | word))
    p_spam_log_sum = 1
    p_not_spam_log_sum = 1
    for word in words:
        (p_not_spam, p_spam) = [1 / (2 + counts[0]), 1 / (2 + counts[1])]
        if word in dict:
            (p_not_spam, p_spam) = dict[word]   
        p_spam_log_sum += np.log(p_spam)
        p_not_spam_log_sum += np.log(p_not_spam)
    p_not_spam_log_sum += np.log(counts[0] / sum(counts))
    p_spam_log_sum += np.log(counts[1] / sum(counts))

    probs = [p_not_spam_log_sum / (p_spam_log_sum + p_not_spam_log_sum), p_spam_log_sum / (p_spam_log_sum + p_not_spam_log_sum)]
    return probs

vocab = "Project 3/vocabulary_list.csv"
data = pd.read_csv(vocab)
words = data.columns[0]
p_ns = data.columns[1]
p_s = data.columns[2]
hams_spams = pd.read_csv("Project 3/hams_spams.csv")
results = np.array(hams_spams.iloc[0, :])
print(f"Results: {results}")
words_dict = {}

for idx in range(len(data[words])):
    words_dict[data[words][idx]] = (data[p_ns][idx], data[p_s][idx])

fin = open("Project 3/SpamTest.txt", "r", encoding = "unicode-escape")
lines = fin.readlines()

confusion_matrix = np.array([[0,0], [0,0]])
for line in lines:
    label = int(line[0])
    line = line[2:]
    probs = calculate_probs(line, words_dict, results)
    idx = 0 if probs[0] > probs[1] else 1
    confusion_matrix[idx][label] += 1

print_cm(confusion_matrix)