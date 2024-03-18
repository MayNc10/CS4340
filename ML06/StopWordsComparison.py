import numpy as np
import pandas as pd
import difflib

filered1file = "Data/EmailSubjectsFiltered.csv"
filered1 = pd.read_csv(filered1file)

filered2file = "Data/EmailSubjectsFilteredNLTK.csv"
filered2 = pd.read_csv(filered2file)

diff_file = open("Data/EmailSubjectsDiff.txt", "w")

column_name = filered1.columns[0]

for idx in range(len(filered1[column_name])):
    custom = filered1[column_name][idx]
    nltk = filered2[column_name][idx]
    diff = difflib.ndiff(custom, nltk)
    diff_list = []
    for i,s in enumerate(diff):
        if s[0] == ' ': continue
        elif len(s[1:].strip()) == 0: continue
        else: diff_list.append(s)

    # compress diff list
    idx = 1
    while idx < len(diff_list):
        if diff_list[idx][0] == diff_list[idx - 1][0]:
            diff_list[idx - 1] = diff_list[idx - 1] + diff_list[idx][1:].strip()
            diff_list.pop(idx)
        else:
            idx += 1
    
    line = f"Custom: {custom}, NLTK: {nltk}, Diff: {', '.join(diff_list) if len(diff_list) > 0 else 'None'} "
    print(line)
    diff_file.write(line + "\n")

diff_file.close()
