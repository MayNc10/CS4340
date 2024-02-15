#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:25:05 2024

@author: may
"""

def num_to_letter(num):
    num /= 10
    if num >= 9:
        return "A"
    elif num == 8:
        return "B"
    elif num == 7:
        return "C"
    elif num == 7:
        return "D"
    else:
        return "F"

def grades(fname, gname):
    with open(fname, 'r') as reading:
        s = reading.readlines()
        with open(gname, "w") as writing:
            for line in s:
                v = line.split(",")
                s_id = v[2]
                grades = [int(grade) for grade in v[3:]]
                avg = sum(grades) / len(grades)
                out = s_id + "," + str(avg) + "," + num_to_letter(avg) + "\n"
                writing.writelines(out)
    
if __name__ == "__main__":
    grades("students.txt", "grades.txt")
                    
        