#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:02:54 2024

@author: may
"""

def avg_list(l):
    return sum(l) / len(l)

def get_grades():
    l = []
    while True:
        grade = input("Please enter a grade, or something that isn't a number to exit: ")
        try:
            num = int(grade)
            l.append(num)
        except:
            break
    return l

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
    
def main():
    name = input("What is your name? ")
    grades = get_grades()
    avg = avg_list(grades)
    print(f"Average: {avg}, letter grade: {num_to_letter(avg)}")
    
if __name__ == "__main__":
    main()