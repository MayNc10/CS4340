#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:14:21 2024

@author: may
"""

def intput(s):
    return int(input(s))


bad_numbers = [13, 14, 17, 18, 19]
def number_val(num):
    return 0 if (num in bad_numbers) else num

small = 1
large = 5

'''
height = intput("What is your height in inches? ")
print(f"You are {height // 12} feet and {height % 12} inches tall.")
'''


small_c = intput("How many small bricks do you have? ")
large_c = intput("How many large bricks do you have? ")
goal = intput("What is your goal length, in inches? ")

if (small_c * small + large_c * large < goal):
    print("You can’t reach your goal with the bricks you have.")
else:
    large_max_c = min(goal // large, large_c)
    small_max_c = min(goal - (large_max_c * large), small_c)
    if (large_max_c * large + small_max_c * small != goal):
        print("You can’t reach your goal with the bricks you have.")
    else:
        print(f"You need {small_max_c} small brick{'s' if (small_max_c != 1) else ''}.")
        print(f"You need {large_max_c} large brick{'s' if (large_max_c != 1) else ''}.")


'''
num1 = intput("What is number 1? ")
num2 = intput("What is number 2? ")
num3 = intput("What is number 3? ")

print(f"The sum is: {number_val(num1) + number_val(num2) + number_val(num3)}")
'''
