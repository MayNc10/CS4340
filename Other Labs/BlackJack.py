#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:27:17 2024

@author: may
"""
from math import *

def intput(s):
    return int(input(s))


'''
num1 = intput("What is the first number? ")
num2 = intput("What is the second number? ")

num1_final = copysign(1, (min(num1, 21) - 21)) * -1 * num1
num2_final = copysign(1, (min(num2, 21) - 21)) * -1 * num2

res = int(max(num1_final, num2_final))
print(f"{res}")

# of code paths is 1
'''

'''
num1 = abs(intput("What is the first number? "))
num2 = abs(intput("What is the second number? "))
num3 = abs(intput("What is the third number? "))

nums = [num1, num2, num3]
nums.sort()
nums[2] -= nums[0]
nums[1] -= nums[0]
nums[0] = 0


res = "not " * (abs(nums[2] - nums[0]) % nums[1])
print(res + "evenly spaced")

# of code paths is 1 
'''

def round_half_up(n, decimals=0):
    multiplier = 10**decimals
    return floor(n * multiplier + 0.5) / multiplier

num1 = round_half_up(intput("What is the first number? "), -1)
num2 = round_half_up(intput("What is the second number? "), -1) 
num3 = round_half_up(intput("What is the third number? "), -1)

print(int(num1 + num2 + num3))

# of code paths is 1

