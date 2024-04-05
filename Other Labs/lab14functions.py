#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:44:20 2024

@author: may
"""

import random
import time
import math

def guessing_game(top):
    number = random.randint(0, top)
    while True:
        guess = int(input("Guess a number: "))
        if guess > number:
            print("Lower!")
        elif guess < number: 
            print("Higher!")
        else:
            break
    print("You guessed the number!")
    
def get_grades():
    l = []
    while True:
        grade = int(input("Please enter a grade: "))
        if grade >= 0:
            l.append(grade)
        else:
            break
    print(sum(l) / len(l))


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
    
def decimal_to_binary(num):
    max_power = math.ceil(math.log2(num))
    s = "0" * max_power
    power = max_power
    while num > 0:
        print(num, power, s)
        if num - 2 ** power >= 0:
            s = s[0:(max_power - power)] + "1" + s[(max_power - power + 1):]
            num -= 2 ** power
        power -= 1
    if s[0] == "0":
        s = s[1:]
    print(s)
        
def ascii_conversion():
    for num in range(97, 123):
        print(f"{num} {chr(num)}")
        