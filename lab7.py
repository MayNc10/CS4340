#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:46:45 2024

@author: may
"""

def intput(s):
    return int(input(s))

apple = .50
pear = .75
peach = .90

cheese = 15.0
pepperoni = 2.0
sausage = 2.0
onion = 2.0
pepper = 1.2
anchovies = 3.0   

test_list = ["true", "yes"]

def str_to_bool(s):
    return s.strip().lower() in test_list

name = input("What's your name? ")

'''
apple_count = intput("How many apples do you want? ")
pear_count = intput("How many pears do you want? ")
peach_count = intput("How many peaches do you want? ")
owe = apple_count * apple + pear_count * pear + peach_count * peach + ( 5 if(name == "Darth") else 0)

print(f"You owe {owe} dollars") 
'''

'''
pep_cost = float(int(str_to_bool(input("Do you want pepperoni? ")))) * pepperoni
sausage_cost = float(int(str_to_bool(input("Do you want sausage? ")))) * sausage
onion_cost = float(int(str_to_bool(input("Do you want onion? ")))) * onion
pepper_cost = float(int(str_to_bool(input("Do you want pepper? ")))) * pepper
anch_cost = float(int(str_to_bool(input("Do you want anchovies? ")) or (name == "Darth"))) * anchovies
owe = cheese + pep_cost + sausage_cost + onion_cost + pepper_cost + anch_cost

print(f"You owe {owe} dollars")
'''


'''
num1 = intput("What is the first number? ")
num2 = intput("What is the seconf number? ")
print(f"{max(num1, num2)} is the larger number")
'''

age = intput("How old are you? ")
flag = (age >= 18) and (name != "Darth")
print(f"You are {'' if (flag) else 'not ' }eligible to vote.{'!' if (flag) else '' }")