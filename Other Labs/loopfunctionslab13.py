#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 08:47:40 2024

@author: may
""" 

def backwards_numbers(n):
    if (n % 2 != 0):
        print("Error: Given number was not even")
    else:
        for i in range(n, -2, -2):
            print(i)
            
def powers(num):
    base = 2
    while base < num:
        print(base)
        base *= 2
        
def one_bottle_of_beer(beer):
    print(f"{beer} bottle{'s' if (beer != 1) else ''} of beer on the wall,")
    print(f"{beer} bottle{'s' if (beer != 1) else ''} of beer!")
    print("Take one down, pass it around,")
    beer -= 1
    print(f"{beer} bottle{'s' if (beer != 1) else ''} of beer on the wall!")
    return beer

def bottles_of_beer(num):
    while num > 0:
        num = one_bottle_of_beer(num)
        
def p4():
    num_nums = int(input("How many numbers do you have? "))
    divisor = int(input("Enter the divisor: "))
    s = 0
    for _ in range(num_nums):
        num = int(input("Enter a number "))
        if num % divisor == 0:
            s += num
    print(f"The sum of the numbers divisible by {divisor} is: {s}")
    
