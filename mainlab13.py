#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:26:39 2024

@author: may
"""

import loopfunctionslab13

def main():
    selection = input("What problem do you want to do? ")
    if selection == "0":
        num = int(input("What number should the start be? "))
        loopfunctionslab13.backwards_numbers(num)
    elif selection == "1":
        num = int(input("Enter a number "))
        loopfunctionslab13.powers(num)
    elif selection == "2":
        num = int(input("How many bottles do you have? "))
        loopfunctionslab13.bottles_of_beer(num)
    elif selection == "3":
        loopfunctionslab13.p4()
    else:
        print("Invalid selection")

if __name__ == "__main__":
    main()