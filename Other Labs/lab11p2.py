#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:59:05 2024

@author: may
"""

bad_numbers = [13, 14, 17, 18, 19]
def new_value(num):
    return 0 if (num in bad_numbers) else num

def main():
    num1 = new_value(int(input("Enter the first number ")))
    num2 = new_value(int(input("Enter the second number ")))
    num3 = new_value(int(input("Enter the third number ")))
    print(f"The sum of the numbers is {num1 + num2 + num3}")
    
if __name__ == "__main__":
    main()