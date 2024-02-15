#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:45:50 2024

@author: may
"""

def p1(num):
    return num if (num < 21) else 0

def main():
    num1 = int(input("Enter the first number "))
    num2 = int(input("Enter the second number "))
    print(f"{ max(p1(num1), p1(num2)) }");
    
if __name__ == "__main__":
    main()