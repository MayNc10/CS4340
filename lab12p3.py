#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:01:44 2024

@author: may
"""

import lab12functions

def main():
    phrase = input("Enter a phrase: ")
    count = lab12functions.count_code(phrase)
    print(f"This string has {count} instance{'s' if count > 1 else '' } of the word code.")
    
if __name__ == "__main__":
    main()