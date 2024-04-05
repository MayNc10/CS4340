#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:53:45 2024

@author: may
"""

import lab12functions

def main():
    word = input("Enter a word to reverse: ")
    print(lab12functions.reverse_word(word))
    
if __name__ == "__main__":
    main()