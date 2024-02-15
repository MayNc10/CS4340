#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:55:07 2024

@author: may
"""

import lab12functions

def main():
    word = input("Enter a word: ")
    letter = input("Enter a letter: ")
    count =  lab12functions.count_letter(word, letter)
    print(f"The word {word} has {count} of the letter {letter}.")
    
if __name__ == "__main__":
    main()