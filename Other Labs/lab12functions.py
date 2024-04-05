#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:48:10 2024

@author: may
"""

def reverse_word(word):
    new_word = ""
    idx = len(word) - 1
    while idx >= 0:
        new_word += word[idx]
        idx -= 1
    return new_word

def count_letter(word, letter):
    count = 0
    idx = 0
    while idx < len(word):
        if word[idx] == letter:
            count += 1
        idx += 1
    return count

def count_code(phrase):
    count = 0
    phrase = phrase.lower()
    idx = 0
    while idx <= (len(phrase) - 4):
        if phrase[idx:idx + 4] == "code":
            count += 1
        idx += 1
    return count