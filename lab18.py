#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:17:24 2024

@author: may
"""

def create_dict(f):
    d = {}
    with open(f, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(",")
            d[line[0]] = line[2]
    return d

def main():
    d = create_dict("words.txt")
    while True:
        word = input("Enter a word to define: ")
        try:
            print(f"Defintion:{d[word]}")
        except:
            print("That word is not in our dictionary")

if __name__ == "__main__":
    main()