1#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:07:17 2024

@author: may
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:26:39 2024

@author: may
"""

import lab14functions

def main():
    selection = input("What problem do you want to do? ")
    if selection == "0":
        lab14functions.get_grades()
    elif selection == "1":
        num = int(input("Enter a number "))
        lab14functions.decimal_to_binary(num)
    elif selection == "2":
        lab14functions.ascii_conversion()
    elif selection == "3":
        num = int(input("What should the maximum guess be? "))
        lab14functions.guessing_game(num)
    else:
        print("Invalid selection")

if __name__ == "__main__":
    main()