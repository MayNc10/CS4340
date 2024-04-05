#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:35:35 2024

@author: may
"""

def s_to_n(l):
    return list(map(int, l))

def n_to_abs(l):
    return list(map(abs, l))

def n_to_float(l):
    return list(map(float, l))

def name_to_len(l):
    return list(map(len, l))

def sum_list(l1, l2):
    return list(map(lambda a, b: a+b, l1, l2))

def diff_list(l1, l2):
    return list(map(lambda a, b: a-b, l1, l2))

def comp_list(l1, l2):
    return list(map(lambda a, b: a >= b, l1, l2))

def trunc_list(l1, l2):
    return list(map(lambda a, b: a[min(len(a), b)], l1, l2))

def filter_list(l):
    return list(filter(lambda a: a > 12, l))

def cap_list(l):
    return list(filter(lambda a: a < 102, l))

def hey_list(l):
    return list(filter(lambda a: a[:3] == "Hey", l))
