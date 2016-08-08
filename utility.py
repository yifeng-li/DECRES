"""
Utilities.

Funtionality include:

Yifeng Li
CMMT, UBC, Vancouver
July 21, 2015
Contact: yifeng.li.cn@gmail.com
"""
from __future__ import division
import numpy as np
import math
import os

def strings_contain(strings,string_given):
    """
    Find whether a given a substring is contained in a list of substrings.
    
    INPUTS:
    strings: numpy 1d array or list.

    string_given: string type.

    OUTPUT:
    ind_log: A logical 1d array.
    """
    strings=np.array(strings)
    ind_log=np.empty(strings.shape,dtype=bool)
    i=0
    for s in strings:
        if string_given in s:
            ind_log[i]=True
        else:
            ind_log[i]=False
        i=i+1

    return ind_log

def convert_each_row_of_matrix_to_a_string(F,sep="_"):
    """ convert each row of matrix F to a string.
    """
    F_str=[]
    for f in F:
        f_str=np.asarray(f,dtype=str)
        F_str.append(sep.join(f_str)) # not attend
    F_str=np.array(F_str)
    return F_str


