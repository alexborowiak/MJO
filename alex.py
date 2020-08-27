import numpy as np



def add(x,y):
    return x + y ** 2


def awap_func(awap):
    a = awap.groupby('time.month').sum()
    
    return a
    