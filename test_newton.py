__author__ = 'kanhua'


import numpy as np
import matplotlib.pyplot as plt


def simple_func(x):
    return x**2-2*x+1

def simple_func_p(x):
    return 2*x-2


def find_root_newton(f,fp,x_init):

    max_iter=1000
    tolerance=1e-2

    current_x=x_init
    for i in range(0,max_iter):
        next_x=current_x-f(current_x)/fp(current_x)
        if (abs(next_x-current_x)/current_x)<tolerance:
            return next_x

    print("reach maximum iteration")
    return next_x



root=find_root_newton(simple_func,simple_func_p,0.99)

print(root)