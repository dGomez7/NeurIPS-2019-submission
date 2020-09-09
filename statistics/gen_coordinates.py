import numpy as np
import matplotlib as mp

def gen_random_coords(file_mode):
    f = open('coordinates', file_mode)
    #  generate random coordinates
    x = np.round(np.random.random() * 100, 3)
    y = np.round(np.random.random() * 100, 3)
    f.write("{},{}".format(x, y))
    for _ in range(20):
        f.write("\n")
        x = np.round(np.random.random() * 100, 3)
        y = np.round(np.random.random() * 100, 3)
        f.write("{},{}".format(x, y))
    f.close()


def linear_func(m, b, x):
    return m*x + b

def quadratic_func(a, b, c, x):
    return a*x**2 + b*x + c

def gen_linear_coords(m, b, file_mode):
    f = open('coordinates', file_mode)
    if file_mode == "a":
         f.write("\n")
    x = 1
    y = linear_func(m,b,x)
    f.write("{},{}".format(x, y))
    for x in range(2, 20):
        f.write("\n")
        y = linear_func(m,b,x)
        f.write("{},{}".format(x, y))
    f.close()

def gen_quadratic_coords(a, b, c, file_mode):
    f = open('coordinates', file_mode)
    if file_mode == "a":
        f.write("\n")
    x = 1
    y = quadratic_func(a,b,c,x)
    f.write("{},{}".format(x, y))
    for x in range(2, 20):
        f.write("\n")
        y = quadratic_func(a,b,c,x)
        f.write("{},{}".format(x, y))
    f.close()

gen_quadratic_coords(3,8,2,"w")
gen_linear_coords(6,1,"a")