#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt

def give_param_names():

    param_names_dict = {
        'm_mul1':[r"$m_{1r1}$",r"$m_{1r2}$",r"$m_{1r3}$",\
    r"$m_{1r4}$",r"$m_{1r5}$",r"$m_{1r6}$",r"$m_{1r7}$", 12], \
        'm_mul2':[r"$m_{2r1}$",r"$m_{2r2}$",r"$m_{2r3}$",\
    r"$m_{2r4}$",r"$m_{2r5}$",r"$m_{2r6}$",r"$m_{2r7}$",13],\

    'beta1':[r"$\beta_{1r1}$",r"$\beta_{1r2}$",r"$\beta_{1r3}$",\
        r"$\beta_{1r4}$",r"$\beta_{1r5}$",r"$\beta_{1r6}$",r"$\beta_{1r7}$",14],
    'beta2':[r"$\beta_{2r1}$",r"$\beta_{2r2}$",r"$\beta_{2r3}$",\
        r"$\beta_{2r4}$",r"$\beta_{2r5}$",r"$\beta_{2r6}$",r"$\beta_{2r7}$",15],
    'beta3':[r"$\beta_{3r1}$",r"$\beta_{3r2}$",r"$\beta_{3r3}$",\
        r"$\beta_{3r4}$",r"$\beta_{3r5}$",r"$\beta_{3r6}$",r"$\beta_{3r7}$",16],
    'beta4':[r"$\beta_{4r1}$",r"$\beta_{4r2}$",r"$\beta_{4r3}$",\
        r"$\beta_{4r4}$",r"$\beta_{4r5}$",r"$\beta_{4r6}$",r"$\beta_{4r7}$",17],
    'beta5':[r"$\beta_{5r1}$",r"$\beta_{5r2}$",r"$\beta_{5r3}$",\
        r"$\beta_{5r4}$",r"$\beta_{5r5}$",r"$\beta_{5r6}$",r"$\beta_{5r7}$",18],
    'beta6':[r"$\beta_{6r1}$",r"$\beta_{6r2}$",r"$\beta_{6r3}$",\
        r"$\beta_{6r4}$",r"$\beta_{6r5}$",r"$\beta_{6r6}$",r"$\beta_{6r7}$",19],
    'beta7':[r"$\beta_{7r1}$",r"$\beta_{7r2}$",r"$\beta_{7r3}$",\
        r"$\beta_{7r4}$",r"$\beta_{7r5}$",r"$\beta_{7r6}$",r"$\beta_{7r7}$",20],
    'beta8':[r"$\beta_{8r1}$",r"$\beta_{8r2}$",r"$\beta_{8r3}$",\
        r"$\beta_{8r4}$",r"$\beta_{8r5}$",r"$\beta_{8r6}$",r"$\beta_{8r7}$",21],
    'beta9':[r"$\beta_{9r1}$",r"$\beta_{9r2}$",r"$\beta_{9r3}$",\
        r"$\beta_{9r4}$",r"$\beta_{9r5}$",r"$\beta_{9r6}$",r"$\beta_{9r7}$",22],
    'beta10':[r"$\beta_{10r1}$",r"$\beta_{10r2}$",r"$\beta_{10r3}$",\
        r"$\beta_{10r4}$",r"$\beta_{10r5}$",r"$\beta_{10r6}$",r"$\beta_{10r7}$",23],

    'global_scalars': [r"$dI$", r"$k_{sn}$", r"$k_{sp}$", r"$\eta$", r"$\beta_{sd}$", (0,1,2,3,11)], \
    'p_dead':  [r"$p_{a1}$",r"$p_{a2}$",r"$p_{a3}$",\
        r"$p_{a4}$",r"$p_{a5}$",r"$p_{a6}$",r"$p_{a7}$",(np.arange(4,11))]
    }
    return param_names_dict
