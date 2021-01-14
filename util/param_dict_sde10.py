#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt

def give_param_names(ind=False):
    if ind:
        param_names_dict = {'psir':[r"$\psi_{r1}$",r"$\psi_{r2}$",r"$\psi_{r3}$",\
        r"$\psi_{r4}$",r"$\psi_{r5}$",r"$\psi_{r6}$",r"$\psi_{r7}$",4], \
            'I0':[r"$Io_{r1}$",r"$Io_{r2}$",r"$Io_{r3}$",\
        r"$Io_{r4}$",r"$Io_{r5}$",r"$Io_{r6}$",r"$Io_{r7}$",5] ,\

            'm_mul1':[r"$m_{1r1}$",r"$m_{1r2}$",r"$m_{1r3}$",\
        r"$m_{1r4}$",r"$m_{1r5}$",r"$m_{1r6}$",r"$m_{1r7}$", 14], \
            'm_mul2':[r"$m_{2r1}$",r"$m_{2r2}$",r"$m_{2r3}$",\
        r"$m_{2r4}$",r"$m_{2r5}$",r"$m_{2r6}$",r"$m_{2r7}$",15],\
            'm_mul3':[r"$m_{3r1}$",r"$m_{3r2}$",r"$m_{3r3}$",\
        r"$m_{3r4}$",r"$m_{3r5}$",r"$m_{3r6}$",r"$m_{3r7}$",16],

        'beta1':[r"$\beta_{1r1}$",r"$\beta_{1r2}$",r"$\beta_{1r3}$",\
            r"$\beta_{1r4}$",r"$\beta_{1r5}$",r"$\beta_{1r6}$",r"$\beta_{1r7}$",17],
        'beta2':[r"$\beta_{2r1}$",r"$\beta_{2r2}$",r"$\beta_{2r3}$",\
            r"$\beta_{2r4}$",r"$\beta_{2r5}$",r"$\beta_{2r6}$",r"$\beta_{2r7}$",18],
        'beta3':[r"$\beta_{3r1}$",r"$\beta_{3r2}$",r"$\beta_{3r3}$",\
            r"$\beta_{3r4}$",r"$\beta_{3r5}$",r"$\beta_{3r6}$",r"$\beta_{3r7}$",19],
        'beta4':[r"$\beta_{4r1}$",r"$\beta_{4r2}$",r"$\beta_{4r3}$",\
            r"$\beta_{4r4}$",r"$\beta_{4r5}$",r"$\beta_{4r6}$",r"$\beta_{4r7}$",20],
        'beta5':[r"$\beta_{5r1}$",r"$\beta_{5r2}$",r"$\beta_{5r3}$",\
            r"$\beta_{5r4}$",r"$\beta_{5r5}$",r"$\beta_{5r6}$",r"$\beta_{5r7}$",21],
        'beta6':[r"$\beta_{6r1}$",r"$\beta_{6r2}$",r"$\beta_{6r3}$",\
            r"$\beta_{6r4}$",r"$\beta_{6r5}$",r"$\beta_{6r6}$",r"$\beta_{6r7}$",22],
        'beta7':[r"$\beta_{7r1}$",r"$\beta_{7r2}$",r"$\beta_{7r3}$",\
            r"$\beta_{7r4}$",r"$\beta_{7r5}$",r"$\beta_{7r6}$",r"$\beta_{7r7}$",23],
        'beta8':[r"$\beta_{8r1}$",r"$\beta_{8r2}$",r"$\beta_{8r3}$",\
            r"$\beta_{8r4}$",r"$\beta_{8r5}$",r"$\beta_{8r6}$",r"$\beta_{8r7}$",24],
        'beta9':[r"$\beta_{9r1}$",r"$\beta_{9r2}$",r"$\beta_{9r3}$",\
            r"$\beta_{9r4}$",r"$\beta_{9r5}$",r"$\beta_{9r6}$",r"$\beta_{9r7}$",25],
        'beta10':[r"$\beta_{10r1}$",r"$\beta_{10r2}$",r"$\beta_{10r3}$",\
            r"$\beta_{10r4}$",r"$\beta_{10r5}$",r"$\beta_{10r6}$",r"$\beta_{10r7}$",26],
        'beta11':[r"$\beta_{11r1}$",r"$\beta_{11r2}$",r"$\beta_{11r3}$",\
            r"$\beta_{11r4}$",r"$\beta_{11r5}$",r"$\beta_{11r6}$",r"$\beta_{11r7}$",27],
        'beta12':[r"$\beta_{12r1}$",r"$\beta_{12r2}$",r"$\beta_{12r3}$",\
            r"$\beta_{12r4}$",r"$\beta_{12r5}$",r"$\beta_{12r6}$",r"$\beta_{12r7}$",28],
        'beta13':[r"$\beta_{13r1}$",r"$\beta_{13r2}$",r"$\beta_{13r3}$",\
            r"$\beta_{13r4}$",r"$\beta_{13r5}$",r"$\beta_{13r6}$",r"$\beta_{13r7}$",29],
        'beta14':[r"$\beta_{14r1}$",r"$\beta_{14r2}$",r"$\beta_{14r3}$",\
            r"$\beta_{14r4}$",r"$\beta_{14r5}$",r"$\beta_{14r6}$",r"$\beta_{14r7}$",30],
        'beta15':[r"$\beta_{15r1}$",r"$\beta_{15r2}$",r"$\beta_{15r3}$",\
            r"$\beta_{15r4}$",r"$\beta_{15r5}$",r"$\beta_{15r6}$",r"$\beta_{15r7}$",31],
        'beta16':[r"$\beta_{16r1}$",r"$\beta_{16r2}$",r"$\beta_{16r3}$",\
            r"$\beta_{16r4}$",r"$\beta_{16r5}$",r"$\beta_{16r6}$",r"$\beta_{16r7}$",32],

        'global_scalars': [r"$dI$", r"$k_{sn}$", r"$k_{sp}$", r"$\eta$", r"$\beta_{sd}$", (0,1,2,3,13)], \
        'p_dead':  [r"$p_{a1}$",r"$p_{a2}$",r"$p_{a3}$",\
            r"$p_{a4}$",r"$p_{a5}$",r"$p_{a6}$",r"$p_{a7}$",(np.arange(6,13))]
        }
    else:
        
        param_names_dict = {'psir':[r"$\psi_{r1}$",r"$\psi_{r2}$",r"$\psi_{r3}$",\
        r"$\psi_{r4}$",r"$\psi_{r5}$",r"$\psi_{r6}$",r"$\psi_{r7}$",12], \
            'I0':[r"$Io_{r1}$",r"$Io_{r2}$",r"$Io_{r3}$",\
        r"$Io_{r4}$",r"$Io_{r5}$",r"$Io_{r6}$",r"$Io_{r7}$",13] ,\

            'm_mul1':[r"$m_{1r1}$",r"$m_{1r2}$",r"$m_{1r3}$",\
        r"$m_{1r4}$",r"$m_{1r5}$",r"$m_{1r6}$",r"$m_{1r7}$", 14], \
            'm_mul2':[r"$m_{2r1}$",r"$m_{2r2}$",r"$m_{2r3}$",\
        r"$m_{2r4}$",r"$m_{2r5}$",r"$m_{2r6}$",r"$m_{2r7}$",15],\
            'm_mul3':[r"$m_{3r1}$",r"$m_{3r2}$",r"$m_{3r3}$",\
        r"$m_{3r4}$",r"$m_{3r5}$",r"$m_{3r6}$",r"$m_{3r7}$",16],

        'beta1':[r"$\beta_{1r1}$",r"$\beta_{1r2}$",r"$\beta_{1r3}$",\
            r"$\beta_{1r4}$",r"$\beta_{1r5}$",r"$\beta_{1r6}$",r"$\beta_{1r7}$",17],
        'beta2':[r"$\beta_{2r1}$",r"$\beta_{2r2}$",r"$\beta_{2r3}$",\
            r"$\beta_{2r4}$",r"$\beta_{2r5}$",r"$\beta_{2r6}$",r"$\beta_{2r7}$",18],
        'beta3':[r"$\beta_{3r1}$",r"$\beta_{3r2}$",r"$\beta_{3r3}$",\
            r"$\beta_{3r4}$",r"$\beta_{3r5}$",r"$\beta_{3r6}$",r"$\beta_{3r7}$",19],
        'beta4':[r"$\beta_{4r1}$",r"$\beta_{4r2}$",r"$\beta_{4r3}$",\
            r"$\beta_{4r4}$",r"$\beta_{4r5}$",r"$\beta_{4r6}$",r"$\beta_{4r7}$",20],
        'beta5':[r"$\beta_{5r1}$",r"$\beta_{5r2}$",r"$\beta_{5r3}$",\
            r"$\beta_{5r4}$",r"$\beta_{5r5}$",r"$\beta_{5r6}$",r"$\beta_{5r7}$",21],
        'beta6':[r"$\beta_{6r1}$",r"$\beta_{6r2}$",r"$\beta_{6r3}$",\
            r"$\beta_{6r4}$",r"$\beta_{6r5}$",r"$\beta_{6r6}$",r"$\beta_{6r7}$",22],
        'beta7':[r"$\beta_{7r1}$",r"$\beta_{7r2}$",r"$\beta_{7r3}$",\
            r"$\beta_{7r4}$",r"$\beta_{7r5}$",r"$\beta_{7r6}$",r"$\beta_{7r7}$",23],
        'beta8':[r"$\beta_{8r1}$",r"$\beta_{8r2}$",r"$\beta_{8r3}$",\
            r"$\beta_{8r4}$",r"$\beta_{8r5}$",r"$\beta_{8r6}$",r"$\beta_{8r7}$",24],
        'beta9':[r"$\beta_{9r1}$",r"$\beta_{9r2}$",r"$\beta_{9r3}$",\
            r"$\beta_{9r4}$",r"$\beta_{9r5}$",r"$\beta_{9r6}$",r"$\beta_{9r7}$",25],
        'beta10':[r"$\beta_{10r1}$",r"$\beta_{10r2}$",r"$\beta_{10r3}$",\
            r"$\beta_{10r4}$",r"$\beta_{10r5}$",r"$\beta_{10r6}$",r"$\beta_{10r7}$",26],        
        'global_scalars': [r"$dI$", r"$k_{sn}$", r"$k_{sp}$", r"$\eta$", r"$\beta_{sd}$", (0,1,2,3,11)], \
        'p_dead':  [r"$p_{a1}$",r"$p_{a2}$",r"$p_{a3}$",\
            r"$p_{a4}$",r"$p_{a5}$",r"$p_{a6}$",r"$p_{a7}$",(np.arange(4,11))]
        }
    return param_names_dict
"""
        'beta6':[r"$\beta_{6r1}$",r"$\beta_{6r2}$",r"$\beta_{6r3}$",\
            r"$\beta_{6r4}$",r"$\beta_{6r5}$",r"$\beta_{6r6}$",r"$\beta_{6r7}$",22],
        'beta7':[r"$\beta_{7r1}$",r"$\beta_{7r2}$",r"$\beta_{7r3}$",\
            r"$\beta_{7r4}$",r"$\beta_{7r5}$",r"$\beta_{7r6}$",r"$\beta_{7r7}$",23],
        'beta8':[r"$\beta_{8r1}$",r"$\beta_{8r2}$",r"$\beta_{8r3}$",\
            r"$\beta_{8r4}$",r"$\beta_{8r5}$",r"$\beta_{8r6}$",r"$\beta_{8r7}$",24],
        'beta9':[r"$\beta_{9r1}$",r"$\beta_{9r2}$",r"$\beta_{9r3}$",\
            r"$\beta_{9r4}$",r"$\beta_{9r5}$",r"$\beta_{9r6}$",r"$\beta_{9r7}$",25],
        'beta10':[r"$\beta_{10r1}$",r"$\beta_{10r2}$",r"$\beta_{10r3}$",\
            r"$\beta_{10r4}$",r"$\beta_{10r5}$",r"$\beta_{10r6}$",r"$\beta_{10r7}$",26],
"""