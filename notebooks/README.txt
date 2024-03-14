const_func_general.py - you'll find generic constants and functions in this file that will be used by rest of the files, especially cons_func_distortion.py

cons_func_distortion.py - here you will find constants and functions mostly specific to our distortion calculations, you can find detailed information regarding these constants and functions in our paper

calc_distortion.py - this file has functions to calculate actuall distortions for different cases and eras

main_calc.ipynb - here we calculate transition probability and save it in folder "data": "../data/Probability.npz". Unless you would like to increase the grid size you should be able to use the presaved file for distortion calculations. Additionally you will find chi^2 maximum likelihood analysis here along with our main plot.

FIRAS.py - this script is used to import and make FIRAS data ready for the chi^2 analysis

