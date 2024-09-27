const_func_general.py - you'll find generic constants and functions in this file that will be used by rest of the files, 
especially cons_func_distortion.py

cons_func_distortion.py - here you will find constants and functions mostly specific to our distortion calculations, 
you can find detailed information regarding these constants and functions in our paper

Prob_Ap_A.ipynb - In this file we calculate transition probability and save it in folder "data": "../data/Probability.npz".
Unless you would like to increase the grid size you should be able to use the presaved file for distortion calculations.

main_calc.ipynb - This is the main notebook to reproduce baund plots from the paper. 
We calculate the bounds here for free streaming case and with greens function method. 
Additionally you will find chi^2 maximum likelihood analysis here along with our main plot. 
To save time of running the actual calculations, unless one needs to increase the grid size, 
you should skip part of the code indicated in the notebook and only import presaved arrays.

FIRAS.py - this script is used to import and make FIRAS data ready for the chi^2 analysis

normalization_check.ipynb - here we check greens function normalization. 
Greens_Function_Plot_VaryTrans - Notebooks with this name are generating greens function fitting plots for various xp values
trans_and_delta_I.py - in this python code are saved functions calculatting delta_I
distorsion_signature.ipynb - In this file one can reproduce distortion signature plots
main_plot.ipynb - In this notebook one will find plotying of the main figures from the paper. One can regenarate needed data from main_calc or just use it from the data folder.
