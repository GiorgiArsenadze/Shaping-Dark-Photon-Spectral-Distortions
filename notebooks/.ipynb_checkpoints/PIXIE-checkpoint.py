{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pckgs.import_pckg import *\n",
    "from pckgs.units import *\n",
    "\n",
    "\n",
    "# Import COBE-FIRAS data from astro-ph/9605054\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# PIXIE Projection Data got from 2405.20403\n",
    "# (Also see lambda.gsfc.nasa.gov/product/pixie/pixie_baseline_noise_get.html)\n",
    "# Stoke I Parameter\n",
    "\n",
    "# [1]. Center Frequency (GHz)\n",
    "# [2]. Sensitivity (Jy * sr^−1 * \\sqrt{sec})\n",
    "# [3]. Map Noise (Jy * sr^−1)\n",
    "\n",
    "# 2 year mission\n",
    "# 45% time for calibrator deployed (sensitive to spectral distortions and polarization)\n",
    "# 70% of the sky\n",
    "year_in_s = 365.2422*24*3600\n",
    "\n",
    "Eff_Time_in_s = 0.7 * 0.45 * (2*year_in_s)\n",
    "\n",
    "Jy_to_MJy = 10**(-6)\n",
    "\n",
    "PIXIE_data_ary = np.transpose( np.array(pd.read_csv('../data/data_input_bound/PIXIE_Proj24_Data.csv') ) )\n",
    "\n",
    "# nu [GHz]\n",
    "PIXIE_nu_GHz_ary = PIXIE_data_ary[0]\n",
    "\n",
    "# Sensitivity (Divided by sqrt(effective_integration_time) ) [Jy]\n",
    "PIXIE_sen_ary = PIXIE_data_ary[1]/np.sqrt(Eff_Time_in_s)\n",
    "\n",
    "# Length of PIXIE projection data\n",
    "len_PIXIE = len(PIXIE_nu_GHz_ary)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
