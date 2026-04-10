---
title: SV-COMP and Test-COMP
---


# Analyzing the results

After running following the instructions provided [here](https://github.com/esbmc/esbmc/wiki/Benchexec) to obtain two .csv files containing the results of two runs of ESBMC on SV-COMP benchmarks, do the following to obtain an analysis of the results:

Obtain the script `analyze_esbmc_results` located in `esbmc/scripts/competitions/results_analysis/` and run the command:

`python analyze_esbmc_results file1.csv file2.csv`

The script will print an analysis of the runs and exit. 

# Creating ESBMC release

Usually, it is just a matter of putting all the files into a folder called `esbmc`. Note that:

1. If you zip from macos, be sure that the archive does not contain the `__MACOSX__` folder
2. The `esbmc` folder should be part of the archive