## Script to compare two runs of ESBMC on SV-COMP  

A python script. To run the script use:

`python analyze_esbmc_results.py file1.csv file2.csv`

where file1 and file2 are the .csv files produced by the BenchExec tool from the first and second run respectively. The output should look similar to:

```
current-naster_result-generation.table.csv
value-set_new (1).csv
total number solved by solver1 7702
total number solved by solver2 7663
total number of uniques solved by solver1 92
total number of uniques solved by solver2 53


Uniques 1:

  ldv-linux-3.16-rc1/43_2a_bitvector_linux-3.16-rc1.tar.xz-43_2a-drivers--usb--host--max3421-hcd.ko-entry_point.cil.out.yml_unreach-call
  ...

Uniques 2:

  eca-rers2012/Problem12_label20.yml_unreach-call
  ...
```