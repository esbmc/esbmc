---
title: Benchexec
---

After running `benchexec` on a folder `f` you can do the following inside the `f` folder:

- **Get the summary**: `tail results/*.txt`. This will print the summary of Correct and Incorrect results
- **Generate CSV/HTML**: by using [table-generator](https://github.com/sosy-lab/benchexec/blob/master/doc/table-generator.md)
- **Score summary**: This is useful for getting an idea of the Score for SV-Comp21. Just run `wget -qO- https://raw.githubusercontent.com/rafaelsamenezes/scripts/master/benchexec_stuff/benchexec_to_csv.py | python3 -` this will create a file `output.csv` which you can select the lines and paste them into our [main spreadsheet](https://docs.google.com/spreadsheets/d/1trD9_kS-S8yhtCdyux3Hm1BBpIOYVvp-zJFYJiM00bs/edit?usp=sharing)

## Actions Machines

- Ubuntu 20.04
- Benchexec installation
- Chrony