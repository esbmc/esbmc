/*
FSE 2025 Tool Demonstration of ESBMC GCSE
This program is the motivating example from the paper titled "VO-GCSE: Verification Optimization through Global Common
Subexpression Elimination" to highlight the efficiency of GCSE optimization.
By enabling the `--gcse` switch in ESBMC, the C program can be verified significantly faster.

In this example, verification took more than 71.8 seconds on an Intel Xeon Platinum 8375C 32-core CPU with 128 GB of RAM.
However, with GCSE enabled, the verification completes SUCCESSFULLY in just 4 seconds.

To run the program without GCSE:
    bin/esbmc GCSE_motivating_example.c 

To run the program with GCSE:
    bin/esbmc GCSE_motivating_example.c --gcse
*/


#include <stdio.h>

typedef struct { unsigned Flags;} Aux;
typedef struct 
{ 
    Aux Aux; 
    unsigned Wc; 
    unsigned V;
} RegEntry;
typedef struct {RegEntry *Map;} table;
void write(table *tbl, unsigned EntryIndex);

int main() 
{
    RegEntry e[10000]; table M; M.Map = e;
    for (int i = 0; i < 10000; i++) write(&M, i);
    return 0;   
}

void write(table *tbl, unsigned EntryIndex) 
{
    unsigned Data64 = 42; 
    tbl->Map[EntryIndex].Aux.Flags &= 1;
    tbl->Map[EntryIndex].Wc++;
    tbl->Map[EntryIndex].V = Data64;

}
