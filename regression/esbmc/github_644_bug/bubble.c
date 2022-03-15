#define ARRAY_SIZE 4
#include "stdio.h"

int main()
{
    int temp;

    int table[ARRAY_SIZE];
    for(i = 0; i < ARRAY_SIZE; i++)
       __ESBMC_assume(table[i] < 10 && table[i] > 0);

    int ti, tj;
    for(i = 0; i < ARRAY_SIZE; i++)
    {
        ti = table[i];
        for(j  = i + 1; j < ARRAY_SIZE; j++)
        {
            tj = table[j];
            if(table[i] > table[j])
            {
                temp = table[j];
                table[j] = table[i];
                table[i] = temp;
                __ESBMC_assert(table[i] <= table[j],"kuku");
            }
            ti = table[i];
            tj = table[j];
            __ESBMC_assert(table[i] <= table[j], "ha-ha");
        }
    }
   return 0;
}