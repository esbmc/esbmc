#include <assert.h>
#include <stdlib.h>

void calculate_output()
{
    int input = nondet_int();
    if (input == 2)
        assert(1);
    if (input == 1)
       ;
}

int main()
{
    calculate_output();
}
