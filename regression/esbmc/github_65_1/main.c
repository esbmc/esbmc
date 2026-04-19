#include <assert.h>

int *p = (int[]){2, 4}; // creates an unnamed static array of type int[2]
                        // initializes the array to the values {2, 4}
                        // creates pointer p to point at the first element of the array
const float *pc = (const float []){1e0, 1e1, 1e2}; // read-only compound literal
 
int main(void)
{
    int n = 2, *p = &n;
    p = (int [2]){*p}; // creates an unnamed automatic array of type int[2]
                       // initializes the first element to the value formerly held in *p
                       // initializes the second element to zero
                       // stores the address of the first element in p
    
    assert(p[0] != n); // this should fail

    return 0;
}
