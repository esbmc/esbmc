#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void __VERIFIER_error(void){
   assert(0);
}
 
#define ERRORMESSAGE(text) printf( stderr )

int nondet_int();

int main() {
    int r = nondet_int();
    int a,b;

    if (r) {
        a = 1;
        b = 10;
        while (b > 0) {
            b = b - 1;
        }
    } else {
        a = 2;
    }

#pragma STDC FENV_ACCESS ON

    if (a == 2) {
        ERRORMESSAGE("Error!!");
        __VERIFIER_error();
        return 1;
    }

    return 0;
}    
