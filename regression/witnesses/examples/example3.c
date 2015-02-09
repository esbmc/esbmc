#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void __VERIFIER_error(void){
   assert(0);
}

int failure() {
    __VERIFIER_error();
    return 1;
}

int main() {
    int a = 1;
    return a && failure();
}    
