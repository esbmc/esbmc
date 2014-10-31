#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void __VERIFIER_error(void){
   assert(0);
}

int main() {
    int a = 1, b = 0, c;

    c = a || b;

    if (!c) {
        __VERIFIER_error();
        return 1;
    }

    return 0;
}  
