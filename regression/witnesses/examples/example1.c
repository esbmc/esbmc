#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void __VERIFIER_error(void){
   assert(0);
}

int global = 5;

int foo() { return 1; };
int bar() { return 5; };
int pred(int p1, int by) { return p1 - by; };

int main() {
    int a = foo(), b = bar(); 
    global = global + b;

    while(global > 1) {
        global = pred(global, bar() + a);
    }
    
    a = 0;

    if (global != 0) {
        a = 1;
        __VERIFIER_error();
        return 1;
    }

    return 0;
}
