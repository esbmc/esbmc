#include <stdio.h>
#include <assert.h>

void __VERIFIER_error(void){
    assert(0);
}

int main() {
    int a = 1;
    int c = 55 || (a = 0);
    if (a) { __VERIFIER_error(); return 1; }
    return 0;
}    
