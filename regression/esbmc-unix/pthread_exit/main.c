#include<assert.h>
#include <pthread.h>

int main() {
        int exit = __VERIFIER_nondet_int();
        //__VERIFIER_assume(exit == 0);
        if(exit) pthread_exit(0);
        assert(0);
}