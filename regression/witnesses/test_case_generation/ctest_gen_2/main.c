#include <assert.h>

extern _Bool __VERIFIER_nondet_bool(void);

int main(void) {
    _Bool flag = __VERIFIER_nondet_bool();

    if (flag) {
        return 1;
    }

    return 0;
}