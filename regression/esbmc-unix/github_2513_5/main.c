#include <stdlib.h>
#include <alloca.h>

extern int __VERIFIER_nondet_int(void);

int test_fun(int x, int y)
{
    int* x_ref = alloca(sizeof(int));
    int* y_ref = alloca(sizeof(int));
    *x_ref = x;
    *y_ref = y;
    while (*x_ref >= 0) {
        *y_ref = 1;
        while (*x_ref > *y_ref && *y_ref < 1073741824) {
            if (*y_ref <= 0) {
                // replace assume
                return *x_ref;
            }
            *y_ref = 2*(*y_ref);
        }
        *x_ref = *x_ref - 1;
    }
    return *y_ref;
}

int main() {
  return test_fun(__VERIFIER_nondet_int(),__VERIFIER_nondet_int());
}
