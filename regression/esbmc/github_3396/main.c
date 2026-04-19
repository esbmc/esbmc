#include <assert.h>

int nondet_int(void);
void __ESBMC_assume(_Bool);

typedef int (*handler_t)(int);

int add_one(int x) { return x + 1; }
int sub_one(int x) { return x - 1; }

handler_t handlers[2] = { add_one, sub_one };

int main() {
    int idx = nondet_int();
    __ESBMC_assume(idx >= 0 && idx < 2);

    handler_t h = handlers[idx];
    int result = h(5);

    assert(result == 6 || result == 4);
    return 0;
}
