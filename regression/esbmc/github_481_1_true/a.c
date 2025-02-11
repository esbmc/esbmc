// extern_main.c
#include "extern.h"
int main() {
    __ESBMC_assert(ASD[5] == 0,"Ok");
    return 0;
}