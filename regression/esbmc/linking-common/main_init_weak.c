#include <assert.h>

int global __attribute__((weak)) = 999;

int main(void) { assert(global == EXPECTED_VALUE); return 0; }
