#include <assert.h>

int global = 999;

int main(void) { assert(global == EXPECTED_VALUE); return 0; }
