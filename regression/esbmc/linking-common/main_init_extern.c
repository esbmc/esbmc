#include <assert.h>

extern int global = 222;

int main(void) { assert(global == EXPECTED_VALUE); return 0; }
