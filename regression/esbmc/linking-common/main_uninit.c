#include <assert.h>

int global;

int main(void) { assert(global == EXPECTED_VALUE); return 0; }
