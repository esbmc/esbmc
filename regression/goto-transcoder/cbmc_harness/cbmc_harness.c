#include <assert.h>
/* Selecting `verify` must run it, not main; main's assert(0) proves that. */
void verify(void) { int y = 7; assert(y == 7); }
int main(void) { assert(0); return 0; }
