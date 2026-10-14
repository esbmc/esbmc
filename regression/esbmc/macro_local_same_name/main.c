/* Two same-named locals declared by one macro expansion are two variables. */
#include <assert.h>
#define TWO { unsigned char s = 200; r1 = s; } { int s = 1000; r2 = s; }
int main(void) { int r1, r2; TWO; assert(r1 == 200); assert(r2 == 1000); return 0; }
