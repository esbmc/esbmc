/* Same-named locals declared by an inner macro expanded twice inside one
 * outer expansion are two variables. */
#include <assert.h>
#define INNER(T, val, out) { T v = (val); out = (int)v; }
#define OUTER(a, b) INNER(unsigned char, 1000, a) INNER(int, 1000, b)
int main(void) { int r1, r2; OUTER(r1, r2); assert(r1 == 232); assert(r2 == 1000); return 0; }
