/* Same-named local structs declared by an inner macro expanded twice inside
 * one outer expansion are distinct types. */
#include <assert.h>
#include <string.h>
#define INNER(A, B, V, out) { struct S { int A; int B; } V; memset(&V, 0, sizeof V); *(int *)&V = 7; out = V.a; }
#define OUTER(x, y) INNER(a, b, s1, x) INNER(b, a, s2, y)
int main(void) { int r1, r2; OUTER(r1, r2); assert(r1 == 7); assert(r2 == 0); return 0; }
