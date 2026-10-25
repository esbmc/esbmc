/* #9000: an assigns clause naming a pointer field covers that field. */
#include <stdlib.h>
struct C { int *p; int k; };
void f(struct C *c) { __ESBMC_assigns(c->p, c->k); c->p = malloc(4); c->k = 1; }
int main() { struct C c; c.p = 0; f(&c); return 0; }
