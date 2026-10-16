/* #9000: naming *c->p permits writing through the field. */
struct C { int *p; int k; };
int g;
void f(struct C *c) { __ESBMC_assigns(*c->p); *c->p = 5; }
int main() { struct C c; c.p = &g; c.k = 0; f(&c); return 0; }
