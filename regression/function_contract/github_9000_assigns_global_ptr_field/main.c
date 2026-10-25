/* #9000: an assigns clause naming a global struct's pointer field covers it. */
struct S { int *p; int k; } s;
int g;
void f(void) { __ESBMC_assigns(s.p); s.p = &g; }
int main() { f(); return 0; }
