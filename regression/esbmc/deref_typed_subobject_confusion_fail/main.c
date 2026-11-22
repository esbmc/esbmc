struct E { int a; char b; };
struct O { int x; struct E es[4]; int y; };
int main() {
  struct O o = {0};
  unsigned k = nondet_uint(); __ESBMC_assume(k < 8);
  struct E *p = (struct E *)((char *)&o + k);   /* not at an element for most k */
  p->a = 1;
  return 0;
}
