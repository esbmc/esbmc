struct E { int a; char b; };
struct O { int x; struct E es[4]; int y; };
int main() {
  struct O o = {0};
  unsigned i = nondet_uint(); __ESBMC_assume(i < 4);
  struct E *p = &o.es[i];
  p->a = 7;
  __ESBMC_assert(o.es[i].a == 8, "should fail");
  return 0;
}
