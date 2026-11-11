struct E { int a; char b; };
struct O { int x; struct E es[4]; int y; };
int main() {
  struct O o = {0};
  unsigned i = nondet_uint(); __ESBMC_assume(i < 4);
  struct E *p = &o.es[i];
  p->a = 7; p->b = 3;
  __ESBMC_assert(o.es[i].a == 7 && o.es[i].b == 3, "written");
  __ESBMC_assert(o.x == 0 && o.y == 0, "neighbours untouched");
  unsigned j = nondet_uint(); __ESBMC_assume(j < 4 && j != i);
  __ESBMC_assert(o.es[j].a == 0, "others untouched");
  return 0;
}
