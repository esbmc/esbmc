struct I { int a[4]; };
struct O { int x; struct I in; int y; };
int main()
{
  struct O o = {0};
  struct I *p = &o.in;
  unsigned i = nondet_uint();
  __ESBMC_assume(i < 4);
  p->a[i] = 5;
  __ESBMC_assert(o.in.a[i] == 5, "written");
  __ESBMC_assert(o.x == 0 && o.y == 0, "neighbours untouched");
  return 0;
}
