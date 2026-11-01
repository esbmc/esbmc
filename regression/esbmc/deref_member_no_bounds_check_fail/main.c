struct I { int a[2]; };
struct O { int x; struct I in; int y; };
int main()
{
  struct O o = {0};
  struct I *p = &o.in;
  unsigned i = nondet_uint();
  __ESBMC_assume(i <= 2);
  p->a[i] = 7;
  __ESBMC_assert(i != 2 || o.y == 0, "false: p->a[2] is o.y");
  return 0;
}
