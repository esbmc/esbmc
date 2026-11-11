struct E { int a; char b; };
struct O { int x; struct E es[4]; int y; };
int main()
{
  struct O o = {0};
  unsigned k = nondet_uint();
  __ESBMC_assume(k == 0 || k == 4);
  struct E *p = (struct E *)((char *)&o + k);
  p->a = 1;
  __ESBMC_assert(k != 0 || o.x == 0, "false: p->a is o.x when k == 0");
  return 0;
}
