struct E { int a; char b; };
struct O { char buf[sizeof(struct E)]; struct E e; };
int main()
{
  struct O o = {0};
  char *base = nondet_bool() ? o.buf : (char *)&o.e;
  struct E *p = (struct E *)base;
  p->a = 1;
  __ESBMC_assert(p->a == 1, "stored");
  return 0;
}
