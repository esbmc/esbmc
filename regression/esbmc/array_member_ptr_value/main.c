typedef struct { unsigned a; unsigned arr[4]; unsigned z; } S;
S g; S *p = &g;
int main(void)
{
  unsigned k = nondet_uint();
  __ESBMC_assume(k < 4);
  unsigned *q = &p->arr[k];
  *q = 7;
  __ESBMC_assert(g.arr[k] == 7 && g.a == 0 && g.z == 0, "value");
  return 0;
}
