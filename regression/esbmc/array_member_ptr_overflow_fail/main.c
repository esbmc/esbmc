typedef struct { unsigned a; unsigned arr[4]; unsigned z; } S;
S g; S *p = &g;
int main(void)
{
  unsigned k = nondet_uint();
  __ESBMC_assume(k <= 4);
  unsigned *q = &p->arr[k];
  *q = 1;               /* k == 4 writes past arr into z */
  return 0;
}
