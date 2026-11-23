int nondet_int(void);
int a[4];
int main(void)
{
  int x = nondet_int();
  int *p = nondet_int() ? 0 : &x;
  __ESBMC_assume(x >= 0 && x < 5);
  /* *p is evaluated only when p is not NULL (C11 6.5.13p4). */
  if (p && a[*p] == 0)
    return 1;
  return 0;
}
