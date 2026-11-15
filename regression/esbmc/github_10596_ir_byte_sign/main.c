// Under --ir, a value rebuilt from bytes reads its bits as two's complement.
int nondet_int(void);
int main(void)
{
  int a[2];
  int *p = a;
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  a[0] = -16777217;
  a[1] = -1;
  __ESBMC_assert(p[i] < 0, "negative through pointer");
  __ESBMC_assert(((char *)a)[4] == -1, "byte as signed char");
  signed char c = -1;
  __ESBMC_assert(*(unsigned char *)&c == 255, "signed char as unsigned");
  return 0;
}
