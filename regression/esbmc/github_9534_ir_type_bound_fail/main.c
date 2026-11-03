// #9534: the assumptions against type bounds are kept under --ir, and the
// properties below still fail because the bounds do not imply them.
extern unsigned long nondet_ulong(void);
extern unsigned char nondet_uchar(void);
extern int nondet_int(void);

int main(void)
{
  unsigned long u = nondet_ulong();
  __ESBMC_assume(u <= 18446744073709551615UL);
  __ESBMC_assert(u / 2 <= 4611686018427387903UL, "unsigned max");

  unsigned long v = nondet_ulong();
  __ESBMC_assume(v >= 0);
  __ESBMC_assume(v <= 10);
  __ESBMC_assert(v * v < 100, "unsigned min");

  int x = nondet_int();
  __ESBMC_assume(x >= -2147483647 - 1);
  __ESBMC_assert(x > -2147483648LL, "signed min");

  unsigned char c = nondet_uchar();
  __ESBMC_assume(c >= 0);
  __ESBMC_assume(c <= 10);
  __ESBMC_assert(c * c < 100, "promoted unsigned");
  return 0;
}
