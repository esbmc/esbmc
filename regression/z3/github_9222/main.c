// #9222: Z3 checks multiplication overflow with its own predicates. Products
// that reach the type's bounds exactly are not overflows.
int nondet_int(void);
unsigned nondet_uint(void);
int main(void)
{
  int a = nondet_int(), b = nondet_int();
  __ESBMC_assume(a >= -46340 && a <= 46340 && b >= -46340 && b <= 46340);
  int p = a * b;

  int c = nondet_int(), d = nondet_int();
  __ESBMC_assume(
    (c == -2147483647 && d == -1) || (c == -2147483647 - 1 && d == 1) ||
    (c == 65536 && d == -32768));
  int q = c * d;

  unsigned u = nondet_uint(), v = nondet_uint();
  __ESBMC_assume(u <= 65535 && v <= 65537);
  unsigned r = u * v;
  return 0;
}
