// #9222: an unsigned product above UINT_MAX is an overflow.
unsigned nondet_uint(void);
int main(void)
{
  unsigned u = nondet_uint(), v = nondet_uint();
  __ESBMC_assume(u <= 65536 && v <= 65536);
  return (int)(u * v);
}
