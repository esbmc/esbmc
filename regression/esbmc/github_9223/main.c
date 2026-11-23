int nondet_int(void);
int main(void)
{
  int a = nondet_int();
  __ESBMC_assume(a >= 0 && a < 1024);
  int b = nondet_int();
  return a << (b & 15);
}
