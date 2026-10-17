int nondet_int();
int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x == -1);
  long y = (long)x;
  signed char c = (signed char)(int)-128;
  return (int)y + c;
}
