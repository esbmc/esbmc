// #8661: under --ir a cast to a signed type was checked against [0, 2^N - 1],
// so every negative value was reported as a cast overflow.
int nondet_int();
int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x == -1);
  long y = (long)x;
  return (int)y;
}
