// *p reads x, and x += 101 after the exit test changes it, so *p < 100 does
// not hold at the next head and must not be assumed there.
int nondet_int();
int main()
{
  int x = 90, m = 0;
  int *p = &x;
  while (1)
  {
    __ESBMC_assert(x != 105, "x != 105");
    x -= 100;
    m = nondet_int();
    if (*p >= 100 || m == 12345)
      break;
    x += 101;
  }
}
