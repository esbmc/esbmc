/* #8102: u == 4 wraps to 2, then 0, so the loop runs three times. */
int main()
{
  unsigned u;
  __ESBMC_assume(u >= 1 && u <= 4);
  int n = 0;
  while (u < 5)
  {
    u += 0xFFFFFFFEu;
    ++n;
  }
  __ESBMC_assert(n <= 3, "iterations");
  return 0;
}
