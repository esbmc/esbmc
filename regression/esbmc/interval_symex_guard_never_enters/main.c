/* The assume establishes x >= 10, so the exit guard !(x < 5) is
   immediately proven TRUE by the interval domain — the loop body
   never executes. */
int main()
{
  int x;
  __ESBMC_assume(x >= 10);
  while (x < 5)
    ++x;
  __ESBMC_assert(x >= 10, "x must stay >= 10");
  return 0;
}
