// Failing assertion: verifier should catch the bug
int main()
{
  int x = 0;
  void inc()
  {
    x++;
  }
  inc();
  __ESBMC_assert(x == 99, "should fail: x is 1 not 99");
  return 0;
}
