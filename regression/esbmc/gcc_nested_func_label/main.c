// Regression test for #4098: __label__ must not be misclassified as a
// typedef variable and captured by nested functions.
int main(void)
{
  __label__ done;
  int x = 0;
  void inc(void)
  {
    x++;
  }
  inc();
  inc();
  __ESBMC_assert(x == 2, "__label__ not captured");
  goto done;
  x = 99;
done:
  return 0;
}
