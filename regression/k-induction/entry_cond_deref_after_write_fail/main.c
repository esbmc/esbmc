// x += 100 changes what *p reads before the exit test, so the test cannot be
// rewritten over the head's values by looking at p alone.
int main()
{
  int x = 0, n = 0;
  int *p = &x;
  while (1)
  {
    __ESBMC_assert(x != 50, "x != 50");
    n++;
    x += 100;
    if (*p < 100 || n > 1000)
      break;
    x -= 99;
  }
}
