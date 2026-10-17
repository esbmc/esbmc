// #8008: a pointer stored into an integer slot through a pointer.
int main()
{
  int x = 1;
  long cell;
  long *lp = &cell;
  *lp = (long)&x;
  int *q = (int *)*lp;
  int a = x + 1;
  __ESBMC_assert(*q == 1, "q aliases x");
  int b = x + 1;
  __ESBMC_assert(a == b, "x unchanged");
  return 0;
}
