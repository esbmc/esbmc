__ESBMC_logic int inc(int x) { return x + 1; }
int main()
{
  int x = 1;
  x = inc(x);
  __ESBMC_assert(x == 2, "inc");
  return 0;
}
