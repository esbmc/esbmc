// Basic nested function with captured write
int main()
{
  int x = 0;
  void inc()
  {
    x++;
  }
  inc();
  __ESBMC_assert(x == 1, "nested function modified enclosing var");
  return 0;
}
