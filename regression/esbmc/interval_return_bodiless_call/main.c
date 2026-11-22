int ext(void);
void f(void)
{
}
int main()
{
  int x = 3, y = 0;
  f();
  y = ext();
  __ESBMC_assert(x == 3, "x");
}
