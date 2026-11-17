// The return from f() put back y's value from before the call after ext()
// had already forgotten it.
int ext(void);
void f(void)
{
}
int main()
{
  int y = 0;
  f();
  y = ext();
  __ESBMC_assert(y == 0, "y");
}
