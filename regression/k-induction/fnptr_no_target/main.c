extern int nondet_int(void);

int main(void)
{
  void (*f)(void) = 0;
  if (nondet_int())
    f();

  int n = nondet_int();
  __ESBMC_assume(n > 0);

  int x = 0;
  for (int i = 0; i < n; i++)
    x = 0;

  __ESBMC_assert(x == 0, "x stays zero across all iterations");
  return 0;
}
