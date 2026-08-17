/* A function designator used as a value is sugar for &f. The two code types
   here differ in argument_names alone, which C11 6.7.6.3p15 says is not a
   difference at all -- casting between them would be a divergence. */
static int callee(int x)
{
  return x + 1;
}

static int apply(int (*f)(int), int v)
{
  return f(v);
}

int main(void)
{
  __ESBMC_assert(apply(callee, 1) == 2, "a function designator decays");
  return 0;
}
