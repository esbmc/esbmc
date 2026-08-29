/* `f(x)` and `(&f)(x)` reach goto_convert with the same shape; only the
   implicit bit tells the sugar from a user-written function pointer, and a
   direct call must not become a dereference of it. */
static int callee(int x)
{
  return x + 1;
}

int main(void)
{
  __ESBMC_assert(callee(1) == 2, "a direct call stays direct");
  __ESBMC_assert((&callee)(1) == 2, "and so does one through a written &f");
  return 0;
}
