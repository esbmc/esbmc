// #9883: a ternary is located at its `?`, not where the expression starts.
extern int __VERIFIER_nondet_int(void);
int f(void) { return 2; }

int main(void)
{
  int x = __VERIFIER_nondet_int();
  int y = (x > 5)
    ? f()
    : 0;
  __ESBMC_assert(y == 0, "reached");
  return 0;
}
