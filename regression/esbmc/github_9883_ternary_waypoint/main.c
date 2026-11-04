// #9883: a ternary branching waypoint points at its ?, parenthesised or not.
extern int __VERIFIER_nondet_int(void);
int f(void) { return 2; }

int main(void)
{
  int x = __VERIFIER_nondet_int();
  int y = (x > 5)
    ? f()
    : 0;
  int z = ((x > 7)
    ? f()
    : 0);
  __ESBMC_assert(y + z == 0, "reached");
  return 0;
}
