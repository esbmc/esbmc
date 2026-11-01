// #9689: a branch on a nondet value keeps its branching waypoint.
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int x = __VERIFIER_nondet_int();
  int y = 0;
  if (x > 5)
    y = 1;
  __ESBMC_assert(y == 0, "reached");
  return 0;
}
