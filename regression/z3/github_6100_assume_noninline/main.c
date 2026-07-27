// A function whose loop trip count depends on the quantified variable cannot be
// summarized into a pure quantifier body, so the forall in the assume must be
// diagnosed rather than silently mismodelled (which would prune all paths and
// mask the assert below).
int impure(int a)
{
  int s = 0;
  for (int i = 0; i < a; i++)
    s += i;
  return s;
}

int main()
{
  int x;
  __ESBMC_assume(__ESBMC_forall(&x, impure(x) >= 0));
  __ESBMC_assert(0, "unreachable when the assume is mismodelled");
}
