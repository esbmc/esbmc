// Infeasible-prefix scenario: the second branch is fully determined by
// the first, so only 2 of the 4 (prefix × direction) combinations are
// reachable. Demonstrates k-path coverage's natural under-coverage when
// branches are correlated; spanning-set scoring (Marré & Bertolino, IEEE
// TSE 2003) is Phase 2.
int main()
{
  int a;
  int taken;
  if (a > 0)
    taken = 1;
  else
    taken = 0;

  if (taken)
    a = a;
  else
    a = -a;

  return a;
}
