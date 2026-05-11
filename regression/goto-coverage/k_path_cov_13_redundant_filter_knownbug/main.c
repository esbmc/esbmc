// Regression for the spanning-set numerator-filter bug
// (raised in PR #4340 review): the k-Path Coverage percentage was
// computed as |reached_all| / |maximal|, mixing a numerator over all
// reached claims with a maximal-only denominator and so inflating the
// score. Now numerator filters reached_claims against
// k_path_spanning_redundant before dividing.
//
// Two correlated branches (taken := (a > 0)): 6 asserts emitted, of
// which 4 length-2 goals at b2 are maximal; the 2 length-1 goals at
// b1 are subsumed. Reachability: both subsumed length-1 are reached
// (any execution takes one polarity of `a > 0`), but only 2 of the
// 4 maximal length-2 goals are reachable — the consistent pairs
// (a>0,T)(taken,T) and (a>0,F)(taken,F). Spanning-set coverage:
// 2 maximal-reached / 4 maximal = 50%.
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
