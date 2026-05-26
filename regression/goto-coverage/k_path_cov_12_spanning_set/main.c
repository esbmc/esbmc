// Phase-2 spanning-set scoring with deeper subsumption (issue #4335
// PR1, Marré-Bertolino, IEEE TSE 29(11), 2003).
//
// Three correlated branches at N=3. Each subsequent branch is fully
// determined by `(a > 0)`, so out of the 14 emitted goals (2 depth-1 +
// 4 depth-2 + 8 depth-3) only 6 are reachable — 2 at each prefix
// length. Phase-1 would report 6/14 ≈ 42.86%.
//
// Spanning-set scoring restricts both numerator and denominator to
// the maximal subsumption-order elements. Every depth-1 and depth-2
// atom-set is a proper subset of at least one depth-3 atom-set, so
// the 6 subsumed shorter-prefix goals leave the spanning set, and
// only the 8 depth-3 maximal emissions remain. Of those 8, just the
// 2 consistent depth-3 paths are reachable, yielding 2 / 8 = 25%
// — the exact spanning-set coverage when correlated branches keep
// most maximal goals genuinely unreachable.
//
// Differs from test 4 (which uses N=2 and reaches 50%) by exercising
// the spanning-set machinery on a deeper partial-coverage case.
int main()
{
  int a;
  int b;
  int c;
  if (a > 0)
    b = 1;
  else
    b = 0;

  if (b)
    c = 1;
  else
    c = 0;

  if (c)
    a = a;
  else
    a = -a;

  return a;
}
