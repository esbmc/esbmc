// Phase-2 spanning-set scoring with deeper subsumption (issue #4335
// PR1, Marré-Bertolino, IEEE TSE 29(11), 2003).
//
// Three correlated branches at N=3. Each subsequent branch is fully
// determined by `(a > 0)`, so out of the 14 emitted goals (2 depth-1 +
// 4 depth-2 + 8 depth-3) only 6 are feasible — 2 at each prefix
// length. Phase-1 reports 6/14 ≈ 42.86%.
//
// Spanning-set scoring drops the 6 subsumed shorter-prefix goals
// (every depth-1 and depth-2 atom-set is a proper subset of at least
// one depth-3 atom-set), leaving 8 maximal depth-3 emissions in the
// denominator. Reaching the 2 feasible depth-3 witnesses plus their
// 4 reached depth-{1,2} subsumed prefixes yields 6 / 8 = 75% — a
// strictly tighter lower bound than Phase-1 without ever exceeding
// the true coverage of 25% (only 2 of 8 maximal goals are reachable).
//
// Differs from test 4 (which uses N=2 and reaches 100%) by exercising
// the spanning-set machinery on a *partial* coverage case where some
// maximal goals stay genuinely uncovered.
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
