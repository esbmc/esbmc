// Regression for #4330 review (LukeW1999): when the same guard appears
// multiple times in the prefix sliding window, mask combinations that
// assign opposing polarities to it produce chained contradictions like
// `g ∧ q ∧ ¬g`. ESBMC's simplifier folds the 2-term case but not the
// chained one, so without a contradiction check at instrumentation time
// these tautological `assert(¬(g ∧ q ∧ ¬g))` assertions sat permanently
// uncovered, inflating the denominator. Three sequential branches on
// the same guard with k=3 should yield exactly 6 reachable witnesses
// (2 per branch), all covered.
int main()
{
  int a;
  if (a > 0)
    ;
  else
    ;
  if (a > 0)
    ;
  else
    ;
  if (a > 0)
    ;
  else
    ;
}
