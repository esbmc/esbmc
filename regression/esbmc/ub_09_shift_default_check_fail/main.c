/*
 * Regression test for GitHub issue #2789.
 * The shift UB check should be enabled by default so that a nondet
 * shift distance is flagged without requiring --ub-shift-check.
 *
 * According to C99 6.5.7: "If the value of the right operand is
 * negative or is greater than or equal to the width of the promoted
 * left operand, the behavior is undefined."
 */

int nondet_int();

int main()
{
  int a = nondet_int();
  int r = 0 >> a;
  return 0;
}
